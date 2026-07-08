"""
token_guard.py

Counts tokens for a chat message array across three models with two different
tokenizer families, then enforces a context-window threshold before an API
call is made.

Model -> tokenizer mapping (this is the load-bearing fact of the whole module):
  gpt-4o          -> o200k_base   (tiktoken, EXACT count)
  gpt-3.5-turbo   -> cl100k_base  (tiktoken, EXACT count)
  llama-3.1-8b-instant (Groq)  -> Llama's own tokenizer, which tiktoken does
                                    NOT have. tiktoken can only PREDICT this
                                    count by approximating with cl100k_base.
                                    The real count only exists after Groq's
                                    API responds with usage.prompt_tokens.

This split is not a bug to hide. It is the concept. A function that reports
one "token count" number for all three models is silently wrong for one of
them, and that's the exact class of bug this file exists to prevent.
"""

import os
import tiktoken
from openai import OpenAI

# ---------------------------------------------------------------------------
# Model registry: single source of truth for which counting strategy applies
# to which model. Add a model here, not by branching logic elsewhere.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "gpt-4o": {
        "family": "openai",
        "tiktoken_encoding": "o200k_base",
        "exact": True,
        "context_window": 128_000,
    },
    "gpt-3.5-turbo": {
        "family": "openai",
        "tiktoken_encoding": "cl100k_base",
        "exact": True,
        "context_window": 16_385,
    },
    "llama-3.1-8b-instant": {
        "family": "groq",
        "tiktoken_encoding": "cl100k_base",  # approximation only, not exact
        "exact": False,
        "context_window": 131_072,
    },
}

# Per-message overhead tokens. OpenAI's chat format wraps every message with
# role/name/separator tokens that aren't part of the visible text. This is
# documented by OpenAI for their own models; for Groq/Llama it is an
# approximation carried over from the same counting scheme, not a verified
# Llama-specific constant.
TOKENS_PER_MESSAGE = 3
TOKENS_PER_NAME = 1
TOKENS_PER_REPLY_PRIMING = 3


def count_tokens_tiktoken(messages: list[dict], encoding_name: str) -> int:
    """
    Exact-or-approximate token count for a chat messages array, using a
    specific tiktoken encoding. Whether this number is EXACT or an
    APPROXIMATION depends entirely on whether encoding_name is the real
    tokenizer for the target model (see MODEL_REGISTRY).
    """
    encoding = tiktoken.get_encoding(encoding_name)
    total = 0
    for message in messages:
        total += TOKENS_PER_MESSAGE
        for key, value in message.items():
            total += len(encoding.encode(str(value)))
            if key == "name":
                total += TOKENS_PER_NAME
    total += TOKENS_PER_REPLY_PRIMING
    return total


def predict_tokens(messages: list[dict], model: str) -> dict:
    """
    Predict token count for `model` BEFORE any API call is made.
    Returns a dict that is explicit about whether the number is exact.
    """
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model}. Add it to MODEL_REGISTRY.")

    spec = MODEL_REGISTRY[model]
    predicted = count_tokens_tiktoken(messages, spec["tiktoken_encoding"])

    return {
        "model": model,
        "predicted_tokens": predicted,
        "is_exact": spec["exact"],
        "context_window": spec["context_window"],
        "pct_of_window": round(predicted / spec["context_window"] * 100, 2),
    }


def get_actual_groq_tokens(messages: list[dict], model: str, api_key: str) -> int:
    """
    Ground truth for a Groq model: make the real call, read
    usage.prompt_tokens back from the response. This is the only source of
    truth for non-OpenAI models -- tiktoken's number for them is always a
    prediction, never a fact, until this function has run.
    """
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1,  # we only need the usage block, not a full completion
    )
    return response.usage.prompt_tokens


def check_threshold(messages: list[dict], model: str, threshold_pct: float = 80.0) -> dict:
    """
    The actual guard. Predicts tokens for `model`, compares against
    threshold_pct of that model's context window, and returns a decision
    the calling code can act on BEFORE spending an API call.
    """
    prediction = predict_tokens(messages, model)
    over_threshold = prediction["pct_of_window"] >= threshold_pct

    return {
        **prediction,
        "threshold_pct": threshold_pct,
        "over_threshold": over_threshold,
        "action": "BLOCK_AND_TRUNCATE" if over_threshold else "PROCEED",
    }


if __name__ == "__main__":
    # A single conversation, counted across all three models, to make the
    # divergence in Section 2's model directly observable.
    sample_conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain how token counting works in production LLM systems."},
        {"role": "assistant", "content": "Token counting matters because every model has a fixed context window, and going over it causes a hard API error rather than a graceful truncation."},
        {"role": "user", "content": "What happens if I go over the limit?"},
    ]

    print("=" * 70)
    print("PREDICTED TOKEN COUNTS (before any API call)")
    print("=" * 70)
    for model in MODEL_REGISTRY:
        result = predict_tokens(sample_conversation, model)
        exactness = "EXACT" if result["is_exact"] else "APPROXIMATION"
        print(
            f"{model:24s} {result['predicted_tokens']:5d} tokens  "
            f"({exactness}, {result['pct_of_window']}% of {result['context_window']}-token window)"
        )