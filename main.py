"""
main.py

A real chat endpoint that enforces the token threshold guard BEFORE calling
Groq. This is the "live flow" the guide's title requires -- not a unit test
of token_guard.py in isolation, but the guard sitting in the actual request
path where a founder's real conversation would grow past the limit.

Run:
    uvicorn main:app --reload --port 8000

Requires:
    GROQ_API_KEY set in the environment (see .env.example)
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

from token_guard import check_threshold, get_actual_groq_tokens, MODEL_REGISTRY

app = FastAPI(title="Token Threshold Guard")

GROQ_MODEL = "llama-3.1-8b-instant"
THRESHOLD_PCT = float(os.environ.get("TOKEN_THRESHOLD_PCT", "80.0"))


class ChatRequest(BaseModel):
    messages: list[dict]
    model: str = GROQ_MODEL


class ChatResponse(BaseModel):
    reply: str
    predicted_tokens: int
    actual_tokens: int
    prediction_error_pct: float
    threshold_check: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not set. See .env.example.",
        )

    # --- Step 1: predict BEFORE any API call is made ---
    threshold_result = check_threshold(request.messages, request.model, THRESHOLD_PCT)

    if threshold_result["over_threshold"]:
        # This is the real trigger. No API call is made past this point.
        # A production system would truncate history here; for this guide,
        # we block and report why, since the point is to OBSERVE the
        # threshold firing, not silently paper over it.
        raise HTTPException(
            status_code=413,
            detail={
                "message": "Conversation exceeds token threshold. Truncate history before retrying.",
                "predicted_tokens": threshold_result["predicted_tokens"],
                "pct_of_window": threshold_result["pct_of_window"],
                "threshold_pct": THRESHOLD_PCT,
            },
        )

    # --- Step 2: make the real call ---
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    completion = client.chat.completions.create(
        model=request.model,
        messages=request.messages,
    )

    # --- Step 3: compare prediction to ground truth ---
    actual_tokens = completion.usage.prompt_tokens
    predicted_tokens = threshold_result["predicted_tokens"]
    error_pct = round(abs(predicted_tokens - actual_tokens) / actual_tokens * 100, 2)

    return ChatResponse(
        reply=completion.choices[0].message.content,
        predicted_tokens=predicted_tokens,
        actual_tokens=actual_tokens,
        prediction_error_pct=error_pct,
        threshold_check=threshold_result,
    )