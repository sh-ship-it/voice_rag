"""Benchmark Groq API latency and token generation throughput."""

import os
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
from openai import OpenAI
from pipeline.config import get_settings
from pipeline.generate import SYSTEM_PROMPT, format_context_prompt
from pipeline.retrieve import hybrid_retrieve, _REGISTRY, warmup
from pipeline.embed import embed_query

settings = get_settings()

print("=" * 76)
print("  MEASURING GROQ API INFERENCE LATENCY & THROUGHPUT (Llama 3.1 8B)")
print("=" * 76)
print(f"  Model       : {settings.groq_model}")
print(f"  Base URL    : {settings.groq_base_url}")
print(f"  Timeout (ms): {settings.groq_timeout_ms}")
print("=" * 76)

if not settings.groq_api_key:
    print("[ERROR] GROQ_API_KEY is not configured in .env!")
    sys.exit(1)

client = OpenAI(
    base_url=settings.groq_base_url,
    api_key=settings.groq_api_key,
    timeout=10.0,
)

# Pre-warm local index
warmup()

test_queries = [
    "भारत की राजधानी क्या है?",
    "विश्व का सबसे बड़ा महासागर कौन सा है?",
    "भारतीय संविधान कब लागू हुआ था?",
    "What is the capital of India?",
    "हिंदी दिवस कब मनाया जाता है?",
    "मानव शरीर में कितनी हड्डियां होती हैं?",
    "what is artificial intelligence?",
    "सौर मंडल का सबसे बड़ा ग्रह कौन सा है?",
]

print("\nRunning live Groq inference benchmark across 8 queries ...\n")

latencies = []
prompt_tokens_list = []
completion_tokens_list = []

for i, q in enumerate(test_queries, 1):
    q_emb = embed_query(q)
    ret_res = hybrid_retrieve(q_emb, q, top_k=5, enable_rerank=True)
    user_prompt = format_context_prompt(q, ret_res.chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        max_tokens=200,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    dur_ms = (time.perf_counter() - t0) * 1000.0
    latencies.append(dur_ms)

    usage = resp.usage
    p_tok = usage.prompt_tokens if usage else 0
    c_tok = usage.completion_tokens if usage else 0
    prompt_tokens_list.append(p_tok)
    completion_tokens_list.append(c_tok)

    tok_per_sec = (c_tok / (dur_ms / 1000.0)) if dur_ms > 0 else 0

    ans_snippet = resp.choices[0].message.content or ""
    # Extract clean preview
    preview = ans_snippet.replace("\n", " ")[:60]

    print(f"  [{i}/8] Latency: {dur_ms:>6.1f} ms | Tokens: {c_tok:>3} ({tok_per_sec:>5.1f} tok/s) | Q: {q[:30]}")

print("\n" + "=" * 76)
print("  GROQ LATENCY BENCHMARK RESULTS")
print("=" * 76)
print(f"  Total Queries Tested   : {len(latencies)}")
print(f"  Average Latency        : {np.mean(latencies):.2f} ms")
print(f"  P50 (Median) Latency   : {np.percentile(latencies, 50):.2f} ms")
print(f"  P70 Latency            : {np.percentile(latencies, 70):.2f} ms")
print(f"  P95 Latency            : {np.percentile(latencies, 95):.2f} ms")
print(f"  P100 (Max) Latency     : {np.max(latencies):.2f} ms")
print(f"  Avg Completion Tokens  : {np.mean(completion_tokens_list):.1f} tokens")
avg_tok_sec = np.sum(completion_tokens_list) / (np.sum(latencies) / 1000.0)
print(f"  Avg Generation Speed   : {avg_tok_sec:.1f} tokens/second")
print("=" * 76)
