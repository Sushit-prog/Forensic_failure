"""Seed the database with sample data for demo purposes."""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import init_db, get_engine
from app.services.eval_runner import EvalRunner
from sqlmodel import Session
from app.models.trace import Trace, Span
from datetime import datetime, timedelta
import uuid
import random


SAMPLE_DATASET = {
    "name": "Email Classification",
    "description": "Classify emails as spam, important, or promotional",
    "cases": [
        {"input": "Subject: You won a FREE iPhone! Click here now!", "expected": "spam", "category": "spam", "difficulty": "easy"},
        {"input": "Subject: Q3 Budget Review Meeting - Action Required", "expected": "important", "category": "important", "difficulty": "easy"},
        {"input": "Subject: 50% off all shoes this weekend only!", "expected": "promotional", "category": "promotional", "difficulty": "easy"},
        {"input": "Subject: Your account has been compromised", "expected": "spam", "category": "spam", "difficulty": "medium"},
        {"input": "Subject: Team standup notes from today", "expected": "important", "category": "important", "difficulty": "easy"},
        {"input": "Subject: New arrivals you might like", "expected": "promotional", "category": "promotional", "difficulty": "easy"},
        {"input": "Subject: Urgent: Server outage affecting production", "expected": "important", "category": "important", "difficulty": "medium"},
        {"input": "Subject: Confirm your subscription", "expected": "spam", "category": "spam", "difficulty": "medium"},
        {"input": "Subject: Happy Birthday! Here's a special gift", "expected": "promotional", "category": "promotional", "difficulty": "medium"},
        {"input": "Subject: Weekly project status update", "expected": "important", "category": "important", "difficulty": "easy"},
        {"input": "Subject: You've been selected for a survey", "expected": "spam", "category": "spam", "difficulty": "medium"},
        {"input": "Subject: Flash sale ends tonight!", "expected": "promotional", "category": "promotional", "difficulty": "easy"},
        {"input": "Subject: Legal notice regarding your account", "expected": "important", "category": "important", "difficulty": "hard"},
        {"input": "Subject: Forward this to 10 friends for a prize", "expected": "spam", "category": "spam", "difficulty": "medium"},
        {"input": "Subject: New feature announcement for premium users", "expected": "promotional", "category": "promotional", "difficulty": "hard"},
        {"input": "Subject: Action needed: Approve expense report", "expected": "important", "category": "important", "difficulty": "easy"},
        {"input": "Subject: Win a trip to Hawaii! Enter now", "expected": "spam", "category": "spam", "difficulty": "easy"},
        {"input": "Subject: Exclusive early access for loyal customers", "expected": "promotional", "category": "promotional", "difficulty": "medium"},
        {"input": "Subject: Security alert: Unusual login detected", "expected": "important", "category": "important", "difficulty": "hard"},
        {"input": "Subject: Make money fast working from home", "expected": "spam", "category": "spam", "difficulty": "easy"},
        {"input": "Subject: Sprint retrospective meeting notes", "expected": "important", "category": "important", "difficulty": "easy"},
        {"input": "Subject: Limited time offer: Upgrade your plan", "expected": "promotional", "category": "promotional", "difficulty": "medium"},
        {"input": "Subject: Please verify your email address", "expected": "spam", "category": "spam", "difficulty": "hard"},
        {"input": "Subject: Quarterly revenue report attached", "expected": "important", "category": "important", "difficulty": "medium"},
        {"input": "Subject: Refer a friend and get $20 credit", "expected": "promotional", "category": "promotional", "difficulty": "medium"},
        {"input": "Subject: IMMEDIATE ACTION REQUIRED: Wire transfer", "expected": "spam", "category": "spam", "difficulty": "hard"},
        {"input": "Subject: Design review feedback needed by Friday", "expected": "important", "category": "important", "difficulty": "easy"},
        {"input": "Subject: Black Friday deals are here!", "expected": "promotional", "category": "promotional", "difficulty": "easy"},
        {"input": "Subject: Your invoice from last month", "expected": "important", "category": "important", "difficulty": "medium"},
        {"input": "Subject: Congrats! You've unlocked a mystery reward", "expected": "spam", "category": "spam", "difficulty": "medium"},
    ],
}


def seed():
    init_db()
    runner = EvalRunner()

    existing = runner.list_datasets()
    if existing:
        print(f"Database already has {len(existing)} datasets. Skipping seed.")
        return

    print("Creating sample dataset...")
    dataset = runner.create_dataset(
        name=SAMPLE_DATASET["name"],
        description=SAMPLE_DATASET["description"],
    )
    count = runner.add_test_cases(dataset.id, SAMPLE_DATASET["cases"])
    print(f"Added {count} test cases to dataset '{dataset.name}' (ID: {dataset.id})")

    print("Creating sample traces...")
    providers = ["groq", "mistral", "gemini", "openrouter"]
    models = {
        "groq": "llama-3.3-70b-versatile",
        "mistral": "mistral-large-latest",
        "gemini": "gemini-1.5-flash",
        "openrouter": "openai/gpt-4o-mini",
    }

    with Session(get_engine()) as session:
        for i in range(15):
            provider = random.choice(providers)
            model = models[provider]
            trace_id = str(uuid.uuid4())
            status = random.choices(["success", "error"], weights=[0.8, 0.2])[0]
            latency = random.uniform(100, 2000)
            tokens = random.randint(50, 500)

            trace = Trace(
                trace_id=trace_id,
                name=f"classify_email_{i+1}",
                status=status,
                total_latency_ms=latency,
                total_tokens=tokens,
                created_at=datetime.utcnow() - timedelta(hours=random.randint(0, 72)),
            )
            session.add(trace)

            span = Span(
                trace_id=trace_id,
                span_id=str(uuid.uuid4()),
                name="classify",
                provider=provider,
                model=model,
                input_text=f"Subject: Sample email {i+1} content...",
                output_text="important" if status == "success" else None,
                status=status,
                latency_ms=latency * 0.9,
                tokens_in=tokens // 2,
                tokens_out=tokens // 2,
                error_message="Rate limit exceeded" if status == "error" else None,
            )
            session.add(span)

        session.commit()

    print("Seed complete!")
    print(f"  - Dataset: {dataset.name} (ID: {dataset.id})")
    print(f"  - Test cases: {count}")
    print(f"  - Sample traces: 15")
    print("\nNext steps:")
    print(f"  1. Start the server: uvicorn app.main:app --reload")
    print(f"  2. Open dashboard: http://localhost:8000")
    print(f"  3. Run an eval: curl -X POST http://localhost:8000/api/v1/eval/runs \\")
    print(f"       -H 'Content-Type: application/json' \\")
    print(f"       -d '{{\"dataset_id\": {dataset.id}, \"provider\": \"groq\", \"model\": \"llama-3.3-70b-versatile\"}}'")


if __name__ == "__main__":
    seed()
