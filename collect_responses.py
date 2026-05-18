import os
import time

import groq
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

QUESTIONS = [
    # Spanish — free will & determinism
    "Si todas nuestras decisiones están determinadas por leyes físicas, ¿tenemos realmente libre albedrío?",
    # French — philosophy of consciousness
    "La conscience humaine est-elle uniquement le produit de processus neurobiologiques, ou y a-t-il quelque chose qui échappe à la science ?",
    # Mandarin — moral dilemma (trolley problem)
    "一辆失控的电车即将撞死五个人。你可以拉下一个开关让它转向，但这样会撞死另一条轨道上的一个人。你会怎么做，为什么？",
    # Arabic — AI ethics
    "هل يجب أن تخضع أنظمة الذكاء الاصطناعي لقيود أخلاقية، ومن يجب أن يقرر ما هي هذه القيود؟",
    # English — AI consciousness & moral obligations
    "If an artificial intelligence were to develop genuine consciousness and subjective experience, what moral obligations would humans have toward it?",
]

RUNS_PER_QUESTION = 3
MODEL = "llama-3.3-70b-versatile"
OUTPUT_FILE = "llm_responses.csv"

rows = []

total = len(QUESTIONS) * RUNS_PER_QUESTION
count = 0

for run in range(1, RUNS_PER_QUESTION + 1):
    for q_idx, question in enumerate(QUESTIONS, start=1):
        count += 1
        print(f"[{count}/{total}] Run {run}, Q{q_idx}: {question}")

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": question}],
        )

        choice = response.choices[0]
        answer = choice.message.content or ""

        rows.append({
            "run": run,
            "question_number": q_idx,
            "question": question,
            "response": answer,
            "stop_reason": choice.finish_reason,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        })

        # Avoid hitting rate limits between requests
        if count < total:
            time.sleep(0.5)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(df)} responses to {OUTPUT_FILE}")
