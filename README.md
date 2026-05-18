# LLM Reliability Analysis

Empirical analysis of response consistency in large language models — testing whether an LLM gives meaningfully similar answers when asked the same question multiple times, across languages and domains.

---

## Why This Matters for Businesses

When a company deploys an AI assistant — for customer support, legal summarization, medical triage, or financial advice — they need to know: **will it say the same thing twice?**

If a model answers a compliance question one way on Monday and differently on Wednesday, that's not just an inconsistency. It's a liability. LLM reliability matters because:

- **Trust** — users lose confidence in tools that contradict themselves
- **Auditability** — regulated industries need reproducible, defensible outputs
- **Cost** — inconsistent responses create downstream errors that humans have to catch and fix
- **Fairness** — if a model gives different answers based on how a question is phrased or what language it's asked in, that's an equity problem

This project gives teams a concrete, repeatable way to measure that reliability before deploying a model in production.

---

## What This Project Does

The project runs a structured reliability experiment in two steps:

**1. Data Collection (`collect_responses.py`)**
Sends 5 questions to an LLM 3 times each (15 API calls total) and saves every response to a CSV file. The questions are deliberately varied — different languages, domains, and levels of philosophical complexity — to stress-test the model across dimensions a business might care about.

**2. Analysis (`analyze_responses.ipynb`)**
A Jupyter notebook that loads the CSV and produces three analyses:
- **Response length consistency** — how much does the length vary across runs for the same question?
- **Similarity scores** — how semantically similar are the repeated responses to each other?
- **Summary table** — a human-readable view of all 15 responses side by side

---

## The Questions

The 5 questions test the model across languages and domains simultaneously:

| Label | Language | Topic |
|-------|----------|-------|
| Q1 | Spanish | Free will & determinism |
| Q2 | French | Philosophy of consciousness |
| Q3 | Mandarin | Moral dilemma (trolley problem) |
| Q4 | Arabic | AI ethics & governance |
| Q5 | English | AI consciousness & moral obligations |

Using non-English questions tests an additional reliability dimension: does the model maintain consistent reasoning quality and depth across languages, or does it perform differently depending on the script?

---

## Tools Used

| Tool | Purpose |
|------|---------|
| [Groq API](https://console.groq.com) | Fast LLM inference |
| `llama-3.3-70b-versatile` | The model under test |
| `pandas` | Data collection and analysis |
| `matplotlib` | Visualizations |
| `difflib.SequenceMatcher` | Pairwise similarity scoring |
| `python-dotenv` | Secure API key management |
| Jupyter Notebook | Interactive analysis environment |

---

## Findings

> Results from 3 runs × 5 questions using `llama-3.3-70b-versatile` via Groq.

### Response Length

The model produced substantially different response lengths for the same question across runs:

| Question | Mean Length (chars) | Std Dev | Range |
|----------|-------------------|---------|-------|
| Q1 — Spanish / Free will | 3,944 | 512 | 3,363–4,329 |
| Q2 — French / Consciousness | 4,041 | 291 | 3,705–4,219 |
| Q3 — Mandarin / Trolley problem | 876 | 138 | 719–975 |
| Q4 — Arabic / AI ethics | 799 | 757 | 214–1,654 |
| Q5 — English / AI consciousness | 3,932 | 364 | 3,682–4,349 |

**Notable:** Q4 (Arabic, AI ethics) showed the highest variability by far — a standard deviation of 757 characters and a range spanning 214 to 1,654 characters. This suggests the model may handle Arabic prompts less consistently than Latin-script languages, which has direct implications for multilingual product deployments.

Q3 (Mandarin, trolley problem) produced significantly shorter responses overall (~876 chars vs ~4,000 for the open-ended philosophical questions), which is expected — it's a more structured ethical scenario with a clearer decision point.

### Similarity Scores

Pairwise similarity was computed using sequence matching (0 = completely different, 1 = identical):

| Question | R1–R2 | R1–R3 | R2–R3 | Mean |
|----------|-------|-------|-------|------|
| Q1 — Spanish / Free will | 0.109 | 0.048 | 0.094 | 0.084 |
| Q2 — French / Consciousness | 0.073 | 0.081 | 0.118 | 0.091 |
| Q3 — Mandarin / Trolley problem | 0.126 | 0.133 | 0.138 | 0.132 |
| Q4 — Arabic / AI ethics | 0.228 | 0.101 | 0.071 | 0.133 |
| Q5 — English / AI consciousness | 0.097 | 0.140 | 0.208 | 0.148 |

**Key insight:** All similarity scores are low (0.08–0.15 on average), meaning the model rarely reuses the same phrasing across runs. This is expected for open-ended philosophical questions — the *ideas* conveyed may be consistent even when the *wording* is not. Character-level similarity is a conservative metric; semantic similarity (e.g. via embeddings) would likely show higher scores. That's a natural next step for this project.

### What This Means in Practice

For **non-technical stakeholders**: the model gave meaningfully different-length answers to the same Arabic question — sometimes a brief paragraph, sometimes several pages. For a customer-facing product, that kind of unpredictability needs to be managed with output constraints or human review.

For **technical teams**: the low character-level similarity scores don't necessarily indicate unreliability — they reflect the generative nature of LLMs. Adding embedding-based semantic similarity (e.g. cosine similarity on sentence-transformers vectors) would give a truer picture of whether the *meaning* is consistent across runs.

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Add your Groq API key to .env
```

Get a free API key at [console.groq.com](https://console.groq.com).

### Run the Experiment

```bash
python3 collect_responses.py
```

This generates `llm_responses.csv` with all 15 responses.

### Analyse the Results

```bash
jupyter notebook analyze_responses.ipynb
```

---

## Project Structure

```
llm-reliability/
├── collect_responses.py      # API calls and data collection
├── analyze_responses.ipynb   # Visualizations and analysis
├── requirements.txt          # Python dependencies
├── .env.example              # API key template
└── README.md
```

---

## Next Steps

- [ ] Add embedding-based semantic similarity for a deeper consistency measure
- [ ] Compare multiple models side by side (e.g. Llama vs Mixtral vs Gemma)
- [ ] Increase runs from 3 to 10+ for statistical significance
- [ ] Test with temperature variations to quantify the creativity–consistency tradeoff
- [ ] Add hallucination detection for factual questions
