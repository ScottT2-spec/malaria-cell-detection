# 🦟 MalariaAI — Intelligent Malaria Screening Platform

> CNN detection + LLM clinical intelligence + RAG + continuous learning from doctor feedback

---

## What It Does

MalariaAI is an AI-powered clinical decision support tool that combines **computer vision** with **large language model intelligence** to assist healthcare workers in malaria-endemic regions.

1. **🔬 CNN Detection** — A 3-layer CNN (95.43% accuracy, trained on 27,558 NIH images) classifies blood cell microscopy images as parasitized or uninfected
2. **🧠 LLM Clinical Intelligence** — Generates clinical summaries, treatment guidance (WHO/CDC protocols), severity assessments, and patient education in multiple languages
3. **📚 RAG Pipeline** — Retrieval-Augmented Generation pulls the latest WHO/CDC guidelines and doctor corrections into every analysis
4. **🩺 Doctor Feedback Loop** — Medical professionals can correct diagnoses; corrections feed back into the RAG knowledge base in real-time, making the system smarter with every use
5. **💬 Interactive Q&A** — Healthcare workers can ask follow-up questions about diagnosis, treatment, prevention, or malaria in general
6. **📄 AI-Generated Reports** — Professional clinical reports generated automatically

## Architecture

```
[Blood Cell Image]
       ↓
  [CNN Detection] ← TensorFlow.js (runs in-browser, works offline)
       ↓
  [RAG Retrieval] ← WHO/CDC Guidelines + Doctor Corrections
       ↓
  [LLM Analysis]  ← Clinical analysis, treatment guidance, education
       ↓
  [Doctor Review]  ← Approve / Correct / Add notes
       ↓                    ↓
  [Smart Report]    [Corrections DB] → feeds back into RAG
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| CNN Model | TensorFlow/Keras → TensorFlow.js (browser) |
| LLM | Configurable (OpenAI-compatible API) |
| Backend | Python / FastAPI |
| RAG | TF-IDF retrieval (lightweight, no GPU needed) |
| Knowledge Base | WHO & CDC malaria guidelines |
| Feedback Store | JSONL (append-only) |
| Frontend | Vanilla HTML/CSS/JS (PWA, works offline) |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/ScottT2-spec/malaria-cell-detection.git
cd malaria-cell-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your API key

# 4. Run
bash scripts/run.sh
# → http://localhost:8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze` | POST | Send CNN results → get clinical analysis |
| `/api/correct` | POST | Submit doctor correction → updates RAG in real-time |
| `/api/chat` | POST | Interactive Q&A about diagnosis/treatment |
| `/api/report` | POST | Generate clinical report |
| `/api/corrections/stats` | GET | Feedback system dashboard |
| `/api/rag/status` | GET | RAG engine status |
| `/health` | GET | Service health check |

## The Continuous Learning Loop

```
Doctor uses MalariaAI → CNN scans cell → LLM analyzes
        ↑                                        ↓
   RAG gets smarter ← Doctor corrects if needed → Correction saved
```

Every correction makes the system more accurate — not by retraining the model, but by enriching the knowledge base that the LLM draws from. It's like a growing clinical memory.

## Impact

- **For rural clinics** — AI-assisted screening where trained microscopists are scarce
- **For community health workers** — Clear, simple explanations in local languages
- **For doctors** — Clinical decision support with cited WHO/CDC guidelines
- **For patients** — Understanding their results and what to do next

## Dataset

[NIH Malaria Cell Images](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) — 27,558 cell images (balanced: 13,779 parasitized + 13,779 uninfected)

## Built By

**Scott Antwi** 🇬🇭 — Student developer from Ghana, building AI for healthcare in Africa.

---

*MalariaAI is a screening tool for educational and research purposes. It is not a medical device. Always consult a healthcare professional for diagnosis and treatment.*
