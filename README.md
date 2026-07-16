# Socratica — Multimodal AI STEM Diagram Feedback System

> **Evaluate student STEM diagrams. Generate Socratic formative feedback. Run fully offline.**

A three-stage multimodal AI pipeline that takes a student's hand-drawn or scanned STEM diagram, extracts a structured scene graph via a local Vision-Language Model (VLM), reasons over it with an LLM, and delivers Socratic feedback tailored to detected misconceptions.

Built for **resource-constrained school environments** — runs fully offline using [Ollama](https://ollama.com), requires no paid API keys, and works on CPU-only hardware.

---

## Architecture

```
Student Upload (JPG/PNG/WebP)
        │
        ▼
┌─────────────────────────┐
│  Image Preprocessor     │  Letterbox resize → 448×448 RGB PNG
│  (image_preprocessor.py)│  EXIF strip, format normalization
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Stage 1 — VLM          │  Ollama llava:13b
│  Perception Engine      │  Extracts scene graph:
│  (stage1_perception.py) │  elements + spatial relations
└────────────┬────────────┘
             │  Scene Graph JSON
             ▼
┌─────────────────────────┐
│  Stage 2 — LLM          │  Ollama llama3.1:8b
│  Reasoning + RAG        │  TF-IDF search over AAAS/MaLT
│  (stage2_reasoning.py)  │  misconception library
└────────────┬────────────┘
             │  Structured Feedback Items
             ▼
┌─────────────────────────┐
│  Stage 3 — Rendering    │  Overlay coordinates + heatmaps
│  (stage3_rendering.py)  │  Longitudinal tracking
└────────────┬────────────┘
             │
             ▼
      FastAPI REST + WebSocket API
      Interactive HTML5 Dashboard
```

---

## Requirements

| Software        | Version   | Notes                            |
|-----------------|-----------|----------------------------------|
| Python          | 3.10+     |                                  |
| Ollama          | latest    | Install from ollama.com          |
| Docker          | 24+       | Optional, for containerized run  |
| RAM             | ≥8 GB     | 16 GB recommended for llava:13b  |
| GPU             | Optional  | Dramatically speeds up inference |

---

## Quick Start (Local)

### 1. Clone and install

```bash
git clone https://github.com/muhammadhussain-2009/Multimodal-LLM-VLM-Architecture-.git
cd Multimodal-LLM-VLM-Architecture-

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env
# Edit .env if you need to change Ollama URL or models
```

### 3. Pull Ollama models

```bash
# Install Ollama first: https://ollama.com/download
ollama serve                    # Start Ollama (keep running in background)
ollama pull llava:13b           # VLM model (~8 GB)
ollama pull llama3.1:8b         # LLM model (~5 GB)
```

### 4. Start the server

```bash
python run.py
# App available at: http://localhost:8000
```

---

## Docker Deployment

```bash
# Build and start all services
docker compose up --build -d

# Pull models into the Ollama container
docker exec socratica-ollama ollama pull llava:13b
docker exec socratica-ollama ollama pull llama3.1:8b

# View logs
docker compose logs -f app
```

---

## API Reference

### `POST /api/analyze`
Upload a diagram for analysis. Returns `202 Accepted` immediately.

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@my_diagram.png" \
  -F "student_id=student_001" \
  -F "context=Grade 9 Physics"
```

Response:
```json
{
  "job_id": "abc123",
  "status": "queued",
  "ws_url": "/ws/job/abc123",
  "poll_url": "/api/job/abc123"
}
```

### `GET /api/job/{job_id}`
Poll job status and retrieve results when done.

### `WS /ws/job/{job_id}`
Real-time progress updates via WebSocket.

### `GET /api/analytics?student_id=xxx`
Get top misconceptions by frequency for longitudinal tracking.

### `POST /api/dataset/ingest`
Trigger background ingestion of the full `lmms-lab/ai2d` dataset.

### `GET /api/health`
Liveness check — also reports Ollama model availability.

---

## Dataset

This system uses the **[AI2D dataset](https://huggingface.co/datasets/lmms-lab/ai2d)** from the Allen Institute for AI — 4,903 science diagrams with Q&A annotations covering biology, chemistry, physics, and earth science.

To ingest the dataset:

```bash
# Via API (preferred — runs in background)
curl -X POST "http://localhost:8000/api/dataset/ingest?max_examples=5000"

# Or directly via CLI
python -m backend.data_pipeline --max-examples 5000 --output-dir ./data
```

Shards are saved to `data/processed/` as JSON files, ready for LoRA fine-tuning.

---

## Fine-Tuning

LoRA fine-tuning configuration is in `training_config.yaml`. Run training with:

```bash
python backend/train_lora.py
```

---

## Project Structure

```
Multimodal LLM/
├── backend/
│   ├── main.py               # FastAPI app + API endpoints
│   ├── stage1_perception.py  # VLM perception engine (llava:13b)
│   ├── stage2_reasoning.py   # LLM reasoning + RAG (llama3.1:8b)
│   ├── stage3_rendering.py   # Feedback rendering + overlays
│   ├── data_pipeline.py      # Real AI2D dataset ingestion
│   ├── image_preprocessor.py # Robust image normalization
│   ├── database.py           # Async SQLite job + feedback DB
│   ├── train_lora.py         # LoRA fine-tuning script
│   ├── evaluate.py           # Evaluation metrics
│   └── stats_analysis.py     # Statistical analysis tools
├── frontend/
│   └── index.html            # Interactive dashboard
├── data/
│   └── processed/            # Sharded AI2D training data
├── uploads/                  # Temporary uploaded diagrams
├── training_config.yaml      # LoRA hyperparameters
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Models Used

| Model          | Role         | Size  | Notes                              |
|----------------|--------------|-------|------------------------------------|
| `llava:13b`    | VLM (Stage 1)| ~8 GB | Best open spatial diagram analysis |
| `llama3.1:8b`  | LLM (Stage 2)| ~5 GB | Strong reasoning + Socratic prompts|

Both models run locally via Ollama. No API keys required.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Pull requests welcome. Please open an issue first to discuss major changes.
