# NER Fine-Tuning Pipeline

This branch contains a self-contained pipeline for generating, annotating, and preparing synthetic job descriptions for Named Entity Recognition (NER) training.

## 📁 Contents

| File | Purpose |
|------|---------|
| `generate_jobs.py` | Uses OpenRouter to generate synthetic job descriptions with labeled skills (`SKILL`). Output is in JSONL format. |
| `add_offsets.py` | Post-processes job data to calculate and inject `start` and `end` character offsets for each labeled skill entity. |
| `jobs-new.jsonl` | Raw generated job data (ID, text, and skill entity spans — no offsets). |
| `jobs-new_with_offsets.jsonl` | Final NER-ready dataset with character-level offsets added. |
| `finetune_ner.ipynb` | Jupyter notebook to train a NER model using the processed dataset. |
| `finetune.requirements.txt` | Fully pinned Python dependencies for training and preprocessing. |

---

## ⚙️ Setup

1. **Install dependencies** (recommended: in a virtual environment)

```bash
pip install -r finetune.requirements.txt
```

2. **Set up OpenRouter API key**

Create a `.env` file with:

```
OPENROUTER_API_KEY=your_key_here
```

---

## 🚀 Usage

### 1. Generate synthetic job descriptions

```bash
python generate_jobs.py
```

This creates (or appends to) `jobs-new.jsonl`, each entry including:
```json
{
  "id": "JD_001",
  "text": "Hiring backend dev with Python and Docker experience.",
  "entities": [
    { "label": "SKILL", "text": "Python" },
    { "label": "SKILL", "text": "Docker" }
  ]
}
```

### 2. Add character-level entity offsets

```bash
python add_offsets.py
```

This outputs `jobs-new_with_offsets.jsonl` with:
```json
{
  "id": "JD_001",
  "text": "...",
  "entities": [
    { "start": 28, "end": 34, "label": "SKILL", "text": "Python" }
  ]
}
```

### 3. Fine-tune the NER model

Open the `finetune_ner.ipynb` notebook to train a model using 🤗 Transformers and the processed dataset.

---

## 📌 Notes

- **Duplicates / Overlaps**: `add_offsets.py` ensures entity spans don’t overlap.
- **Resumable generation**: `generate_jobs.py` supports resuming based on existing line counts.
- **LFS note**: If files are large, consider using [Git LFS](https://git-lfs.github.com/).

---

## 🧠 Authors & License

MIT License. Created by [Your Name or Org].
