# 🇫🇷→🇬🇧 French to English Translation — NLP

> A sequence-to-sequence neural machine translation model built from scratch using LSTM encoder-decoder architecture, trained on the French-English parallel corpus.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-red?style=flat-square&logo=keras&logoColor=white)](https://keras.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 🔗 Part of the LinguaFlow Project

This repository is the **research foundation** for [LinguaFlow](https://github.com/paradise-007/linguaflow-frontend) — a full-stack multilingual translation web app.

| Component | Repository | Description |
|-----------|-----------|-------------|
| 📓 **This repo** | `French-To-English-Translation-NLP` | Original LSTM seq2seq research & notebook |
| ⚡ **API Backend** | [`linguaflow-api`](https://github.com/paradise-007/linguaflow-api) | FastAPI backend serving Helsinki-NLP MarianMT models |
| 🌐 **Web Frontend** | [`linguaflow-frontend`](https://github.com/paradise-007/linguaflow-frontend) | Vercel-hosted UI (French, English, Hindi, Gujarati) |

---

## 📖 Overview

This project explores **Neural Machine Translation (NMT)** using a classic encoder-decoder LSTM architecture. Starting from raw parallel sentence pairs, the notebook walks through every stage of an NLP pipeline — from tokenization and vocabulary building to model training and inference.

The work here directly informed the production deployment in LinguaFlow, where we moved from a custom LSTM to pretrained [Helsinki-NLP MarianMT](https://huggingface.co/Helsinki-NLP) models for better accuracy and multilingual support.

---

## 🧠 Model Architecture

```
Input Sentence (French)
        ↓
   Tokenization
        ↓
  Word Embedding Layer
        ↓
  LSTM Encoder  ──→  Context Vector (hidden state + cell state)
                              ↓
                       LSTM Decoder
                              ↓
                    Dense + Softmax Layer
                              ↓
              Output Sentence (English tokens)
```

**Key design choices:**
- **Encoder:** Single/stacked LSTM that reads the source sentence and compresses it into a fixed-size context vector
- **Decoder:** LSTM that generates the target sentence token by token, conditioned on the context vector
- **Teacher Forcing:** Used during training to stabilize gradients
- **Vocabulary:** Built from the training corpus with `<START>`, `<END>`, and `<UNK>` special tokens

---

## 📂 Repository Structure

```
French-To-English-Translation-NLP/
│
├── French_to_English_Translation.ipynb   # Main Jupyter notebook
├── fra.txt                               # French-English parallel corpus (tab-separated)
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook or JupyterLab
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/paradise-007/French-To-English-Translation-NLP.git
cd French-To-English-Translation-NLP

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook
```

### Run the Notebook

Open `French_to_English_Translation.ipynb` and run all cells. The notebook is self-contained and walks through:

1. **Data loading** — reading and parsing `fra.txt`
2. **Preprocessing** — cleaning, tokenizing, padding sequences
3. **Vocabulary building** — word-to-index mappings
4. **Model definition** — encoder-decoder LSTM with Keras
5. **Training** — with validation split and early stopping
6. **Inference** — translating new French sentences

---

## 📊 Dataset

The model is trained on the [ManyThings.org French-English parallel corpus](http://www.manythings.org/anki/) — a curated set of sentence pairs commonly used for NMT research.

| Stat | Value |
|------|-------|
| Language pair | French → English |
| Corpus format | Tab-separated `.txt` |
| Sentence pairs | ~135,000 |
| Vocabulary (FR) | ~20,000 tokens |
| Vocabulary (EN) | ~12,000 tokens |

---

## 📦 Dependencies

```txt
tensorflow >= 2.0
keras
numpy
pandas
matplotlib
scikit-learn
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📈 Results

The LSTM seq2seq model demonstrates solid performance on short-to-medium French sentences. Longer sentences with complex grammar show the inherent limitation of fixed-size context vectors — which is what motivated the move to transformer-based MarianMT models in the production LinguaFlow app.

---

## 🔭 What's Next — LinguaFlow

This notebook proved the concept. The production version at [LinguaFlow](https://github.com/paradise-007/linguaflow-frontend) extends it with:

- ✅ Pretrained Helsinki-NLP MarianMT transformer models
- ✅ 4 languages: French, English, Hindi, Gujarati
- ✅ REST API via FastAPI (with CORS)
- ✅ Web UI deployed on Vercel
- ✅ Image OCR + translation

---

## 👤 Author

**Vishv** — [@paradise-007](https://github.com/paradise-007)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built as an NLP learning project · Evolved into <a href="https://github.com/paradise-007/linguaflow-frontend">LinguaFlow</a></sub>
</div>
