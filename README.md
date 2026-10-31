# 🛒 ShopBot — E-Commerce NLP Chatbot

> A bilingual (English + Arabic) AI-powered e-commerce chatbot with semantic product search, product comparison, policy Q&A, and voice input via Whisper.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Smart Search** | Natural language understanding — say *"something to keep water cold"* and get water bottles |
| ⚖️ **Product Comparison** | Side-by-side comparison table for 2–3 products with an AI-written review summary |
| 📋 **Policy Q&A** | Instant answers on warranty, cash on delivery, and return/exchange policies |
| 🌐 **Bilingual** | Full English + Arabic support with auto language detection and RTL layout |
| 🎙️ **Voice Input** | Mic button → Whisper transcription (supports both EN and AR speech) |

---

## 📁 Project Structure

```
Chatbot/
├── backend/
│   ├── app.py              # Flask server (main entry point)
│   ├── nlp_engine.py       # Intent classifier (sentence-transformers)
│   ├── product_search.py   # FAISS semantic search
│   ├── comparison.py       # Product comparison table + summary
│   ├── policy_qa.py        # Warranty / COD / Return Q&A
│   ├── translation.py      # Language detection + translation
│   ├── transcription.py    # Whisper speech-to-text
│   └── data_loader.py      # Parquet data loader
├── data/
│   ├── catalog.parquet     # Product catalog
│   └── reviews.parquet     # Product reviews
├── frontend/
│   ├── index.html          # Storefront shell
│   ├── style.css           # Store styles
│   ├── common.js           # Shared helpers, API base, cart/compare, event bus
│   ├── app.js              # Hash router + store pages
│   └── shopbot/            # ShopBot chat widget (Shadow-DOM, EN/AR, voice)
│       ├── shopbot.js
│       └── shopbot.css
├── generate_mock_data.py   # Generates mock data (run first!)
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.10+ 
- pip
- (Optional) CUDA GPU for faster Whisper transcription

### 2. Install dependencies

```powershell
cd "c:\Users\Super Sravan\OneDrive\Desktop\Chatbot"
pip install -r requirements.txt
```

### 3. Generate mock data

```powershell
python generate_mock_data.py
```

> **Swap in real data**: Replace `data/catalog.parquet` and `data/reviews.parquet` with your Amazon Reviews 2023 files. The column names must match exactly — see schema below.

### 4. Start the server

```powershell
python backend/app.py
```

Open your browser to **http://localhost:5000** 🎉

---

## 📊 Data Schema

### `data/catalog.parquet`

| Column | Type | Description |
|---|---|---|
| `parent_asin` | str | Unique product identifier |
| `title` | str | Product title |
| `store` | str | Brand/store name |
| `price` | float | Price in USD |
| `average_rating` | float | Average star rating (1–5) |
| `rating_number` | int | Number of ratings |
| `category_path` | str | Category hierarchy (e.g., `Electronics > Headphones`) |
| `features_text` | str | Product features (used for search + policy Q&A) |
| `description_text` | str | Product description |
| `n_reviews_sampled` | int | Number of reviews sampled |

### `data/reviews.parquet`

| Column | Type | Description |
|---|---|---|
| `parent_asin` | str | Product identifier (FK to catalog) |
| `rating` | float | Review rating (1–5) |
| `title` | str | Review headline |
| `text` | str | Full review text |
| `helpful_vote` | int | Helpful votes count |
| `verified_purchase` | bool | Verified purchase flag |
| `timestamp` | int | Unix timestamp (milliseconds) |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the chat UI |
| GET | `/api/health` | Health check |
| POST | `/api/chat` | Main chat endpoint |
| POST | `/api/compare` | Compare 2–3 products by ASIN |
| POST | `/api/transcribe` | Whisper audio transcription |
| GET | `/api/products` | Paginated product list |

### POST `/api/chat`
```json
Request:  { "message": "show me wireless headphones", "session_id": "optional-uuid" }
Response: { "session_id": "...", "intent": "search_product", "response_type": "product_list",
            "text": "...", "products": [...], "comparison": null, "language": "en" }
```

### POST `/api/compare`
```json
Request:  { "asins": ["B003HPH001", "B003HPH002"], "language": "en" }
Response: { "table": { "headers": [...], "rows": [[...]] }, "summary": "..." }
```

### POST `/api/transcribe`
```
Request:  multipart/form-data, field "audio" = audio blob, optional field "model_size" = "base"
Response: { "text": "transcribed text", "language": "en", "detected_language": "en" }
```

---

## 🗣️ Example Queries

**English:**
- *"I want something to store water for drinking"* → shows water bottles
- *"Find wireless noise canceling headphones under $200"*
- *"Compare Sony WH-1000XM5 and Bose QuietComfort 45"*
- *"Is there warranty on this product?"*
- *"Can I pay with cash on delivery?"*
- *"What is the return policy?"*

**Arabic:**
- *"أريد شيئاً لتخزين المياه"* → يعرض زجاجات المياه
- *"قارن بين سامسونج وآيفون"*
- *"هل يوجد ضمان؟"*

---

## 🧠 NLP Architecture

```
User Input
    │
    ▼
Language Detection (langdetect)
    │
[Arabic] ──► Translate to English (deep-translator)
    │
    ▼
Intent Classification (sentence-transformers MiniLM + cosine similarity)
    │
    ├─► search_product / vague_search ──► FAISS Semantic Search
    ├─► compare_products ──► Comparison Table + Review Summary
    ├─► policy_warranty / policy_cod / policy_return ──► Feature Text Extraction
    └─► greeting ──► Welcome message
    │
[Arabic input] ──► Translate response back to Arabic
    │
    ▼
JSON Response → Frontend renders UI
```

---

---

## 🧩 Storefront & ShopBot widget

The static frontend is a dependency-free single-page storefront with hash routing
(`#/search`, `#/category/...`, `#/product/{asin}`, `#/compare`, `#/cart`,
`#/orders`, `#/help/{topic}`), served by the ShopBot app (`app.py`, port 5000).
The old rule-based server is kept as `legacy/app_baseline.py` for comparison.

Chat lives in the **ShopBot widget** (`frontend/shopbot/shopbot.js`), a
self-contained, Shadow-DOM-isolated panel that any page can embed with
`ShopBot.init({ apiBase, host: { navigate, getContext } })`. It opens in three
modes — Large (centred), Docked (right-hand panel) and Compact — is fullscreen on
phones, is EN/AR with RTL, records voice, and navigates the store (search results,
product cards, comparisons, orders, help) instead of listing products in the chat.

> **Dev preview:** append `?mock=1` to the URL to run the widget fully
> client-side — fake health/site/chat replies, **no network and no LLM** — with a
> visible "MOCK DATA" badge.

### ShopBot (LLM agent) configuration

ShopBot is the single LLM agent server (`app.py`, port 5000). It answers with a
bounded tool-calling loop over grounded tools and keeps the full chat contract,
so the evaluation harness reads it unchanged.

Copy `.env.example` to `.env` (untracked, never logged) and set:

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | `gemini` (default) or `groq` |
| `GEMINI_API_KEY` | Gemini provider key |
| `GEMINI_MODEL` | First-choice Gemini model |
| `GEMINI_FALLBACK_MODELS` | Comma-separated Gemini models tried in order |
| `GROQ_API_KEY` | Groq provider key |
| `GROQ_MODEL` | First-choice Groq model (default `llama-3.3-70b-versatile`) |
| `GROQ_FALLBACK_MODELS` | Comma-separated Groq models tried in order |
| `LLM_DAILY_BUDGET` | Max calls per model per day (free-tier guard, default 20) |
| `LLM_CACHE` | `1` caches whole `/api/chat` responses for eval re-runs (off by default) |

Both providers speak the same neutral tool-calling interface, so switching is just
`LLM_PROVIDER=groq` plus a `GROQ_API_KEY` — useful when Gemini is unavailable or
out of quota. Groq uses its OpenAI-compatible endpoint via `httpx`.

Resilience: a retired model (HTTP 404 `NOT_FOUND`) or a per-day quota (429) is
retired for the day and the agent rotates to the next fallback model. When every
model is exhausted, `GET /api/health` reports `"llm": "quota_exhausted"` and a
chat turn returns a bilingual `response_type:"error"` (`error:"quota_exhausted"`);
the widget then shows an amber status dot instead of green. With `LLM_CACHE=1`
identical requests replay from `data/llm_cache.json` (gitignored) instead of
spending quota.

---

## 🎙️ Voice Input Notes
- Whisper model downloads automatically on first transcription (~140MB for `base` model)
- Use `model_size=tiny` for faster (but less accurate) transcription
- Supports both English and Arabic speech with automatic language detection
- Audio recorded in the browser as WebM, converted server-side

---

## 🧪 Testing

Unit tests (no network, no API key, never reads the real `.env`):

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

### Browser / visual checks (dev-only, optional)

An end-to-end visual check drives the storefront with a headless browser and
writes screenshots to `docs/screenshots/`. It runs against **System A only**
(port 5000) and never starts or calls System B, and never reads `.env`. The chat
widget is exercised in mock mode (`?mock=1`), so the run never calls an LLM.

```powershell
pip install -r requirements-dev.txt
playwright install chromium
.venv\Scripts\python.exe tests\e2e\screenshot_journey.py
```

These checks live in `tests/e2e/` and are intentionally **not** part of the
default unit-test run.

---

## 🏫 Academic Notes (College Project)

This project demonstrates the following NLP techniques:
1. **Semantic Search** — Dense retrieval using sentence embeddings + FAISS vector index
2. **Intent Classification** — Zero-shot classification via cosine similarity of sentence embeddings
3. **Price Constraint Parsing** — rule-based extraction of min/max price bounds from English and Arabic queries
4. **Information Extraction** — Rule-based extraction of policy information from text
5. **Multilingual NLP** — Language detection, translation, and Arabic speech recognition
6. **Automatic Speech Recognition** — OpenAI Whisper for multilingual transcription

---

*Built with Flask, sentence-transformers, FAISS, Whisper, and ❤️*
