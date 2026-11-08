# fluffy-engine

A multimodal Retrieval-Augmented Generation (RAG) assistant over news from [The Batch](https://www.deeplearning.ai/the-batch/).
Ask a question, get an answer grounded in Batch articles, with links to the sources and their images.

## Table of Contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Hardware notes](#hardware-notes)
- [Disclosure](#disclosure)
- [License](#license)

## How it works

1. **ETL** (`src/etl.py`) collects article links from the site's sitemap, scrapes each article
   (metadata from JSON-LD, body from the article content), splits the text into chunks
   and stores everything in [ChromaDB](https://www.trychroma.com/):

   | Collection | Content | Embedding |
   | --- | --- | --- |
   | `texts` | article chunks (≤ 512 tokens) | `BAAI/bge-small-en-v1.5` |
   | `images` | one hero image per article | OpenCLIP `ViT-B-32` (`laion2b_s34b_b79k`) |
   | `articles` | metadata: title, date, tags… | none |

2. **RAG** (`src/ai/rag.py`) retrieves the chunks closest to the question. If none is close enough,
   the question is answered without context and the app says so. Otherwise the chunks go to the LLM
   as context, and the images of the source articles are ranked by text-to-image similarity.
3. **App** (`src/app.py`) is a Streamlit chat. The sidebar selects the LLM provider:
   [Ollama](https://ollama.com), [Docker Model Runner](https://docs.docker.com/ai/model-runner/)
   or OpenAI, all through the OpenAI-compatible Chat Completions API.

## Installation

```bash
git clone https://github.com/rojikaru/fluffy-engine.git
cd fluffy-engine
uv sync
```

`uv sync` installs PyTorch with CUDA 13.4. For a CPU-only install use
`uv sync --no-default-groups --group cpu`.

## Configuration

Copy `.env.example` to `.env` and adjust it. Every variable is optional:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `ollama`, `docker` (Docker Model Runner) or `openai` |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | provider preset | override the preset of `LLM_PROVIDER` |
| `OPENAI_API_KEY` | — | key for the `openai` provider when `LLM_API_KEY` is unset |
| `TORCH_DEVICE` | `cuda` if available | device for the embedding models |
| `CHROMA_SERVER_HOST`, `CHROMA_AUTH_TOKEN_TRANSPORT_HEADER`, `CHROMA_CLIENT_AUTH_PROVIDER`, `CHROMA_CLIENT_AUTH_CREDENTIALS` | — | hosted Chroma; otherwise a local `./chroma` directory |

Provider presets:

| Provider | Base URL | Model |
| --- | --- | --- |
| `ollama` | `http://localhost:11434/v1` | `qwen3.5:9b` |
| `docker` | `http://localhost:12434/engines/llama.cpp/v1` | `ai/qwen3.5:9b` |
| `openai` | OpenAI API | `gpt-5.6-terra` |

The provider, base URL, model and API key can also be changed in the app's sidebar.

## Usage

Run every command from the repository root: modules are imported as `src.*`,
so plain `streamlit run src/app.py` fails with `ModuleNotFoundError: src`.

### 1. Index the articles

```bash
uv run python -m src.etl              # all articles (about 6 s each)
uv run python -m src.etl --limit 200  # a quick sample
```

Articles have deterministic ids, so re-running continues where an interrupted run stopped and skips
articles already in the database. `--refresh` re-scrapes them instead.

### 2. Start a model

```bash
ollama pull qwen3.5:9b                # Ollama
# or
docker model pull ai/qwen3.5:9b       # Docker Model Runner
```

### 3. Run the app

```bash
TORCH_DEVICE=cpu uv run python -m streamlit run src/app.py
```

Open <http://localhost:8501>. Add `LLM_PROVIDER=docker` to start on Docker Model Runner,
or switch providers in the sidebar.

### Docker

The image uses CPU-only PyTorch. Mount the database, and use the host network so the container
reaches Ollama or Docker Model Runner on `localhost`:

```bash
docker build -t fluffy-engine .
docker run --env-file .env --network host -v ./chroma:/app/chroma fluffy-engine
```

## Hardware notes

Tested on an 8 GB laptop GPU (RTX 5060). A 9B model takes about 6.5 GB of VRAM, so:

- run the ETL while no LLM is loaded (`ollama stop qwen3.5:9b` or `docker model unload ai/qwen3.5:9b`);
  it uses the GPU for the embedding models;
- run the app with `TORCH_DEVICE=cpu`, leaving the GPU to the LLM. Embedding one question on the CPU is fast.

## Disclosure

Parts of the code and of this README were written with the help of an AI coding assistant
(Claude Code by Anthropic). All changes were reviewed, tested and committed by the author.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

[Demo.webm](https://github.com/user-attachments/assets/1e0c4c9b-45d9-48d6-a009-63727354ecfc)
