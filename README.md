# fluffy-engine

A Retrieval-Augmented Generation (RAG) system enriched with news from [The Batch](https://www.deeplearning.ai/thebatch/).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Overview

**fluffy-engine** is an open-source RAG (Retrieval-Augmented Generation) system designed to integrate the latest news and insights from The Batch, a leading AI newsletter by DeepLearning.AI. The system fetches, indexes, and uses news content to enrich AI-driven question answering and summarization tasks.

## Features

- **Retrieval-Augmented Generation:** Combines external knowledge sources with generative models for improved responses.
- **News Integration:** Regularly pulls and indexes articles from The Batch.
- **Customizable Pipelines:** Easily adapt retrieval and generation components for specific use cases.
- **Fully in Python:** Simple to install, extend, and integrate into other Python projects.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/rojikaru/fluffy-engine.git
   cd fluffy-engine
    ```

2. **Install dependencies:**

    ```bash
    uv sync
    ```

## Configuration  

Before running the application, you need to configure your environment variables.
Create a `.env` file in the root directory and add your configuration settings, such as API keys and database URLs.  
Required environment variables:

  ```plaintext
  OPENAI_API_KEY=your_database_url_here

  # Add huggingface/google/anthropic API keys if needed
  ```

## Usage

1. **Extract - transform - load (ETL):**  
   Run the ETL pipeline to fetch and index news articles.

   ```bash
   uv run python -m src.etl_pipeline
   ```

   Also, there is langchain implementation of the ETL pipeline:

   ```bash
   uv run python -m src.etl_langchain
   ```

2. **Client application:**
    Start the Streamlit app to handle queries.

    ```bash
    uv run python -m streamlit run src/app.py
    ```

   > Run all commands from the repository root: modules are imported as `src.*`,
   > so plain `streamlit run src/app.py` fails with `ModuleNotFoundError: src`.

## Contributing  

I welcome contributions to **fluffy-engine**! If you have ideas for improvements or new features, please open an issue or submit a pull request.  

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.  

[Demo.webm](https://github.com/user-attachments/assets/1e0c4c9b-45d9-48d6-a009-63727354ecfc)
