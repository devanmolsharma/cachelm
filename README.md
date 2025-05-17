# cachelm

**cachelm** is a highly modular semantic caching framework for LLM (Large Language Model) applications. It enables fast, efficient, and intelligent caching of LLM responses by leveraging vector similarity search and pluggable components for databases, vectorizers, and model adaptors.

## Features

- **Semantic Caching:** Caches LLM responses based on semantic similarity, not just exact text match.
- **Highly Modular:** Easily swap out databases, vectorizers, and LLM adaptors.
- **Streaming Support:** Works with streaming LLM APIs.
- **Plug-and-Play:** Minimal setup to get started with your favorite LLM and vector database.

---

## Installation

You can install cachelm directly from GitHub:

```bash
pip install git+https://github.com/devanmolsharma/cachelm
```

Or clone the repository and install manually:

```bash
git clone https://github.com/devanmolsharma/cachelm
cd cachelm
pip install .
```

---

## Architecture Overview

- **Adaptors:** Interface between cachelm and LLM APIs (e.g., OpenAI).
- **Databases:** Store and retrieve cached responses using vector similarity (e.g., ChromaDB, Redis).
- **Vectorizers:** Convert text to embeddings for semantic search (e.g., FastEmbed, RedisVL).

---

## Supported Components

### Databases

| Database   | Integration Status |
|------------|-------------------|
| ChromaDB   | ✅ Supported      |
| Redis      | ✅ Supported      |
| ClickHouse | ✅ Supported      |

### Adaptors

| Adaptor    | Integration Status |
|------------|-------------------|
| OpenAI     | ✅ Supported      |

### Vectorizers

| Vectorizer   | Integration Status |
|--------------|-------------------|
| FastEmbed    | ✅ Supported      |
| RedisVL      | ✅ Supported      |
| Chroma       | ✅ Supported      |

---

## Example Usage

```python
from cachelm.adaptors.openai import OpenAIAdaptor
from cachelm.databases.chroma import ChromaDatabase
from cachelm.vectorizers.fastembed import FastEmbedVectorizer
from openai import AsyncOpenAI 
# or from openai import OpenAI

adaptor = OpenAIAdaptor(
    module=AsyncOpenAI(api_key="YOUR_OPENAI_API_KEY"), # or OpenAI
    database=ChromaDatabase(vectorizer=FastEmbedVectorizer()),
    distance_threshold=0.1,
)

openai_adapted = adaptor.get_adapted()
# Use openai_adapted as you would use the OpenAI client, with caching enabled!
```


## Extending cachelm

To add support for a new database, vectorizer, or LLM API:
- Implement the corresponding abstract base class.
- Register your component in your application code.

---

## License

MIT

