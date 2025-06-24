# cachelm 🌟

**Your Smart Caching Layer for LLM Applications**

Stop wasting money on redundant API calls. **cachelm** intelligently caches LLM responses by understanding the **meaning** of queries, not just the exact words. Slash costs, accelerate response times, and build smarter, faster AI products.

[](https://www.google.com/search?q=https://pypi.org/project/cachelm/)
[](https://opensource.org/licenses/MIT)
[](https://www.google.com/search?q=https://pypi.org/project/cachelm/)
[](https://www.google.com/search?q=https://pypi.org/project/cachelm/)

-----

## The Problem: Repetitive Queries are Expensive

LLMs are powerful but costly. Users often ask the same question in slightly different ways, leading to identical, expensive API calls that traditional key-value caches can't detect.

**"Explain quantum computing"** vs. **"Break down quantum computing basics"**

A traditional cache sees two different requests. **cachelm sees one.**

By understanding semantic intent, `cachelm` serves a cached response for the second query, saving you money and delivering the answer in milliseconds.

## Why Use cachelm?

| Feature | Benefit |
| :--- | :--- |
| **Semantic Caching** | Intelligently handles paraphrased queries to maximize cache hits. |
| **Cost & Latency Reduction** | **Cut LLM API costs by 20-40%** and slash response times from seconds to milliseconds. |
| **Seamless Integration** | Drop it into your existing `openai` client code with just a few lines. No major refactoring needed. |
| **Pluggable Architecture** | Modular design lets you easily swap vector databases (Chroma, Redis, etc.) and models. |
| **Streaming Support** | Full, out-of-the-box compatibility with streaming chat completions. |
| **Production-Ready** | Battle-tested and built for scale with enterprise-grade integrations. |

**Perfect For:**

  - High-traffic LLM applications where API costs are a concern.
  - Real-time chatbots and virtual assistants that require instant responses.
  - Cost-sensitive production deployments and internal tools.

-----

## How It Works

`cachelm` intercepts your LLM API calls and adds a smart caching layer.

1.  **Intercept**: A user sends a prompt through the `cachelm`-enhanced client.
2.  **Vectorize**: The prompt is converted into a numerical representation (an embedding) that captures its semantic meaning.
3.  **Search**: `cachelm` searches your vector database for a similar, previously cached prompt.
4.  **Decision**:
      - **Cache Hit**: If a semantically similar prompt is found within a configurable threshold, the cached response is returned instantly. ⚡
      - **Cache Miss**: If no match is found, the request is sent to the LLM provider (e.g., OpenAI). The new response is then vectorized and stored in the cache for future use.

-----

## 🛠️ Quick Start

### 1\. Installation

Install `cachelm` with the default dependencies (ChromaDB & FastEmbed):

```bash
pip install "cachelm[chroma,fastembed]"
```

### 2\. Basic Usage

Enhance your OpenAI client with caching in just a few lines.

```python
from openai import OpenAI
from cachelm import OpenAIAdaptor, ChromaDatabase, FastEmbedVectorizer

# 1. Initialize the caching components
# By default, ChromaDatabase runs in-memory.
database = ChromaDatabase(
    vectorizer=FastEmbedVectorizer(),
    distance_threshold=0.1  # Lower = stricter matching, Higher = looser matching
)

# 2. Create the adaptor and get your enhanced client
# Replace with your actual OpenAI API key
client = OpenAI(api_key="sk-...")
adaptor = OpenAIAdaptor(module=client, database=database)
smart_client = adaptor.get_adapted()

# 3. Use the client as you normally would!
print("--- First call (will be slow and hit the API) ---")
response = smart_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain the basics of quantum computing in simple terms."}],
)
print(response.choices[0].message.content)


print("\n--- Second call (will be fast and served from cache) ---")
# This query is phrased differently but has the same meaning
cached_response = smart_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Could you break down how quantum computers work?"}],
)
print(cached_response.choices[0].message.content)

# The 'x-cachelm-status' header confirms if the response was a cache HIT or MISS
print(f"\nCache status: {cached_response.headers.get('x-cachelm-status')}")
```

### Customizing Your Cache (e.g., Persistent Storage)

To persist your cache across application restarts, configure your database.

```python
import chromadb

# Configure ChromaDB to save data to disk
persistent_settings = chromadb.config.Settings(
    is_persistent=True,
    persist_directory="./my_llm_cache" # Directory to store the database
)

database = ChromaDatabase(
    vectorizer=FastEmbedVectorizer(),
    chromaSettings=persistent_settings,
    distance_threshold=0.1
)

# The rest of your setup remains the same!
# adaptor = OpenAIAdaptor(...)
```

-----

## Middleware: Customize Caching Behavior

The middleware system lets you hook into the caching process to modify or filter data. This is perfect for handling variable data (like names or IDs) and protecting sensitive information.

Key hooks:

  - `pre_cache_save`: Runs *before* a new response is saved to the cache.
  - `post_cache_retrieval`: Runs *after* a response is retrieved from the cache.

### Example: Normalizing Data with `Replacer`

Imagine your prompts contain usernames that change but the core question is the same. The `Replacer` middleware substitutes placeholders to ensure these queries result in a cache hit.

```python
from cachelm.middlewares.replacer import Replacer, Replacement

# Define replacements: "Anmol" will be treated as {{name}} for caching
replacements = [
    Replacement(key="{{name}}", value="Anmol"),
    Replacement(key="{{user_id}}", value="user_12345"),
]

adaptor = OpenAIAdaptor(
    ...,
    middlewares=[Replacer(replacements)]
)

# Now these two queries will map to the same cache entry:
# 1. "My name is Anmol. What's my order status for ID user_12345?"
# 2. "My name is Bob. What's my order status for ID user_67890?" (if Bob and user_67890 are also in replacements)
```

Before caching, `"Anmol"` becomes `{{name}}`. After retrieval, `{{name}}` is changed back to `"Anmol"`. This dramatically improves cache hits for template-like queries.

-----

## Supported Integrations & Installation

`cachelm` is designed to be modular. Install only what you need.

| Category | Technology | `pip install "cachelm[...]"` |
| :--- | :--- | :--- |
| **Databases** | ChromaDB | `[chroma]` |
| | Redis | `[redis]` |
| | ClickHouse | `[clickhouse]` |
| | Qdrant | `[qdrant]` |
| **Vectorizers** | FastEmbed | `[fastembed]` |
| | RedisVL | `[redis]` |
| | Text2Vec-Chroma | `[chroma]` |
| **LLMs** | OpenAI | (Included by default) |

*More integrations for providers like Anthropic and Cohere are coming soon\!*

-----

## Enterprise & High-Performance Setups

`cachelm` is ready for demanding production environments.

### Redis + RedisVL for High Throughput

```python
from cachelm.databases.redis import RedisDatabase
from cachelm.vectorizers.redisvl import RedisVLVectorizer

# Assumes you have a Redis instance with the RediSearch module
database = RedisDatabase(
    vectorizer=RedisVLVectorizer(model="sentence-transformers/all-MiniLM-L6-v2"),
    redis_url="redis://localhost:6379",
    index_name="llm_cache_prod"
)
```

### ClickHouse for Cloud-Scale Analytics

```python
from cachelm.databases.clickhouse import ClickHouse
from cachelm.vectorizers.fastembed import FastEmbedVectorizer

# Connect to a self-hosted or ClickHouse Cloud instance
database = ClickHouse(
    vectorizer=FastEmbedVectorizer(),
    host="your.clickhouse.cloud.host",
    port=8443,
    username="default",
    password="your-password"
)
```

-----

## Extending cachelm & Contributing

We welcome contributions\! The modular design makes it easy to add new components.

### 1\. Add a New Vectorizer

Implement the `Vectorizer` interface to support a new embedding model.

```python
from cachelm.vectorizers.vectorizer import Vectorizer

class MyVectorizer(Vectorizer):
    def embed(self, text: str) -> list[float]:
        return my_embedding_model.encode(text).tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return my_embedding_model.encode(texts).tolist()
```

### 2\. Add a New Vector Database

Implement the `Database` interface to connect to a different vector store.

```python
from cachelm.databases.database import Database
from cachelm.types.chat_history import Message

class MyDatabase(Database):
    def find(self, history: list[Message], distance_threshold=0.1) -> Message | None:
        # Your logic to search for a similar history vector
        pass
    def write(self, history: list[Message], response: Message):
        # Your logic to store the history vector and response
        pass
    # ... implement connect() and disconnect()
```

See our **[Contribution Guide](https://www.google.com/search?q=CONTRIBUTING.md)** to get started. We're excited to see what you build\!

-----

## License

`cachelm` is licensed under the **[MIT License](https://www.google.com/search?q=LICENSE)**. It is free for both personal and commercial use.

-----

**Ready to Accelerate Your LLM Workloads?** [Report an Issue](https://github.com/devanmolsharma/cachelm/issues) | [View the Source](https://www.google.com/search?q=https://github.com/devanmolsharma/cachelm)