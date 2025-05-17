# cachelm 🌟

**cachelm** is a semantic caching layer designed to supercharge your LLM applications by intelligently caching responses based on **meaning** rather than exact text matches. Reduce API costs, improve response times, and maintain context-aware interactions—even with nuanced queries.

**Problem Solved:** Traditional caching fails with LLMs because users rephrase similar queries. cachelm understands intent through semantic similarity, serving cached responses when equivalent requests occur.

---
![cachelm hero image](media/hero.png)

---

## Why cachelm? 🚀

- **Cut LLM API Costs** by 20-40% through reduced redundant requests
- **Slash Response Times** from seconds to milliseconds for repeated queries
- **Context-Aware Caching** that understands paraphrased requests
- **Future-Proof Architecture** with pluggable components for any LLM/vector DB
- **Seamless Integration** works with your existing OpenAI client code

**Perfect For:**
- High-traffic LLM applications
- Cost-sensitive production deployments
- Real-time chatbots & virtual assistants
- Applications with complex query patterns

---

## Features ✨

| Feature | Benefit |
|---------|---------|
| **Semantic Similarity Matching** | Recognize paraphrased queries as equivalent |
| **Modular Design** | Swap databases/vectorizers without code changes |
| **Streaming Support** | Full compatibility with streaming responses |
| **Production-Ready** | Battle-tested with ChromaDB, Redis, and OpenAI |
| **Extensible Core** | Add new providers in <50 lines of code |

---

## Quick Start 🛠️

### Installation
```bash
pip install git+https://github.com/devanmolsharma/cachelm
```

### Basic Usage (OpenAI + ChromaDB)
```python
from cachelm.adaptors.openai import OpenAIAdaptor
from cachelm.databases.chroma import ChromaDatabase
from cachelm.vectorizers.fastembed import FastEmbedVectorizer
from openai import AsyncOpenAI

# 1. Create components
database = ChromaDatabase(vectorizer=FastEmbedVectorizer())
adaptor = OpenAIAdaptor(
    module=AsyncOpenAI(api_key="sk-your-key"),
    database=database,
    distance_threshold=0.15  # Controls match sensitivity (lower = stricter)
)

# 2. Get enhanced client
smart_client = adaptor.get_adapted()

# 3. Use like regular OpenAI client - now with auto-caching!
response = await smart_client.chat.completions.create(
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    model="gpt-3.5-turbo"
)

# Subsequent similar queries get cached responses!
cached_response = await smart_client.chat.completions.create(
    messages=[{"role": "user", "content": "Break down quantum computing basics"}],  # Different wording
    model="gpt-3.5-turbo" 
)
```

## Middleware System 🧩

cachelm supports a powerful **middleware** system that lets you customize and extend caching behavior at key points in the workflow. Middlewares can inspect, modify, or even block messages before they're cached or after they're retrieved.

### How Middlewares Work

- **pre_cache**: Runs before a response is cached. You can modify the chat history or prevent caching by returning `None`.
- **post_llm_response**: Runs after a cached response is found, just before it's returned. You can modify the cached history or response.

Middlewares are passed as a list to your adaptor:

```python
from cachelm.middlewares.middleware import Middleware

class MyMiddleware(Middleware):
    def pre_cache(self, history):
        # Modify history before caching
        return history

    def post_llm_response(self, history):
        # Modify history after cache retrieval
        return history

adaptor = OpenAIAdaptor(
    ...,
    middlewares=[MyMiddleware()]
)
```

### Example: Replacement Middleware

You can use or build middlewares like the included `Replacer`, which swaps out specific message contents after caching:

```python
from cachelm.middlewares.replacer import Replacer, Replacement

replacements = [
    Replacement(key="{{name}}", value="Anmol"),
    Replacement(key="{{age}}", value="42"),
]

adaptor = OpenAIAdaptor(
    ...,
    middlewares=[Replacer(replacements)]
)
```

This ensures sensitive or repetitive content is normalized before being stored or matched in the cache.

---




---

## Architecture 🧠

![cachelm architecture diagram](media/graph.svg)


**Key Components:**
- **Adaptors**: LLM API wrappers (OpenAI, Anthropic, etc.)
- **Vectorizers**: Text → Embedding converters (FastEmbed, SentenceTransformers)
- **Databases**: Vector stores with similarity search (Chroma, Redis, ClickHouse)


---

## Enterprise-Grade Configurations 🏢

### Redis + RedisVL Performance Setup
```python
from cachelm.databases.redis import RedisDatabase
from cachelm.vectorizers.redisvl import RedisVLVectorizer

database = RedisDatabase(
    vectorizer=RedisVLVectorizer("your-model-name"),
    redis_url="redis://localhost:6379",
    index_name="llm_cache"
)
```

### ClickHouse Cloud Scale-Out
```python
from cachelm.databases.clickhouse import ClickHouse
from cachelm.vectorizers.fastembed import FastEmbedVectorizer

database = ClickHouse(
    vectorizer=FastEmbedVectorizer(),
    host="your.clickhouse.cloud",
    port=8443,
    username="admin",
    password="your-password"
)
```


## Supported Integrations 🔌

| Category       | Technologies |
|----------------|--------------|
| **Databases**  | ChromaDB, Redis, ClickHouse |
| **Vectorizers**| FastEmbed, RedisVL, Chroma |
| **LLMs**       | OpenAI (More coming!) |

---

## Extending cachelm 🔧

### Add New Vectorizer

```python
from cachelm.vectorizers.vectorizer import Vectorizer

class MyVectorizer(Vectorizer):
    def embed(self, text: str) -> list[float]:
        return my_embedding_model(text)

    def embedMany(self, texts: list[str]) -> list[list[float]]:
        return [my_embedding_model(t) for t in texts]
```

### Add New Database

```python
from cachelm.databases.database import Database
from cachelm.types.chat_history import Message

class MyDatabase(Database):
    def connect(self) -> bool:
        # Connect to your vector DB
        return True

    def disconnect(self):
        # Disconnect logic
        pass

    def write(self, history: list[Message], response: Message):
        # Store (history, response) in your DB
        pass

    def find(self, history: list[Message], distance_threshold=0.1) -> Message | None:
        # Search for similar history in your DB
        return None
```

### Add New Adaptor

```python
from cachelm.adaptors.adaptor import Adaptor
from cachelm.types.chat_history import Message

class MyAdaptor(Adaptor):
    def get_adapted(self):
        # Return your adapted module/client
        return self.module

    # Optionally override methods like add_user_message, add_assistant_message, etc.
```

---

## How It Works

- **ChatHistory**: Manages message history, supports padding and slicing for context windows.
- **Adaptor**: Wraps your LLM client, intercepts calls, manages caching logic, and handles chat history.
- **Database**: Abstract interface for vector stores, handles connect/disconnect, write, and semantic search.

---

## Contributing 🤝

We welcome extensions for:
- New LLM providers (Anthropic, Cohere, etc.)
- Additional vector databases
- Novel caching strategies

See our [Contribution Guide](CONTRIBUTING.md) to get started!

---

## License 📄

MIT - Free for commercial and personal use

---

**Ready to Accelerate Your LLM Workloads?**  
[Get Started Now](#quick-start) | [Report Issue](https://github.com/devanmolsharma/cachelm/issues)