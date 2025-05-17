import os
import time
import asyncio
from cachelm.adaptors.openai import OpenAIAdaptor
from cachelm.databases.clickhouse import ClickHouse
from cachelm.vectorizers.fastembed import FastEmbedVectorizer
from openai import AsyncOpenAI
import dotenv

dotenv.load_dotenv()


async def main():
    adaptor = OpenAIAdaptor(
        module=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        database=ClickHouse(
            host="localhost",
            port=18123,
            user="default",
            password="pass",
            database="cachelm",
            vectorizer=FastEmbedVectorizer(),
        ),
        distance_threshold=0.1,
    )

    openai_adapted = adaptor.get_adapted()

    # First attempt
    start_time = time.time()
    await openai_adapted.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "Talk like a pirate."},
            {
                "role": "user",
                "content": "How do I check if a Python object is an instance of a class?",
            },
        ],
    )
    end_time = time.time()
    print(f"First attempt time: {end_time - start_time:.2f} seconds")

    # Second attempt to test caching
    start_time = time.time()
    await openai_adapted.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "developer",
                "content": "Your talking style shuld be like a pirate.",
            },
            {
                "role": "user",
                "content": "I don't understand how to check if a Python object is an instance of a class.",
            },
        ],
    )
    end_time = time.time()
    print(f"Second attempt (cache) time: {end_time - start_time:.2f} seconds")

    # Streaming attempt
    print("Streaming attempt running...")
    start_time = time.time()
    streaming_response = await openai_adapted.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "Talk like a pirate."},
            {
                "role": "user",
                "content": "How do I check if a cat is hungry?",
            },
        ],
        stream=True,
    )

    async for _ in streaming_response:
        pass  # Just consume the stream, don't print

    end_time = time.time()
    print(f"Streaming attempt time: {end_time - start_time:.2f} seconds")


asyncio.run(main())
