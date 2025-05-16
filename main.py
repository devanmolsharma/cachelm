import os
import time
import asyncio
from cachelm.src.adaptors.openai import OpenAIAdaptor
from cachelm.src.databases.chroma import ChromaDatabase
from cachelm.src.vectorizers.fastembed import FastEmbedVectorizer
from openai import AsyncOpenAI
import dotenv

dotenv.load_dotenv()


async def main():
    adaptor = OpenAIAdaptor(
        module=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        database=ChromaDatabase(
            vectorizer=FastEmbedVectorizer(),
        ),
        distance_threshold=0.1,
    )

    openai_adapted = adaptor.get_adapted()

    # First attempt
    start_time = time.time()
    completion1 = await openai_adapted.chat.completions.create(
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

    print("First attempt:")
    print(completion1.choices[0].message.content)
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    # Second attempt to test caching
    start_time = time.time()
    completion2 = await openai_adapted.chat.completions.create(
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

    print("\nSecond attempt:")
    print(completion2.choices[0].message.content)
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    # Streaming attempt
    print("\nStreaming attempt:")
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
        stream=True,  # Enable streaming
    )

    response_content = ""
    async for chunk in streaming_response:
        chunk_content = chunk.choices[0].delta.content
        if chunk_content:
            # Only print non-empty chunks
            response_content += chunk_content
            print(chunk_content, end="", flush=True)  # Print each chunk as it arrives

    end_time = time.time()
    print("\n\nTime taken for streaming: {:.2f} seconds".format(end_time - start_time))


# Run the async main function
asyncio.run(main())
