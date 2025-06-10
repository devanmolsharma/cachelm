import os
import time
import asyncio
from cachelm.adaptors.openai import OpenAIAdaptor
from cachelm.databases.qdrant import QdrantDatabase
from cachelm.vectorizers.fastembed import FastEmbedVectorizer
from openai import AsyncOpenAI
import dotenv

# filepath: /home/anmolsharma/Documents/FunProjects/cachelm/examples/example_qdrant.py

dotenv.load_dotenv()


async def main():
    adaptor = OpenAIAdaptor(
        module=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        database=QdrantDatabase(
            vectorizer=FastEmbedVectorizer(decay=0.4),
            location="http://localhost:6333",  # Or :memory: for in-memory storage
        ),
        distance_threshold=0.1,
    )

    openai_adapted = adaptor.get_adapted()

    system_message = {"role": "developer", "content": "Talk like a pirate."}
    messages = [system_message]

    while True:
        user_input = input("\nAsk your question (or type 'exit' to quit): ")
        if user_input.strip().lower() == "exit":
            break

        messages.append({"role": "user", "content": user_input})

        start_time = time.time()
        completion = await openai_adapted.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        end_time = time.time()

        assistant_message = completion.choices[0].message.content
        print("\nResponse:")
        print(assistant_message)
        print(f"Time taken: {end_time - start_time:.2f} seconds")

        messages.append({"role": "assistant", "content": assistant_message})


# Run the async main function
asyncio.run(main())
