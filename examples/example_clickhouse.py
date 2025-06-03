import os
import time
import asyncio
from cachelm.adaptors.openai import OpenAIAdaptor
from cachelm.databases.clickhouse import ClickHouse
from cachelm.vectorizers.fastembed import FastEmbedVectorizer
from openai import AsyncOpenAI
from cachelm.middlewares.replacer import Replacer, Replacement
import dotenv

dotenv.load_dotenv()


async def main():
    replacer = Replacer(
        replacements=[
            Replacement(key="{{name}}", value="Anmol"),
            Replacement(key="{{age}}", value="23"),
        ]
    )
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
        middlewares=[replacer],
        distance_threshold=0.1,
    )

    openai_adapted = adaptor.get_adapted()

    system_message = {
        "role": "developer",
        "content": "use my name Anmol and age 23 in your responses. ",
    }
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


asyncio.run(main())
