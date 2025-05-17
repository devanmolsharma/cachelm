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

    # First attempt
    start_time = time.time()
    res = await openai_adapted.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "developer",
                "content": "use {{name}} and {{age}} in your response.",
            },
            {
                "role": "user",
                "content": "Hi, how are you?",
            },
        ],
    )
    end_time = time.time()
    print(f"First attempt time: {end_time - start_time:.2f} seconds")

    print("First attempt response:", res.choices[0].message.content)


asyncio.run(main())
