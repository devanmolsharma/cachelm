import os
from cachelm.src.adaptors.openai import OpenAIAdaptor
from cachelm.src.databases.redis import RedisCache
from cachelm.src.vectorizers.fastembed import FastEmbedVectorizer
import openai
import dotenv

dotenv.load_dotenv()

adaptor = OpenAIAdaptor(
    module=openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
    database=RedisCache(
        vectorizer=FastEmbedVectorizer(),
        host="localhost",
        port=6379,
    ),
    distance_threshold=0.1,
)

openai_adapted = adaptor.get_adapted()

# First attempt
completion1 = openai_adapted.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "developer", "content": "Talk like a pirate."},
        {
            "role": "user",
            "content": "How do I check if a Python object is an instance of a class?",
        },
    ],
)

print("First attempt:")
print(completion1.choices[0].message.content)

# Second attempt to test caching
completion2 = openai_adapted.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "developer", "content": "Your talking style shuld be like a cow."},
        {
            "role": "user",
            "content": "I don't understand how to check if a Python object is an instance of a class.",
        },
    ],
)
print("\nSecond attempt:")
print(completion2.choices[0].message.content)
