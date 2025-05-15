from fastembed import TextEmbedding


# Example list of documents
documents: list[str] = [
    "This is built to be faster and lighter than other embedding libraries e.g. Transformers, Sentence-Transformers, etc.",
    "fastembed is supported by and maintained by Qdrant.",
]

# This will trigger the model download and initialization
embedding_model = TextEmbedding()
print("The model BAAI/bge-small-en-v1.5 is ready to use.")

embeddings_generator = embedding_model.embed(documents)  # reminder this is a generator

embeddings_list = list(embedding_model.embed(documents))
# you can also convert the generator to a list, and that to a numpy array
len(embeddings_list[0])  # Vector of 384 dimensions
