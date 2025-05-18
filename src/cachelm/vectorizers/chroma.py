from cachelm.vectorizers.vectorizer import Vectorizer

try:
    from chromadb.utils import embedding_functions
except ImportError:
    raise ImportError(
        "ChromaDB library is not installed. Run `pip install chromadb` to install it."
    )


class ChromaVectorizer(Vectorizer):
    """
    ChromaDB embedding function.
    """

    def __init__(
        self,
        vectorizer: embedding_functions.EmbeddingFunction[
            embedding_functions.Documents
        ] = embedding_functions.Text2VecEmbeddingFunction(),
    ):
        """
        Initialize the ChromaDB embedding function
        Args:
            vectorizer (embedding_functions.EmbeddingFunction[Documents]): The ChromaDB vectorizer to use.
        """
        super().__init__()
        if not isinstance(
            vectorizer,
            embedding_functions.EmbeddingFunction,
        ):
            raise TypeError(
                "vectorizer must be an instance of chromadb.EmbeddingFunction[Documents] "
            )
        self.vectorizer = vectorizer

    def embed(self, text):
        """
        Embed the chat history.
        """
        out = self.vectorizer([text])[0]
        return out.tolist()

    def embed_many(self, text: list[str]) -> list[list[float]]:
        """
        Embed the chat history.
        """
        outs = self.vectorizer(text)
        return [o.tolist() for o in outs]
