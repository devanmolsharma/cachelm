from typing import Any
from cachelm.utils.aggregator import AggregateMethod
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
        decay: float = 0.4,
        aggregate_method: AggregateMethod = AggregateMethod.CONCATENATE,
        window_size: int = 4,
    ):
        """
        Initialize the ChromaDB embedding function
        Args:
            vectorizer (embedding_functions.EmbeddingFunction[Documents]): The ChromaDB vectorizer to use.
            decay (float): The decay factor for embedding weights.
            aggregate_method (AggregateMethod): The method to use for aggregating embeddings.
            window_size (int): The size of the window for aggregation.
        """
        super().__init__(
            decay=decay, aggregate_method=aggregate_method, window_size=window_size
        )
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
