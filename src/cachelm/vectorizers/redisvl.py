from cachelm.utils.aggregator import AggregateMethod
from cachelm.vectorizers.vectorizer import Vectorizer

try:
    from redisvl.utils.vectorize import BaseVectorizer, HFTextVectorizer
except ImportError:
    raise ImportError(
        "RedisVL library is not installed. Run `pip install redisvl` to install it."
    )


class RedisvlVectorizer(Vectorizer):
    """
    RedisVL embedding model.
    """

    def __init__(
        self,
        vectorizer: BaseVectorizer = HFTextVectorizer(
            model="sentence-transformers/all-mpnet-base-v2",
        ),
        decay: float = 0.4,
        aggregate_method: AggregateMethod = AggregateMethod.CONCATENATE,
        window_size: int = 4,
    ):
        """
        Initialize the RedisVL embedding model.
        Args:
            vectorizer (BaseVectorizer): The RedisVL vectorizer to use.
            decay (float): The decay factor for embedding weights.
            aggregate_method (AggregateMethod): The method to use for aggregating embeddings.
            window_size (int): The size of the window for aggregation.
        """
        super().__init__(
            decay=decay, aggregate_method=aggregate_method, window_size=window_size
        )
        self.vectorizer = vectorizer

    def embed(self, text):
        """
        Embed the chat history.
        """
        out = self.vectorizer.embed(text)
        return out

    def embed_many(self, text: list[str]) -> list[list[float]]:
        """
        Embed the chat history.
        """
        outs = self.vectorizer.embed_many(text)
        return outs
