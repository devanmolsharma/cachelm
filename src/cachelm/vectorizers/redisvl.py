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
    ):
        """
        Initialize the RedisVL embedding model.
        Args:
            vectorizer (BaseVectorizer): The RedisVL vectorizer to use.
            decay (float): The decay factor for embedding weights.
        """
        super().__init__(
            decay=decay,
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
