from cachelm.src.vectorizers.vectorizer import Vectorizer

try:
    from redisvl.utils.vectorize import BaseVectorizer
except ImportError:
    raise ImportError(
        "RedisVL library is not installed. Run `pip install redisvl` to install it."
    )


class RedisvlVectorizer(Vectorizer):
    """
    RedisVL embedding model.
    """

    def __init__(self, vectorizer: BaseVectorizer):
        """
        Initialize the RedisVL embedding model.
        """
        super().__init__()
        self.vectorizer = vectorizer

    def embed(self, text):
        """
        Embed the chat history.
        """
        out = self.vectorizer.embed(text)
        return out

    def embedMany(self, text: list[str]) -> list[list[float]]:
        """
        Embed the chat history.
        """
        outs = self.vectorizer.embed_many(text)
        return outs
