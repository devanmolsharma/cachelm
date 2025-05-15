from fastembed import TextEmbedding

from cachelm.src.vectorizers.vectorizer import Vectorizer

embedding_model = TextEmbedding()


class FastEmbedVectorizer(Vectorizer):
    """
    FastEmbed embedding model.
    """

    def embed(self, text: list[str]) -> list[list[float]]:
        """
        Embed the chat history.
        """
        return list(embedding_model.embed(text))
