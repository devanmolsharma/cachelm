from fastembed import TextEmbedding

from cachelm.src.vectorizers.vectorizer import Vectorizer

embedding_model = TextEmbedding()


class FastEmbedVectorizer(Vectorizer):
    """
    FastEmbed embedding model.
    """

    def embed(self, text):
        """
        Embed the chat history.
        """
        out = list(embedding_model.embed(text))[0].tolist()
        return out

    def embedMany(self, text: list[str]) -> list[list[float]]:
        """
        Embed the chat history.
        """
        outs = embedding_model.embed(text)
        return [o.tolist() for o in outs]
