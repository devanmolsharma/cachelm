from fastembed import TextEmbedding

from cachelm.src.embedders.embedder import Embedder

embedding_model = TextEmbedding()


class FastEmbedEmbedder(Embedder):
    """
    FastEmbed embedding model.
    """

    def get_embedding(self, text: list[str]) -> list[list[float]]:
        """
        Embed the chat history.
        """
        return list(embedding_model.embed(text))
