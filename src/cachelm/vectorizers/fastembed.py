from typing import Any, Sequence, Union
from cachelm.vectorizers.vectorizer import Vectorizer

try:
    from fastembed import TextEmbedding
except ImportError:
    raise ImportError(
        "FastEmbed library is not installed. Run `pip install fastembed` to install it."
    )


class FastEmbedVectorizer(Vectorizer):
    """
    FastEmbed embedding model.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en",
        cache_dir: str | None = None,
        threads: int | None = None,
        providers: Sequence[Union[str, tuple[str, dict[Any, Any]]]] | None = None,
        cuda: bool = False,
        device_ids: list[int] | None = None,
        lazy_load: bool = False,
    ):
        """
        Initialize the FastEmbed embedding model.
        Args:
            model_name (str): The name of the FastEmbed model to use.
            cache_dir (str | None): The directory to cache the model.
            threads (int | None): The number of threads to use for embedding.
            providers (Sequence[Union[str, tuple[str, dict[Any, Any]]]] | None): The providers to use for embedding.
            cuda (bool): Whether to use CUDA for embedding.
            device_ids (list[int] | None): The device IDs to use for embedding.
            lazy_load (bool): Whether to lazy load the model.
        """
        super().__init__()
        self.embedding_model = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            cuda=cuda,
            device_ids=device_ids,
            lazy_load=lazy_load,
        )

    def embed(self, text):
        """
        Embed the chat history.
        """
        out = list(self.embedding_model.embed(text))[0].tolist()
        return out

    def embed_many(self, text: list[str]) -> list[list[float]]:
        """
        Embed the chat history.
        """
        outs = self.embedding_model.embed(text)
        return [o.tolist() for o in outs]
