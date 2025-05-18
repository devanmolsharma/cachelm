from abc import ABC, abstractmethod


class Vectorizer(ABC):
    """
    Base class for all embedders.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string into a vector.
        Args:
            text (str): The text to embed.
        Returns:
            list[float]: The embedded vector.
        """
        raise NotImplementedError("embed method not implemented")

    @abstractmethod
    def embed_many(self, text: list[str]) -> list[list[float]]:
        """
        Embed multiple text strings into vectors.
        Args:
            text (list[str]): The list of texts to embed.
        Returns:
            list[list[float]]: The list of embedded vectors.
        """
        raise NotImplementedError("embed method not implemented")
