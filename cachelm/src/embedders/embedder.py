from abc import ABC, abstractmethod


class Embedder(ABC):
    """
    Base class for all embedders.
    """

    @abstractmethod
    def get_embedding(self, text: list[str]) -> list[list[float]]:
        """
        Embed the chat history.
        """
        ...
