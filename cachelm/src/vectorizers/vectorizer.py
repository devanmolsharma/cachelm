from abc import ABC, abstractmethod


class Vectorizer(ABC):
    """
    Base class for all embedders.
    """

    @abstractmethod
    def embed(self, text: list[str]) -> list[list[float]]:
        """
        Embed the chat history.
        """
        raise NotImplementedError("embed method not implemented")
