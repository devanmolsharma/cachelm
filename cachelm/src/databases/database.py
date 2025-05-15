from abc import ABC, abstractmethod
from cachelm.src.adaptors.adaptor import ChatHistory


class Database(ABC):
    """Abstract base class for a database."""

    @abstractmethod
    def connect(self):
        """Connect to the database."""
        ...

    @abstractmethod
    def disconnect(self):
        """Disconnect from the database."""
        ...

    @abstractmethod
    def write(self, embeddings: list[list[float]], chat: ChatHistory):
        """Write data to the database."""
        ...

    @abstractmethod
    def find(
        self, embeddings: list[list[float]], min_similarity=0.9
    ) -> ChatHistory | None:
        """Find data in the database."""
        ...
