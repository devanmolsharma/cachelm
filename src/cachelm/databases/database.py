from abc import ABC, abstractmethod
from cachelm.vectorizers.vectorizer import Vectorizer


class Database(ABC):
    """Abstract base class for a database."""

    def __init__(self, vectorizer: Vectorizer, unique_id: str = "cachelm"):
        self.vectorizer = vectorizer
        self.unique_id = unique_id

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the database."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def disconnect(self):
        """Disconnect from the database."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def write(self, history: list[str], response: str):
        """Write data to the database."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def find(self, history: list[str], distance_threshold=0.9) -> str | None:
        """Find data in the database."""
        raise NotImplementedError("Subclasses must implement this method.")
