from abc import ABC, abstractmethod
from cachelm.utils import async_wrap
from cachelm.utils.chat_history import Message
from cachelm.vectorizers.vectorizer import Vectorizer


class Database(ABC):
    """Abstract base class for a database."""

    def __init__(
        self,
        vectorizer: Vectorizer,
        unique_id: str = "cachelm",
        distance_threshold: float = 0.1,
        max_size: int = 100,
    ):
        """
        Initialize the database.
        Args:
            vectorizer (Vectorizer): The vectorizer to use for embedding messages.
            unique_id (str): Unique identifier for the database instance.
            distance_threshold (float): Similarity threshold for cache retrieval.
            max_size (int): Maximum number of rows in the database.
        """
        self.vectorizer = vectorizer
        self.unique_id = unique_id
        self.distance_threshold = distance_threshold
        self.max_size = max_size

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the database."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def reset(self):
        """Reset the database."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def disconnect(self):
        """Disconnect from the database."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def write(self, history: list[Message], response: Message):
        """Write data to the database."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def find(self, history: list[Message]) -> Message | None:
        """Find data in the database."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def size(self) -> int:
        """Get the size of the database."""
        raise NotImplementedError("Subclasses must implement this method.")
