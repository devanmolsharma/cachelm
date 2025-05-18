from abc import ABC, abstractmethod
from cachelm.types.chat_history import ChatHistory, Message


class Middleware(ABC):
    """Abstract base class for a middleware."""

    @abstractmethod
    def pre_cache_save(self, message: Message, history: ChatHistory) -> Message | None:
        """Pre-cache hook. Modify the history before caching.
        Args:
            message: The message to be cached.
            history: The chat history to be cached.

        Returns:
            The modified message.
            None if you don't want to cache the message.
        """
        ...

    @abstractmethod
    def post_cache_retrieval(self, message: Message, history: ChatHistory) -> Message:
        """Post-cache hook. Just before returning the response.
        Args:
            message: The message to be cached.
            history: The chat history to be cached.

        Returns:
            The modified message.
        """
        ...
