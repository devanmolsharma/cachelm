from abc import ABC, abstractmethod

from cachelm.types.chat_history import ChatHistory


class Middleware(ABC):
    """Abstract base class for a middleware."""

    @abstractmethod
    def pre_cache(self, history: ChatHistory) -> ChatHistory | None:
        """Pre-cache hook. Modify the history before caching.
        Args:
            history: The chat history to be modified.
        Returns:
            The modified chat history.
            None if you don't want to cache the history.
        """
        ...

    @abstractmethod
    def post_cache(self, history: ChatHistory) -> ChatHistory:
        """Post-cache hook. Just before returning the response.
        Args:
            history: The chat history to be modified.
        Returns:
            The modified chat history.
        """
        ...
