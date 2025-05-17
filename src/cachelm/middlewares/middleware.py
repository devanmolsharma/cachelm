from abc import ABC, abstractmethod
from cachelm.types.chat_history import ChatHistory, Message


class Middleware(ABC):
    """Abstract base class for a middleware."""

    @abstractmethod
    def pre_cache(self, message: Message, history: ChatHistory) -> Message | None:
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
    def post_llm_response(self, message: Message, history: ChatHistory) -> Message:
        """Post-cache hook. Just before returning the response.
        Args:
            message: The message to be cached.
            history: The chat history to be cached.

        Returns:
            The modified message.
        """
        ...
