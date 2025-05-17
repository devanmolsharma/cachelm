from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from cachelm.databases.database import Database
from loguru import logger
import signal

from cachelm.middlewares.middleware import Middleware
from cachelm.types.chat_history import ChatHistory, Message

T = TypeVar("T")


class Adaptor(ABC, Generic[T]):
    """
    Base class for all adaptors.
    """

    def __init__(
        self,
        module: T,
        database: Database,
        window_size: int = 4,
        distance_threshold: float = 0.2,
        dispose_on_sigint: bool = False,
        middlewares: list[Middleware] = [],
    ):
        """
        Initialize the adaptor with a module, database, and configuration options.

        Args:
            module: The module to be adapted.
            database: The database instance used for caching.
            window_size: Number of recent messages to consider for caching (default: 4).
            distance_threshold: Similarity threshold for cache retrieval (default: 0.2).
            dispose_on_sigint: If True, dispose adaptor on SIGINT signal (default: False).
        """
        self._validate_inputs(database, window_size, distance_threshold)
        self._initialize_attributes(
            module, database, window_size, distance_threshold, middlewares
        )
        if dispose_on_sigint:
            signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum, frame):
        """
        Handle SIGINT signal.
        """
        logger.info("SIGINT received, disposing of the adaptor")
        self.dispose()
        exit(0)

    def _validate_inputs(
        self, database: Database, window_size: int, distance_threshold: float
    ):
        """
        Validate the inputs for the adaptor.
        """
        if not isinstance(database, Database):
            raise TypeError("Database must be an instance of Database")
        if distance_threshold < 0 or distance_threshold > 1:
            raise ValueError("Distance threshold must be between 0 and 1")
        if window_size < 0:
            raise ValueError("Window size must be greater than or equal to 0")

    def _initialize_attributes(
        self,
        module: T,
        database: Database,
        window_size: int,
        distance_threshold: float,
        middlewares: list[Middleware],
    ):
        """
        Initialize the attributes for the adaptor.
        """
        success = database.connect()
        if not success:
            raise Exception("Failed to connect to the database")
        logger.info("Connected to the database")
        self.database = database
        self.module = module
        self.history = ChatHistory()
        self.window_size = window_size
        self.distance_threshold = distance_threshold
        self.middlewares = middlewares

    @abstractmethod
    def get_adapted(self) -> T:
        """
        Get the adapted module.
        """
        raise NotImplementedError("getAdapted method not implemented")

    def set_history(self, messages: list[Message]):
        """
        Set the chat history.
        """
        self.history.setMessages(messages)

    def add_user_message(self, message: Message):
        """
        Add a user message to the chat history.
        """
        self.history.add_user_message(message)

    def add_assistant_message(self, message: Message):
        """
        Add an assistant message to the chat history.
        Applies all middlewares to the message (pre-cache)
        """
        lastMessagesWindow = self.history.getMessages(self.window_size)
        self.history.add_assistant_message(message)
        for middleware in self.middlewares:
            message = middleware.pre_cache(message)
        self.database.write(lastMessagesWindow, message)

    def get_cache(self):
        """
        Get the cache from the database.
        Applies all middlewares to the cache (post-cache).
        """
        cache = self.database.find(
            self.history.getMessages(self.window_size),
            self.distance_threshold,
        )
        if not cache:
            return None

        for middleware in self.middlewares:
            cache = middleware.post_llm_response(cache)
        return

    def dispose(self):
        """
        Dispose of the adaptor.
        """
        self.database.disconnect()
        logger.info("Disconnected from the database")
