from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from cachelm.databases.database import Database
from loguru import logger
import signal

from cachelm.middlewares.deduper import Deduper
from cachelm.middlewares.middleware import Middleware
from cachelm.utils.async_wrap import async_wrap
from cachelm.utils.chat_history import ChatHistory, Message
from threading import Thread

T = TypeVar("T")


class Adaptor(ABC, Generic[T]):
    """
    Base class for all adaptors.
    """

    def __init__(
        self,
        module: T,
        database: Database,
        dispose_on_sigint: bool = False,
        middlewares: list[Middleware] = [],
        dedupe: bool = True,
        ignore_system_messages: bool = True,
    ):
        """
        Initialize the adaptor with a module, database, and configuration options.

        Args:
            module: The module to be adapted.
            database: The database instance used for caching.
            distance_threshold: Similarity threshold for cache retrieval (default: 0.4).
            dispose_on_sigint: If True, dispose adaptor on SIGINT signal (default: False).
            middlewares: List of middlewares to apply to the messages (default: empty list).
            dedupe: If True, apply deduplication middleware (default: True).
            max_db_rows: Maximum number of rows in the database (default: 0, meaning no limit).
            ignore_system_messages: If True, ignore system messages in the chat history when saving and retrieving messages (default: True).
        """
        self._validate_inputs(
            database,
            middlewares,
            dedupe,
            ignore_system_messages,
        )
        self._initialize_attributes(
            module,
            database,
            middlewares,
            dedupe,
            ignore_system_messages,
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
        self,
        database: Database,
        middlewares: list[Middleware],
        dedupe: bool,
        ignore_system_messages: bool = True,
    ):
        """
        Validate the inputs for the adaptor.
        """
        if not isinstance(database, Database):
            raise TypeError("Database must be an instance of Database")

        if not isinstance(middlewares, list):
            raise TypeError("Middlewares must be a list of Middleware instances")
        if not all(isinstance(m, Middleware) for m in middlewares):
            raise TypeError("All middlewares must be instances of Middleware")
        if not isinstance(dedupe, bool):
            raise TypeError("Dedupe must be a boolean value")
        if not isinstance(ignore_system_messages, bool):
            raise TypeError("ignore_system_messages must be a boolean value")

    def _initialize_attributes(
        self,
        module: T,
        database: Database,
        middlewares: list[Middleware],
        dedupe: bool,
        ignore_system_messages: bool = True,
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
        self.window_size = database.vectorizer.window_size
        self.middlewares = middlewares
        self.max_db_rows = database.max_size
        self.ignore_system_messages = ignore_system_messages
        if dedupe:
            self.middlewares.append(Deduper())

    @abstractmethod
    def get_adapted(self) -> T:
        """
        Get the adapted module.
        """
        raise NotImplementedError("getAdapted method not implemented")

    def _filter_out_system_messages(self, messages: list[Message]) -> list[Message]:
        """
        Filter out system messages from the chat history.
        If ignore_system_messages is True, it will remove messages with role 'system'.
        """
        if not self.ignore_system_messages:
            return messages
        return [msg for msg in messages if msg.role != "system"]

    def set_history(self, messages: list[Message]):
        """
        Set the chat history.
        """
        self.history.set_messages(self._filter_out_system_messages(messages))

    def add_user_message(self, message: Message):
        """
        Add a user message to the chat history.
        """
        self.history.add_user_message(message)

    def add_assistant_message(self, message: Message, save_to_db: bool = True):
        """
        Handles adding an assistant message to the chat history and optionally saving it to the database.
        Runs the saving process in a separate thread to avoid blocking the main thread.
        This method applies all middlewares to the message before saving it to the database.
        If the database size exceeds the maximum limit, it skips saving the message to the database.
        """
        self._process_add_assistant_message_async(message)

    async def add_assistant_message_async(self, message: Message):
        """
        Asynchronously add an assistant message to the chat history.
        Runs the saving process in a separate thread to avoid blocking the main thread.
        This method applies all middlewares to the message before saving it to the database.
        If the database size exceeds the maximum limit, it skips saving the message to the database.
        """
        return await async_wrap(self._process_add_assistant_message_async)(message)

    def _process_add_assistant_message_async(self, message: Message):
        """
        Asynchronously add an assistant message to the chat history.
        Applies all middlewares to the message (pre-cache).
        """
        try:
            db_size = self.database.size() if self.max_db_rows > 0 else 0
            if self.max_db_rows > 0 and db_size >= self.max_db_rows:
                logger.warning(
                    f"Database size {db_size} has reached the maximum limit of {self.max_db_rows}. "
                    "Skipping saving the message to the database."
                )
                return
            self._apply_pre_cache_to_history()
            lastMessagesWindow = self.history.get_messages(self.window_size)
            for middleware in self.middlewares:
                message = middleware.pre_cache_save(message, self.history)
                if message is None:
                    return
            self.database.write(lastMessagesWindow, message)
        except Exception as e:
            logger.error(f"Error while adding assistant message: {e}")
            return

    def _apply_pre_cache_to_history(self):
        """
        Apply pre-cache middlewares to the history.
        This is used before saving the history to the database.
        """
        messages = self.history.messages
        for i, message in enumerate(messages):
            for middleware in self.middlewares:
                newMessage = middleware.pre_cache_save(message, self.history)
                if newMessage is not None:
                    messages[i] = newMessage
                else:
                    break
        # Set the modified messages back to the history
        # This ensures that the history is updated with the pre-cache modifications
        self.history.set_messages(messages)

    def _apply_post_cache_middlewares(self, message: Message):
        """
        Apply post-cache middlewares to the message.
        """
        for middleware in self.middlewares:
            message = middleware.post_cache_retrieval(message, self.history)
            if message is None:
                return None
        return message

    def _apply_pre_cache_middlewares(self, message: Message):
        """
        Apply pre-cache middlewares to the message.
        """
        for middleware in self.middlewares:
            message = middleware.pre_cache_save(message, self.history)
            if message is None:
                return None
        return message

    def get_cache(self):
        """
        Get the cache from the database.
        Applies all middlewares to the cache (post-cache).

        If the cache is empty, return None.
        If the cache is not empty, add it to the history.

        """
        self._apply_pre_cache_to_history()
        cache = self.database.find(self.history.get_messages(self.window_size))
        if not cache:
            return None

        # Apply post-cache middlewares to the cache
        cache = self._apply_post_cache_middlewares(cache)
        if cache is None:
            return None
        # Add the cache to the history
        self.history.add_assistant_message(cache)
        return cache

    async def get_cache_async(self):
        """
        Asynchronously get the cache from the database.
        Applies all middlewares to the cache (post-cache).

        If the cache is empty, return None.
        If the cache is not empty, add it to the history.
        """
        return await async_wrap(self.get_cache)()

    def dispose(self):
        """
        Dispose of the adaptor.
        """
        self.database.disconnect()
        logger.info("Disconnected from the database")
