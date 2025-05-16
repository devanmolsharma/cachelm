from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from cachelm.src.databases.database import Database
from loguru import logger
import signal

T = TypeVar("T")


class ChatHistory:
    """
    Class to represent the chat history.
    """

    def __init__(self):
        self.messages: list[dict] = []

    def add_user_message(self, message: str):
        """
        Add a user message to the chat history.
        """
        self.messages.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        """
        Add an assistant message to the chat history.
        """
        self.messages.append({"role": "assistant", "content": message})

    def setMessages(self, messages: list[dict]):
        """
        Set the messages in the chat history.
        """
        self.messages = messages

    def getMessageTexts(self, length: int = 0) -> list[str]:
        """
        Get the message texts from the chat history.
        If length is 0, return all messages.
        If length is greater than the number of messages, return the last 'length' messages.
        If length is less than the number of messages, return the first 'length' messages. Prepend with empty strings if necessary.
        Example:
        >>> chat_history = ChatHistory()
        >>> chat_history.add_user_message("Hello")
        >>> chat_history.add_assistant_message("Hi there!")
        >>> chat_history.getMessageTexts()
        ['Hi there!', 'Hello']
        >>> chat_history.getMessageTexts(1)
        ['Hi there!']
        >>> chat_history.getMessageTexts(3)
        ['', 'Hi there!', 'Hello']
        >>> chat_history.getMessageTexts(5)
        ['', '', 'Hi there!', 'Hello']
        """
        if length == 0:
            length = len(self.messages)
        if length > len(self.messages):
            return [msg["content"] for msg in self.messages[-length:]]
        else:
            texts = [msg["content"] for msg in self.messages]
            texts.reverse()
            texts.extend(["" for _ in range(length - len(texts))])
            texts.reverse()
            return texts

    def __len__(self):
        """
        Get the length of the chat history.
        """
        return len(self.messages)

    def __getitem__(self, index: int):
        """
        Get an item from the chat history.
        """
        return self.messages[index]


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
    ):
        """
        Initialize the adaptor with a module, database, and embedder.
        """
        self._validate_inputs(database, window_size, distance_threshold)
        self._initialize_attributes(module, database, window_size, distance_threshold)
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
        self, module: T, database: Database, window_size: int, distance_threshold: float
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

    @abstractmethod
    def get_adapted(self) -> T:
        """
        Get the adapted module.
        """
        raise NotImplementedError("getAdapted method not implemented")

    def setHistory(self, messages: list[dict]):
        """
        Set the chat history.
        """
        self.history.setMessages(messages)

    def add_user_message(self, message: str):
        """
        Add a user message to the chat history.
        """
        self.history.add_user_message(message)

    def add_assistant_message(self, message: str):
        """
        Add an assistant message to the chat history.
        """
        lastMessagesWindow = self.history.getMessageTexts(self.window_size)
        self.history.add_assistant_message(message)
        self.database.write(lastMessagesWindow, message)

    def get_cache(self):
        """
        Get the cache from the database.
        """
        return self.database.find(
            self.history.getMessageTexts(self.window_size),
            self.distance_threshold,
        )

    def dispose(self):
        """
        Dispose of the adaptor.
        """
        self.database.disconnect()
        logger.info("Disconnected from the database")
