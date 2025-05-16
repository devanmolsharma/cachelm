from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from cachelm.src.databases.database import Database


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
    ):
        """
        Initialize the adaptor with a module, database, and embedder.
        """
        self.database = database
        success = database.connect()
        if not success:
            raise Exception("Failed to connect to the database")
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
