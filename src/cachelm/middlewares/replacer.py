from typing import Literal
from cachelm.middlewares.middleware import Middleware


class Replacement:
    """
    A class representing a replacement operation.
    """

    def __init__(self, key: str, value: str):
        """
        Initialize the Replacement object.

        Args:
            key (str): The inner representation of the string to be replaced.
            value (str): The string to replace with.
        """
        self.key = key
        self.value = value


class Replacer(Middleware):
    """
    Middleware for replacing specific strings in messages before saving to cache
    and after retrieving from cache.
    This is useful for handling special tokens or placeholders in the message content and improving cache efficiency.

    It replaces the `key` with `value` before saving to cache and vice versa after retrieval.
    """

    def __init__(self, replacements: list[Replacement]):
        """
        Initialize the Replacer middleware.

        Args:
            replacements: list[Replacement]: A list of Replacement objects.
        """
        self.replacements = replacements

    def pre_cache_save(self, message, history):
        for replacement in self.replacements:
            message.content = message.content.replace(
                replacement.value, replacement.key
            )
        return message

    def post_cache_retrieval(self, message, history):
        for replacement in self.replacements:
            message.content = message.content.replace(
                replacement.key, replacement.value
            )
        return message
