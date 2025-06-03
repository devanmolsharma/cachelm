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
