from cachelm.middlewares.middleware import Middleware
import re

from cachelm.types.chat_history import Message


class Skipper(Middleware):
    """
    Middleware that skips saving messages to cache if they match any of the provided regex patterns.

    This is useful for filtering out messages that should not be cached based on regular expressions.

    Example:
        from cachelm.middlewares.skipper import Skipper

        # Create a Skipper middleware instance with regex patterns to skip
        skipper = Skipper(patterns=[r"skip_this.*", r"ignore_\d+"])
    """

    def __init__(self, patterns: list[str], function_calls: list[str] = []):
        """
        Initialize the Skipper middleware.

        Args:
            patterns: list[str]: A list of Replacement objects.
            function_calls: list[str]: A list of function calls to be skipped/
        """
        self.patterns = patterns

    def pre_cache_save(self, message, history):
        """
        Pre-cache save method to check if the message should be skipped.
        Args:
            message: The message to be checked.
            history: The history of messages.
        Returns:
            None if the message should be skipped, otherwise the message itself.
        """
        for pattern in self.patterns:
            if re.search(pattern, message.content):
                # If the pattern is found in the message content, skip saving it to cache
                return None
        return message

    def _should_skip_function_call(self, message: Message):
        """
        Check if the message is a function call that should be skipped.
        Args:
            message: The message to be checked.
        Returns:
            True if the message is a function call that should be skipped, otherwise False.
        """
        if (
            hasattr(message, "tool_calls")
            and message.tool_calls
            and isinstance(message.tool_calls, list)
        ):
            function_call = message.tool_calls[0]
            if function_call.tool in self.patterns:
                # If the function call matches any of the patterns, skip saving it to cache
                return True
        return False

    def post_cache_retrieval(self, message, history):
        return message
