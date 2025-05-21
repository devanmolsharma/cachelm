from cachelm.middlewares.middleware import Middleware
import re


class Skipper(Middleware):
    """
    Middleware that skips saving messages to cache if they match any of the provided regex patterns.

    This is useful for filtering out messages that should not be cached based on regular expressions.

    Example:
        from cachelm.middlewares.skipper import Skipper

        # Create a Skipper middleware instance with regex patterns to skip
        skipper = Skipper(patterns=[r"skip_this.*", r"ignore_\d+"])
    """

    def __init__(self, patterns: list[str]):
        """
        Initialize the Skipper middleware.

        Args:
            patterns: list[str]: A list of Replacement objects.
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

    def post_cache_retrieval(self, message, history):
        return message
