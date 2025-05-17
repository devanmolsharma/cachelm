from cachelm.middlewares.middleware import Middleware


class Replacement:
    """
    A class representing a replacement operation.
    """

    def __init__(
        self,
        key: str,
        value: str,
    ):
        """
        Initialize the Replacement object.

        Args:
            key (str): The string to be replaced.
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

    def pre_cache(self, history):
        for replacement in self.replacements:
            for message in history:
                if message.content == replacement.key:
                    message.content = replacement.value
        return history

    def post_cache(self, history):
        return history
