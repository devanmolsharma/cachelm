from loguru import logger
from cachelm.middlewares.middleware import Middleware


class Deduper(Middleware):
    """
    Middleware that returns None if the reply is already present in history.
    """

    def pre_cache_save(self, message, history):
        return message

    def post_cache_retrieval(self, message, history):
        # Check if message content is already in any previous message in history
        for past_message in history:
            if getattr(past_message, "content", None) == message.content:
                return None
        return message
