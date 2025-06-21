import openai
from typing import Generic, TypeVar
from cachelm.adaptors.adaptor import Adaptor
from loguru import logger

T = TypeVar("T", openai.OpenAI, openai.AsyncOpenAI)


class OpenAIAdaptor(Adaptor[T], Generic[T]):
    def __init__(self, *args, **kwargs):
        """
        Initialize the OpenAIAdaptor with the OpenAI module.
        Args:
            *args: Positional arguments for the Adaptor.
            **kwargs: Keyword arguments for the Adaptor.
        """
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def get_adapted(self):
        """
        Get the adapted OpenAI API.
        """
        base = self.module

        if isinstance(base, openai.OpenAI):
            from cachelm.adaptors.openai.sync_openai import SyncOpenAIAdaptor

            logger.warning(
                "OpenAIAdaptor is deprecated. "
                "import cachelm.adaptors.openai.sync_openai instead."
            )
            adaptor = SyncOpenAIAdaptor(*self.args, **self.kwargs)
            return adaptor.get_adapted()

        elif isinstance(base, openai.AsyncOpenAI):
            from cachelm.adaptors.openai.async_openai import AsyncOpenAIAdaptor

            logger.warning(
                "OpenAIAdaptor is deprecated. "
                "import cachelm.adaptors.openai.async_openai instead."
            )
            adaptor = AsyncOpenAIAdaptor(*self.args, **self.kwargs)
            return adaptor.get_adapted()
        else:
            raise TypeError(
                f"Unsupported OpenAI module type: {type(base)}. "
                "Expected openai.OpenAI or openai.AsyncOpenAI."
            )
