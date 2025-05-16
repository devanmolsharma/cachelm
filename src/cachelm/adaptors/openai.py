from uuid import uuid4
import openai
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import openai.types.chat.chat_completion_chunk as chat_completion_chunk
from typing import Any, Generic, Literal, TypeVar, overload
from cachelm.adaptors.adaptor import Adaptor
from openai import NotGiven
from loguru import logger

T = TypeVar("T", openai.OpenAI, openai.AsyncOpenAI)


class OpenAIAdaptor(Adaptor[T], Generic[T]):

    def _preprocess_chat(self, *args, **kwargs) -> ChatCompletion | None:
        """
        Preprocess the chat messages to set the history.
        """
        if kwargs.get("messages") is not None:
            # Check if the message is already in the cache
            # and return the cached response if it exists
            logger.info("Setting history")
            self.setHistory(kwargs["messages"])
        cached = self.get_cache()
        if cached is not None:
            logger.info("Found cached response")
            res = ChatCompletion(
                id=str(uuid4()),
                choices=[
                    Choice(
                        index=0,
                        finish_reason="stop",
                        message=ChatCompletionMessage(
                            role="assistant",
                            content=cached,
                        ),
                    )
                ],
                created=0,
                model=kwargs["model"],
                object="chat.completion",
            )
            return res
        return None

    def _preprocess_streaming_chat(
        self, *args, **kwargs
    ) -> openai.Stream[chat_completion_chunk.ChatCompletionChunk] | None:
        """
        Preprocess the streaming chat messages to set the history.
        """
        if kwargs.get("messages") is not None:
            # Check if the message is already in the cache
            # and return the cached response if it exists
            logger.info("Setting history")
            self.setHistory(kwargs["messages"])
        cached = self.get_cache()
        if cached is not None:
            logger.info("Found cached response")

            def cached_iterator():
                yield chat_completion_chunk.ChatCompletionChunk(
                    id=str(uuid4()),
                    choices=[
                        chat_completion_chunk.Choice(
                            index=0,
                            finish_reason="stop",
                            delta=chat_completion_chunk.ChoiceDelta(
                                role="assistant",
                                content=cached,
                            ),
                        )
                    ],
                    created=0,
                    model=kwargs["model"],
                    object="chat.completion.chunk",
                )

            return cached_iterator()
        return None

    def _preprocess_streaming_chat_async(
        self, *args, **kwargs
    ) -> openai.AsyncStream[chat_completion_chunk.ChatCompletionChunk] | None:
        """
        Preprocess the streaming chat messages to set the history.
        """
        if kwargs.get("messages") is not None:
            # Check if the message is already in the cache
            # and return the cached response if it exists
            logger.info("Setting history")
            self.setHistory(kwargs["messages"])
        cached = self.get_cache()
        if cached is not None:
            logger.info("Found cached response")

            async def cached_iterator():
                yield chat_completion_chunk.ChatCompletionChunk(
                    id=str(uuid4()),
                    choices=[
                        chat_completion_chunk.Choice(
                            index=0,
                            finish_reason="stop",
                            delta=chat_completion_chunk.ChoiceDelta(
                                role="assistant",
                                content=cached,
                            ),
                        )
                    ],
                    created=0,
                    model=kwargs["model"],
                    object="chat.completion.chunk",
                )

            return cached_iterator()
        return None

    def _postprocess_chat(self, completion: ChatCompletion) -> None:
        """
        Postprocess the chat messages to set the history.
        """
        self.add_assistant_message(completion.choices[0].message.content)

    def _postprocess_streaming_chat(
        self, response: openai.Stream[chat_completion_chunk.ChatCompletionChunk]
    ) -> Any:
        """
        Postprocess the streaming chat messages to set the history.
        """
        full_response = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                full_response += content
            yield chunk
        self.add_assistant_message(full_response)

    async def _postprocess_streaming_chat_async(
        self, response: openai.AsyncStream[chat_completion_chunk.ChatCompletionChunk]
    ) -> Any:
        """
        Preprocess the streaming chat messages to set the history.
        """
        full_response = ""
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                full_response += content
            yield chunk
        self.add_assistant_message(full_response)

    def _get_adapted_openai_sync(adaptorSelf, module: openai.OpenAI) -> openai.OpenAI:
        """
        Get the adapted OpenAI API for synchronous calls.
        """
        base = module
        completions = base.chat.completions

        class AdaptedCompletions(completions.__class__):
            def create_with_stream(
                self,
                *args,
                stream: Literal[True],
                **kwargs,
            ):
                logger.info("Creating completion")
                cached = adaptorSelf._preprocess_streaming_chat(
                    *args, stream=stream, **kwargs
                )
                if cached is not None:
                    logger.info("Found cached response")
                    return cached
                parent = super()
                res = parent.create(*args, stream=stream, **kwargs)
                iterator = adaptorSelf._postprocess_streaming_chat(res)
                return iterator

            def create_without_stream(
                self,
                *args,
                stream: Literal[False] | NotGiven | None = NotGiven,
                **kwargs,
            ):
                logger.info("Creating completion")
                cached = adaptorSelf._preprocess_chat(*args, stream=stream, **kwargs)
                if cached is not None:
                    logger.info("Found cached response")
                    return cached
                parent = super()
                res = parent.create(*args, **kwargs)
                adaptorSelf._postprocess_chat(res)
                logger.info("Storing response in cache")
                return res

            def create(
                self,
                *args,
                **kwargs,
            ):
                logger.info(
                    f"Creating completion with streaming = {kwargs.get('stream')}"
                )
                if kwargs.get("stream") is True:
                    return self.create_with_stream(*args, **kwargs)
                else:
                    return self.create_without_stream(*args, **kwargs)

        # Replace the original completions with the adapted one
        base.chat.completions = AdaptedCompletions(
            client=base.chat.completions._client,
        )

        return base

    def _get_adapted_openai_async(
        adaptorSelf, module: openai.AsyncOpenAI
    ) -> openai.AsyncOpenAI:
        """
        Get the adapted OpenAI API for asynchronous calls.
        """
        base = module
        completions = base.chat.completions

        class AdaptedCompletions(completions.__class__):
            async def create_with_stream(
                self,
                *args,
                stream: Literal[True],
                **kwargs,
            ):
                logger.info("Creating completion")
                cached = adaptorSelf._preprocess_streaming_chat_async(
                    *args, stream=stream, **kwargs
                )
                if cached is not None:
                    logger.info("Found cached response")
                    return cached
                parent = super()
                res = await parent.create(*args, stream=stream, **kwargs)
                iterator = adaptorSelf._postprocess_streaming_chat_async(res)
                return iterator

            async def create_without_stream(
                self,
                *args,
                stream: Literal[False] | NotGiven | None = NotGiven,
                **kwargs,
            ):
                logger.info("Creating completion")
                cached = adaptorSelf._preprocess_chat(*args, stream=stream, **kwargs)
                if cached is not None:
                    logger.info("Found cached response")
                    return cached
                parent = super()
                res = await parent.create(*args, **kwargs)
                adaptorSelf._postprocess_chat(res)
                logger.info("Storing response in cache")
                return res

            async def create(
                self,
                *args,
                **kwargs,
            ):
                logger.info(
                    f"Creating completion with streaming = {kwargs.get('stream')}"
                )
                if kwargs.get("stream") is True:
                    return await self.create_with_stream(*args, **kwargs)
                else:
                    return await self.create_without_stream(*args, **kwargs)

        # Replace the original completions with the adapted one
        base.chat.completions = AdaptedCompletions(
            client=base.chat.completions._client,
        )

        return base

    # def _generate_fak_streaming_response(

    def get_adapted(self):
        """
        Get the adapted OpenAI API.
        """
        base = self.module

        if isinstance(base, openai.OpenAI):
            return self._get_adapted_openai_sync(base)

        elif isinstance(base, openai.AsyncOpenAI):
            return self._get_adapted_openai_async(base)
        else:
            raise TypeError(
                f"Unsupported OpenAI module type: {type(base)}. "
                "Expected openai.OpenAI or openai.AsyncOpenAI."
            )
