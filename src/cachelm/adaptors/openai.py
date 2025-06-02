from uuid import uuid4
import openai
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import openai.types.chat.chat_completion_chunk as chat_completion_chunk
from typing import Any, Generic, Literal, TypeVar
from cachelm.adaptors.adaptor import Adaptor
from openai import NotGiven
from loguru import logger
from cachelm.types.chat_history import Message, ToolCall  # Use correct import
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

T = TypeVar("T", openai.OpenAI, openai.AsyncOpenAI)


class OpenAIAdaptor(Adaptor[T], Generic[T]):
    def _preprocess_chat(self, *args, **kwargs) -> ChatCompletion | None:
        """
        Preprocess the chat messages to set the history.
        """
        if kwargs.get("messages") is not None:
            logger.info("Setting history")
            # Convert dicts to Message objects if needed
            messages = [
                (
                    msg
                    if isinstance(msg, Message)
                    else Message(
                        role=msg.get("role", ""),
                        content=msg.get("content", ""),
                        tool_calls=msg.get("tool_calls"),
                    )
                )
                for msg in kwargs["messages"]
            ]
            self.set_history(messages)
        cached = self.get_cache()
        if cached is not None:
            logger.info("Found cached response")
            # cached is a Message object
            res = ChatCompletion(
                id=str(uuid4()),
                choices=[
                    Choice(
                        index=0,
                        finish_reason="stop",
                        message=ChatCompletionMessage(
                            role=cached.role,
                            content=cached.content,
                            tool_calls=(
                                [
                                    ChatCompletionMessageToolCall(
                                        id=str(uuid4()),
                                        function=Function(
                                            name=tool_call.tool,
                                            arguments=tool_call.args,
                                        ),
                                    )
                                    for tool_call in cached.tool_calls
                                ]
                                if cached.tool_calls is not None
                                else None
                            ),
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
            logger.info("Setting history")
            messages = [
                (
                    msg
                    if isinstance(msg, Message)
                    else Message(
                        role=msg.get("role", ""),
                        content=msg.get("content", ""),
                        tool_calls=msg.get("tool_calls"),
                    )
                )
                for msg in kwargs["messages"]
            ]
            self.set_history(messages)
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
                                role=cached.role,
                                content=cached.content,
                                tool_calls=(
                                    [
                                        chat_completion_chunk.ChoiceDeltaToolCall(
                                            id=str(uuid4()),
                                            function=chat_completion_chunk.ChoiceDeltaToolCallFunction(
                                                name=tool_call.tool,
                                                arguments=tool_call.args,
                                            ),
                                        )
                                        for tool_call in cached.tool_calls
                                    ]
                                    if cached.tool_calls is not None
                                    else None
                                ),
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
            logger.info("Setting history")
            messages = [
                (
                    msg
                    if isinstance(msg, Message)
                    else Message(
                        role=msg.get("role", ""),
                        content=msg.get("content", ""),
                        tool_calls=msg.get("tool_calls"),
                    )
                )
                for msg in kwargs["messages"]
            ]
            self.set_history(messages)
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
                                role=cached.role,
                                content=cached.content,
                                tool_calls=(
                                    [
                                        chat_completion_chunk.ChoiceDeltaToolCall(
                                            function=chat_completion_chunk.ChoiceDeltaToolCallFunction(
                                                name=tool_call.tool,
                                                arguments=tool_call.args,
                                            )
                                        )
                                        for tool_call in cached.tool_calls
                                    ]
                                    if cached.tool_calls is not None
                                    else None
                                ),
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
        msg = completion.choices[0].message
        message_obj = Message(
            role=msg.role,
            content=msg.content,
            tool_calls=(
                [
                    ToolCall(tool_call.function.name, tool_call.function.arguments)
                    for tool_call in msg.tool_calls
                ]
                if msg.tool_calls is not None
                else None
            ),
        )
        self.add_assistant_message(message_obj)

    def _postprocess_streaming_chat(
        self, response: openai.Stream[chat_completion_chunk.ChatCompletionChunk]
    ) -> Any:
        """
        Postprocess the streaming chat messages to set the history.
        """
        full_content = ""
        tool_calls = None
        role = "assistant"
        for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content is not None:
                full_content += delta.content
            if hasattr(delta, "tool_calls") and delta.tool_calls is not None:
                tool_calls = (
                    [
                        ToolCall(tool_call.function.name, tool_call.function.arguments)
                        for tool_call in delta.tool_calls
                    ]
                    if delta.tool_calls is not None
                    else None
                )
            if hasattr(delta, "role") and delta.role is not None:
                role = delta.role
            yield chunk
        message_obj = Message(
            role=role,
            content=full_content,
            tool_calls=tool_calls if tool_calls is not None else None,
        )
        self.add_assistant_message(message_obj)

    async def _postprocess_streaming_chat_async(
        self, response: openai.AsyncStream[chat_completion_chunk.ChatCompletionChunk]
    ) -> Any:
        """
        Postprocess the streaming chat messages to set the history.
        """
        full_content = ""
        tool_calls = None
        role = "assistant"
        async for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content is not None:
                full_content += delta.content
            if hasattr(delta, "tool_calls") and delta.tool_calls is not None:
                tool_calls = (
                    [
                        ToolCall(tool_call.function.name, tool_call.function.arguments)
                        for tool_call in delta.tool_calls
                    ]
                    if delta.tool_calls is not None
                    else None
                )
            if hasattr(delta, "role") and delta.role is not None:
                role = delta.role
            yield chunk
        message_obj = Message(
            role=role,
            content=full_content,
            tool_calls=tool_calls,
        )
        self.add_assistant_message(message_obj)

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

        base.chat.completions = AdaptedCompletions(
            client=base.chat.completions._client,
        )

        return base

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
