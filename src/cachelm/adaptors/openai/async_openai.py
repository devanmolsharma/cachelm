from uuid import uuid4
import openai
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import openai.types.chat.chat_completion_chunk as chat_completion_chunk
from typing import Any, Literal
from cachelm.adaptors.adaptor import Adaptor
from openai import NotGiven
from loguru import logger
from cachelm.utils.chat_history import Message, ToolCall
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)


class AsyncOpenAIAdaptor(Adaptor[openai.AsyncOpenAI]):
    async def _preprocess_chat(self, *args, **kwargs) -> ChatCompletion | None:
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
        cached = await self.get_cache_async()
        if cached is not None:
            logger.info("Found cached response")
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
                                if cached.tool_calls
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

    async def _preprocess_streaming_chat_async(
        self, *args, **kwargs
    ) -> openai.AsyncStream[chat_completion_chunk.ChatCompletionChunk] | None:
        if kwargs.get("messages") is not None:
            logger.info("Setting history")
            messages = [
                (
                    msg
                    if isinstance(msg, Message)
                    else Message(
                        role=msg.get("role", ""),
                        content=msg.get("content", ""),
                        tool_calls=[
                            ToolCall(
                                tool_call.get("function", {}).get("name", ""),
                                tool_call.get("function", {}).get("arguments", ""),
                            )
                            for tool_call in msg.get("tool_calls", [])
                        ],
                    )
                )
                for msg in kwargs["messages"]
            ]
            self.set_history(messages)
        cached = await self.get_cache_async()
        if cached is not None:
            logger.info("Found cached response")

            async def cached_iterator():
                splitted_content = cached.content.split(" ")
                for i in range(len(splitted_content)):
                    content_chunk = " " + splitted_content[i]
                    tool_calls = (
                        [
                            chat_completion_chunk.ChoiceDeltaToolCall(
                                id=str(uuid4()),
                                index=0,
                                function=chat_completion_chunk.ChoiceDeltaToolCallFunction(
                                    name=tool_call.tool,
                                    arguments=tool_call.args,
                                ),
                            )
                            for tool_call in cached.tool_calls
                        ]
                        if cached.tool_calls is not None
                        and i == len(splitted_content) - 1
                        else None
                    )
                    yield chat_completion_chunk.ChatCompletionChunk(
                        id=str(uuid4()),
                        choices=[
                            chat_completion_chunk.Choice(
                                index=0,
                                finish_reason="stop",
                                delta=chat_completion_chunk.ChoiceDelta(
                                    role=cached.role,
                                    content=content_chunk,
                                    tool_calls=tool_calls,
                                ),
                            )
                        ],
                        created=0,
                        model=kwargs["model"],
                        object="chat.completion.chunk",
                    )

            return cached_iterator()
        return None

    async def _postprocess_streaming_chat_async(
        self, response: openai.AsyncStream[chat_completion_chunk.ChatCompletionChunk]
    ) -> Any:
        full_content = ""
        tool_name = None
        tool_params = ""
        tool_calls = None
        role = "assistant"
        async for chunk in response:
            if chunk.choices is None or len(chunk.choices) == 0:
                logger.warning("No choices in completion, skipping postprocessing.")
                yield chunk
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                full_content += delta.content
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if tool_call.function.name:
                        tool_name = tool_call.function.name
                    if tool_call.function.arguments:
                        tool_params += tool_call.function.arguments
            if delta.role:
                role = delta.role
            yield chunk
        if tool_name and tool_params:
            tool_calls = [ToolCall(tool_name, tool_params)]
        await self.add_assistant_message_async(
            Message(role=role, content=full_content, tool_calls=tool_calls)
        )

    async def _postprocess_chat(self, completion: ChatCompletion) -> None:
        if completion.choices is None or len(completion.choices) == 0:
            logger.warning("No choices in completion, skipping postprocessing.")
            return
        msg = completion.choices[0].message
        message_obj = Message(
            role=msg.role,
            content=msg.content,
            tool_calls=(
                [
                    ToolCall(tool_call.function.name, tool_call.function.arguments)
                    for tool_call in msg.tool_calls
                ]
                if msg.tool_calls
                else None
            ),
        )
        await self.add_assistant_message_async(message_obj)

    def get_adapted(self) -> openai.AsyncOpenAI:
        base = self.module
        completions = base.chat.completions
        adaptorSelf = self

        class AdaptedCompletions(completions.__class__):
            async def create_with_stream(self, *args, stream: Literal[True], **kwargs):
                cached = await adaptorSelf._preprocess_streaming_chat_async(
                    *args, stream=stream, **kwargs
                )
                if cached:
                    return cached
                res = await super().create(*args, stream=stream, **kwargs)
                return adaptorSelf._postprocess_streaming_chat_async(res)

            async def create_without_stream(self, *args, stream=NotGiven, **kwargs):
                cached = await adaptorSelf._preprocess_chat(
                    *args, stream=stream, **kwargs
                )
                if cached:
                    return cached
                res = await super().create(*args, **kwargs)
                await adaptorSelf._postprocess_chat(res)
                return res

            async def create(self, *args, **kwargs):
                if kwargs.get("stream") is True:
                    return await self.create_with_stream(*args, **kwargs)
                return await self.create_without_stream(*args, **kwargs)

        base.chat.completions = AdaptedCompletions(client=completions._client)
        return base
