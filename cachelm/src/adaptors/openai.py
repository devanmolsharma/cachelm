from uuid import uuid4
import openai
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from typing import Type
from cachelm.src.adaptors.adaptor import Adaptor


class OpenAIAdaptor(Adaptor[openai.OpenAI]):
    def get_adapted(self):
        """
        Get the adapted OpenAI API.
        """
        base = self.module
        completions = base.chat.completions
        adaptedSelf = self

        class AdaptedCompletions(completions.__class__):
            def create(self, *args, **kwargs):
                print("Creating completion")
                if kwargs.get("messages") is not None:
                    # Check if the message is already in the cache
                    # and return the cached response if it exists
                    print("Setting history")
                    adaptedSelf.setHistory(kwargs["messages"])
                cached = adaptedSelf.get_cache()
                if cached is not None:
                    print("Found cached response")
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
                parent = super(AdaptedCompletions, self)
                res = parent.create(*args, **kwargs)
                choice = res.choices[0].message.content
                adaptedSelf.add_assistant_message(choice)
                print("Storing response in cache")
                return res

        # Replace the original completions with the adapted one
        base.chat.completions = AdaptedCompletions(
            client=base.chat.completions._client,
        )

        return base
