from uuid import uuid4
import openai
from openai.types import Completion
from typing import Type
from cachelm.src.adaptors.adaptor import Adaptor


class OpenAIAdaptor(Adaptor[openai.OpenAI]):
    def get_adapted(self):
        """
        Get the adapted OpenAI API.
        """
        base = self.module
        completions = base.completions
        adaptedSelf = self

        class AdaptedCompletions(completions.__class__):
            def create(self, *args, **kwargs):
                if kwargs.get("messages") is not None:
                    adaptedSelf.setHistory(kwargs["messages"])
                cached = adaptedSelf.get_cache()
                if cached is not None:
                    res = Completion()
                    res.choices = [adaptedSelf.history[-1]["content"]]
                    return res
                parent = super(AdaptedCompletions, self)
                res = parent.create(*args, **kwargs)
                choice = res.choices[0].text
                adaptedSelf.add_assistant_message(choice)
                return res

        # Replace the original completions with the adapted one
        base.completions = AdaptedCompletions()

        return base
