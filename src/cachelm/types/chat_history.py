import json


class ToolCall:
    """
    Class to represent a tool call in the chat history.
    """

    def __init__(self, tool: str, args: list):
        self.tool = tool
        self.args = args

    def __repr__(self):
        return f"ToolCall(tool={self.tool}, args={self.args})"

    def to_json_str(self):
        """
        Convert the tool call to a JSON string.
        """
        return json.dumps({"tool": self.tool, "args": self.args})

    def to_json(self):
        """
        Convert the tool call to a JSON object.
        """
        return {"tool": self.tool, "args": self.args}

    @staticmethod
    def from_json_str(json_str: str):
        """
        Create a ToolCall object from a JSON string.
        """
        data = json.loads(json_str)
        return ToolCall(tool=data.get("tool", ""), args=data.get("args", []))

    @staticmethod
    def from_json(json_obj: dict):
        """
        Create a ToolCall object from a JSON object.
        """
        return ToolCall(tool=json_obj.get("tool", ""), args=json_obj.get("args", []))


class Message:
    """
    Class to represent a message in the chat history.
    """

    def __init__(
        self, role: str, content: str, tool_calls: list[ToolCall] | None = None
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls

    def __repr__(self):
        return f"Message(role={self.role}, content={self.content})"

    def to_json_str(self):
        """
        Convert the message to a JSON string.
        """
        return json.dumps(
            {
                "role": self.role,
                "content": self.content,
                "tool_calls": [
                    tool_call.to_json() for tool_call in (self.tool_calls or [])
                ],
            },
        )

    def to_formatted_str(self):
        """
        Convert the message to a formatted string.
        example:
        >>> message = Message(role="user", content="Hello")
        >>> message.to_formatted_str()
        'user: Hello'
        >>> message = Message(role="assistant", content="Hi there!", tool_calls=[{"tool": "calculator", "args": [1, 2]}])
        >>> message.to_formatted_str()
        'assistant: Hi there! (Tool calls: [{"tool": "calculator", "args": [1, 2]}])'
        """
        if self.content != "" and self.tool_calls is None:
            return f"{self.role}: {self.content}"
        elif self.content == "":
            return f""

        return f"{self.role}: {self.content} (Tool calls: {json.dumps([tool_call.to_json() for tool_call in (self.tool_calls or [])])})"

    @staticmethod
    def from_json_str(json_str: str):
        """
        Create a Message object from a JSON string.
        """
        data = json.loads(json_str)
        return Message(
            role=data.get("role", ""),
            content=data.get("content", ""),
            tool_calls=[
                ToolCall.from_json(tool_call)
                for tool_call in data.get("tool_calls", [])
            ],
        )


class ChatHistory:
    """
    Class to represent the chat history.
    """

    def __init__(self):
        self.messages: list[Message] = []

    def add_user_message(self, message: str):
        """
        Add a user message to the chat history.
        """
        self.messages.append(Message(role="user", content=message))

    def add_assistant_message(self, message: str):
        """
        Add an assistant message to the chat history.
        """
        self.messages.append(Message(role="assistant", content=message))

    def set_messages(self, messages: list[Message]):
        """
        Set the messages in the chat history.
        """
        self.messages = messages

    def get_messages(self, length: int = 0) -> list[Message]:
        """
        Get the messages from the chat history.
        If length is 0, return all messages.
        If length is greater than the number of messages, return the last 'length' messages.
        If length is less than the number of messages, return the first 'length' messages. Prepend with empty strings if necessary.
        Example:
        >>> chat_history = ChatHistory()
        >>> chat_history.add_user_message("Hello")
        >>> chat_history.add_assistant_message("Hi there!")
        >>> chat_history.get_messages()
        [Message(role='assistant', content='Hi there!'), Message(role='user', content='Hello')]
        >>> chat_history.get_messages(1)
        [Message(role='assistant', content='Hi there!')]
        >>> chat_history.get_messages(3)
        [Message(role='', content=''), Message(role='assistant', content='Hi there!'), Message(role='user', content='Hello')]
        >>> chat_history.get_messages(4)
        [Message(role='', content=''), Message(role='', content=''), Message(role='assistant', content='Hi there!'), Message(role='user', content='Hello')]
        """
        if length == 0:
            length = len(self.messages)
        if length > len(self.messages):
            # Pad with empty strings at the beginning
            pad = [Message("", "")] * (length - len(self.messages))
            return pad + self.messages
        else:
            return self.messages[-length:]

    def remove_message(self, index: int):
        """
        Remove a message from the chat history.
        """
        if 0 <= index < len(self.messages):
            del self.messages[index]
        else:
            raise IndexError("Index out of range")

    def to_formatted_str(self):
        """
        Convert the chat history to a formatted string.
        """
        return "\n".join([message.to_formatted_str() for message in self.messages])

    def clear(self):
        """
        Clear the chat history.
        """
        self.messages = []

    def __len__(self):
        """
        Get the length of the chat history.
        """
        return len(self.messages)

    def __getitem__(self, index: int):
        """
        Get an item from the chat history.
        """
        return self.messages[index]
