from langchain_core.messages import ChatMessage
from attr import dataclass


@dataclass
class ChatMessageWithType:
    chat_message: ChatMessage
    msg_type: str
    tool_name: str
