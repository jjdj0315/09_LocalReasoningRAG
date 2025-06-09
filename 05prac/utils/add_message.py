# utils/add_message.py

import streamlit as st
from langchain_core.messages import ChatMessage
from utils.dataclass import ChatMessageWithType
from utils.handler import (
    format_search_result,
)  # handler.py의 format_search_result를 임포트


def add_message(role, message, msg_type="text", tool_name=""):
    if msg_type == "text":
        st.session_state["messages"].append(
            ChatMessageWithType(
                chat_message=ChatMessage(role=role, content=message),
                msg_type="text",
                tool_name=tool_name,
            )
        )
    elif msg_type == "tool_result":
        # tool_result 타입일 때, message는 Document 객체 리스트로 가정합니다.
        # format_search_result는 이 Document 객체 리스트를 마크다운으로 포맷팅합니다.
        st.session_state["messages"].append(
            ChatMessageWithType(
                chat_message=ChatMessage(
                    role="assistant", content=format_search_result(message)
                ),
                msg_type="tool_result",
                tool_name=tool_name,
            )
        )
