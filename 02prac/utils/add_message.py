import streamlit as st
from langchain_core.messages import ChatMessage

def add_message(role, content):
    st.session_state.messages.append(ChatMessage(role= role, content= content))