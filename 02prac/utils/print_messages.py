import streamlit as st

def print_messages():
    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)
