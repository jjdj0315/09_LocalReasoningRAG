import streamlit as st


def session_control():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
