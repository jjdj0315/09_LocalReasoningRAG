import streamlit as st


def session_control():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    if "config" not in st.session_state:
        # LangGraph의 체크포인터가 필요로 하는 configurable 키를 초기화합니다.
        # thread_id를 문자열 '0'으로 설정합니다.
        st.session_state["config"] = {"configurable": {"thread_id": "0"}} # <-- 이 부분을 "0" (문자열)으로 변경