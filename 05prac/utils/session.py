# session
import streamlit as st


def session_control():
    # 대화기록을 저장하기 위한 용도로 생성
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # ReAct Agent 초기화
    if "react_agent" not in st.session_state:
        st.session_state["react_agent"] = None

    # include_domains 초기화
    if "include_domains" not in st.session_state:
        st.session_state["include_domains"] = []

    if "config" not in st.session_state:
        # LangGraph의 체크포인터가 필요로 하는 configurable 키를 초기화합니다.
        # thread_id를 문자열 '0'으로 설정합니다.
        st.session_state["config"] = {
            "configurable": {"thread_id": "0"}
        }  # <-- 이 부분을 "0" (문자열)으로 변경
