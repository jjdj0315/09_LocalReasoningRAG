# app.py

import streamlit as st
from dotenv import load_dotenv

from utils.session import session_control
from utils.uuid import random_uuid
from utils.create_dir import create_dir
from utils.creat_compression_retriever import creat_compression_retriever
from utils.upload import upload_file
from utils.node import create_app
from utils.add_message import add_message
from utils.print_messages import print_messages
from utils.handler import stream_handler

load_dotenv()
session_control()
create_dir()

st.title("LOCAL RAG LLM")
st.markdown("온프라미스 RAG LLM입니다., 멀티턴대화를 지원합니다.")

# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    st.markdown("# **made by JDJ**")
    selected_loader = st.radio("로더 선택", ["docling", "PDFPlumber"], index=0)
    file = st.file_uploader("PDF 파일 업로드", type=["pdf"])
    apply_btn = st.button("설정 완료", type="primary")


if clear_btn:
    st.session_state["messages"] = []
    st.session_state["thread_id"] = random_uuid()
    st.session_state["app"] = None
    st.session_state["config"] = None
    st.session_state["compression_retriever"] = None
    st.success("대화 및 설정이 초기화되었습니다.")

warning_msg = st.empty()
print_messages()

if apply_btn:
    if file:
        with st.spinner("문서 처리 및 RAG 앱 설정 중..."):
            FILE_PATH = upload_file(file)
            st.session_state["compression_retriever"] = creat_compression_retriever(
                FILE_PATH, selected_loader
            )
            st.session_state["app"] = create_app()
            st.session_state["config"] = {
                "configurable": {"thread_id": st.session_state["thread_id"]}
            }
            st.success("RAG 앱 설정 완료!")
    else:
        warning_msg.warning("PDF 파일을 먼저 업로드해주세요.")


user_input = st.chat_input("궁금한 내용을 물어보세요")


if user_input:
    if (
        st.session_state.get("app") is not None
        and st.session_state.get("config") is not None
    ):

        add_message("user", user_input)
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            # 각 UI 요소를 위한 placeholder를 명시적으로 생성
            # 순서가 중요합니다. Streamlit은 코드 순서대로 UI를 그립니다.
            node_status_placeholder = (
                st.empty()
            )  # "분류 중", "검색 중" 등의 상태 메시지
            retrieved_docs_expander_placeholder = st.empty()  # 검색된 문서 expander
            thinking_placeholder = st.empty()  # 추론 과정 스트리밍
            answer_placeholder = st.empty()  # 최종 답변 스트리밍

            inputs = {
                "query": user_input,
                "messages": [("human", user_input)],
                "documents": [],
                "thinking": "",
                "answer": "",
                "mode": "",
            }

            try:
                langgraph_stream_generator = st.session_state["app"].stream(
                    inputs, st.session_state["config"], stream_mode="updates"
                )

                # stream_handler에 모든 placeholder를 전달
                retrieved_docs, final_answer, final_thinking = stream_handler(
                    node_status_placeholder,
                    retrieved_docs_expander_placeholder,
                    thinking_placeholder,
                    answer_placeholder,
                    langgraph_stream_generator,
                )

                # 스트리밍이 완료된 후, 최종 결과들을 session_state["messages"]에 추가
                # 이때는 각 placeholder가 이미 최종 내용을 표시했으므로,
                # add_message는 대화 기록 저장을 위한 용도로만 사용됩니다.

                if retrieved_docs:
                    add_message(
                        "assistant",
                        retrieved_docs,
                        "tool_result",
                        "문서 검색 결과",
                    )

                if final_thinking:
                    add_message(
                        "assistant",
                        f"**🧠 추론 과정:**\n{final_thinking}",
                        "text",
                        "추론 과정",
                    )

                if final_answer:
                    add_message("assistant", final_answer, "text")

            except Exception as e:
                error_message = f"오류 발생: {e}"
                st.error(f"LangGraph 실행 중 오류 발생: {e}")
                add_message("assistant", error_message, "text")

    else:
        warning_msg.warning("PDF 파일을 업로드하고 설정을 완료해주세요.")
