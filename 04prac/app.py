# app.py

import streamlit as st
from dotenv import load_dotenv

from utils.session import session_control
from utils.uuid import random_uuid
from utils.create_dir import create_dir  # create_dir() 호출
from utils.creat_compression_retriever import creat_compression_retriever
from utils.upload import upload_file  # upload_file() 호출
from utils.node import create_app
from utils.add_message import add_message
from utils.print_messages import print_messages
from utils.handler import stream_handler  # stream_handler 임포트

load_dotenv()
session_control()
create_dir()  # 디렉토리 생성 함수 호출

st.title("LOCAL RAG LLM")
st.markdown("온프라미스 RAG LLM입니다., 멀티턴대화를 지원합니다.")

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼
    clear_btn = st.button("대화 초기화")

    st.markdown("# **made by JDJ**")

    # 로더 선택
    selected_loader = st.radio(
        "로더 선택", ["docling", "PDFPlumber"], index=0  # "정대진" 옵션은 제거
    )

    # 파일 업로드
    file = st.file_uploader("PDF 파일 업로드", type=["pdf"])

    # 설정 버튼
    apply_btn = st.button("설정 완료", type="primary")


# 초기화 버튼
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["thread_id"] = random_uuid()
    st.session_state["app"] = None
    st.session_state["config"] = None
    st.session_state["compression_retriever"] = None  # retriever도 초기화
    st.success("대화 및 설정이 초기화되었습니다.")

# 경고메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 이전 대화기록출력
print_messages()  # 여기에 위치해야 함 (메시지 입력 전에 이미 출력되어 있어야 함)


# 설정 버튼이 눌리면
if apply_btn:
    if file:
        with st.spinner("문서 처리 및 RAG 앱 설정 중..."):
            FILE_PATH = upload_file(file)  # 파일 업로드 유틸 함수 사용
            st.session_state["compression_retriever"] = creat_compression_retriever(
                FILE_PATH, selected_loader
            )  # 이 부분에서 retriever가 세션에 저장됨

            # app은 retriever가 생성된 후에 생성되어야 함
            st.session_state["app"] = create_app()
            st.session_state["config"] = {
                "configurable": {"thread_id": st.session_state["thread_id"]}
            }
            st.success("RAG 앱 설정 완료!")
    else:
        warning_msg.warning("PDF 파일을 먼저 업로드해주세요.")

# 사용자 입력
user_input = st.chat_input("궁금한 내용을 물어보세요")  # 이 user_input이 사용되어야 함.


if user_input:  # 새로운 user_input 변수 사용
    # 앱이 설정되어 있는지 확인
    if (
        st.session_state.get("app") is not None
        and st.session_state.get("config") is not None
    ):

        # 사용자 입력 메시지를 먼저 session_state에 추가하고 화면에 표시
        add_message("user", user_input)
        st.chat_message("user").write(user_input)

        # 어시스턴트 메시지가 표시될 공간
        with st.chat_message("assistant"):
            container = st.empty()  # 스트리밍 메시지가 여기에 표시됩니다.

            inputs = {
                "query": user_input,
                "messages": [
                    ("human", user_input)
                ],  # 대화 기록 (LangGraph messages 필드)
                "documents": [],
                "thinking": "",
                "answer": "",
                "mode": "",
            }

            try:
                # LangGraph 앱을 스트리밍 모드로 실행
                # stream_mode="updates"로 노드 상태 변화를 스트리밍 받음
                langgraph_stream_generator = st.session_state["app"].stream(
                    inputs,
                    st.session_state["config"],
                    stream_mode="updates",  # <-- "updates" 모드 사용
                )

                # stream_handler를 호출하여 스트리밍 결과 처리 및 최종 결과 반환
                retrieved_docs, final_answer, final_thinking = stream_handler(
                    container, langgraph_stream_generator
                )

                # 스트리밍이 완료된 후, 최종 결과들을 session_state["messages"]에 추가
                # 1. 검색된 문서 결과 추가 (Tool Result 타입으로)
                if retrieved_docs:
                    add_message(
                        "assistant",
                        retrieved_docs,  # Document 객체 리스트를 add_message로 전달
                        "tool_result",
                        "문서 검색 결과",
                    )

                # 2. 추론 과정 추가 (일반 텍스트 타입으로)
                if final_thinking:
                    add_message(
                        "assistant",
                        f"**🧠 추론 과정:**\n{final_thinking}",
                        "text",
                        "추론 과정",  # tool_name에 유사하게 추론과정임을 명시
                    )

                # 3. 최종 LLM 답변 추가
                if final_answer:
                    add_message("assistant", final_answer, "text")  # 일반 텍스트로 추가

            except Exception as e:
                error_message = f"오류 발생: {e}"
                st.error(f"LangGraph 실행 중 오류 발생: {e}")
                add_message("assistant", error_message, "text")  # 오류 메시지도 저장

    else:
        warning_msg.warning("PDF 파일을 업로드하고 설정을 완료해주세요.")
