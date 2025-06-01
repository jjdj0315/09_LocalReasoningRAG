import streamlit as st

from utils.uuid import random_uuid
from utils.session import session_control
from utils.create_dir import create_dir
from utils.creat_compression_retriever import creat_compression_retriever
from utils.upload import upload_file
from utils.node import create_app
from utils.add_message import add_message

session_control()
create_dir()

st.title("LOCAL RAG LLM")
st.markdown("온프라미스 RAG LLM입니다., 멀티턴대화를 지원합니다.")

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼
    clear_btn = st.button("대화 초기화")

    st.markdown("# **made by JDJ**")

    # 로더 선택
    selected_loader = st.radio(
        "로더 선택", ["docling", "PDFPlumber", "정대진"], index=0
    )
    # selected_loader = st.selectbox(
    #     "로더 선택", ["docling", "PDFPlumber", "정대진"], index=0
    # )
    # 파일 업로드
    file = st.file_uploader("PDF 파일 업로드", type=["pdf"])

    # 설정 버튼
    apply_btn = st.button("설정 완료", type="primary")


# 초기화 버튼
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["thread_id"] = random_uuid()

# 설정 버튼이 눌리면
if apply_btn:
    if file:
        FILE_PATH = upload_file(file)

        st.session_state["compression_retriever"] = creat_compression_retriever(
            FILE_PATH, selected_loader
        )  # 이 부분에서 retriever가 세션에 저장됨
        compression_retriever = st.session_state["compression_retriever"]
    app = create_app()
    st.session_state["app"] = app
    st.session_state["thread_id"] = random_uuid()

# 사용자 입력
user_input = st.chat_input("궁금한 내용을 물어보세요")

# 경고메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()


user_input = st.chat_input()
if user_input:
    # 사용자 메시지 화면에 출력 및 세션에 추가
    st.chat_message("user").write(user_input)
    add_message("user", user_input)

    if "app" in st.session_state:
        app = st.session_state["app"]

        # app.py 의 user_input 처리 부분
        try:
            inputs = {
                "query": user_input,
                "documents": [],
                "thinking": "",
                "answer": "",
            }
            result = app.invoke(inputs, config=st.session_state["config"])
            print(
                f"LangGraph Invoke Result: {result}"
            )  # 이 부분을 추가하여 result의 전체 구조 확인
            ai_answer = result.get("answer", "답변을 생성하지 못했습니다.")
        except Exception as e:
            ai_answer = f"오류 발생: {e}"
            st.error(f"LangGraph 실행 중 오류 발생: {e}")

        # AI의 메시지가 출력력
        st.chat_message("ai").write(ai_answer)
        add_message("ai", ai_answer)
    # else:
    #     st.warning("PDF 파일을 먼저 업로드하고 그래프를 초기화해주세요.")
