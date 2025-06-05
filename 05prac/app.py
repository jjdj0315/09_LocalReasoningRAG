import streamlit as st

from utils.uuid import random_uuid
from utils.print_messages import print_messages

st.title("LOCAL LLM RAG")
st.markdown("OnPromise LLM RAG")

# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    st.markdown("# **made by JDJ**")
    selected_loader = st.radio(
        "로더 선택", ["docling", "PDFPlumber", "정대진"], index=0
    )
    file = st.file_uploader("pdf파일 업로드", type=["pdf"])
    apply_btn = st.button("설정완료", type="primary")


# 초기화 버튼
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["thread_id"] = random_uuid()
    if "app" in st.session_state:
        del st.session_state["app"]
    if "compression_retriever" in st.session_state:
        del st.session_state["compression_retriever"]
    st.rerun()

# 사용자 입력
user_input = st.chat_input("궁금한 내용을 물어보세요")

# 경고메시지 영역
warning_msg = st.empty()

# 이전 대화 기록 출력
print_messages()

#설정 버튼이 눌리면
if apply_btn:
    if file:
        with st.spinner("파일 처리 및 RAG 설정 중"):
            st.session_state["compression_retriever"] = create_compression_retriever(FILE_PATH, selected_loader)
            