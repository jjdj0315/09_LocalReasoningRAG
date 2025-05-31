import streamlit as st

from utils.session import session_control
from utils.create_dir import create_dir
from utils.upload import upload_file
from utils.creat_compression_retriever import creat_compression_retriever
from utils.node import create_app

from utils.print_message import print_messages
from utils.add_message import add_message


# 세션 초기화
session_control()
# 디렉터리 생성
create_dir()

st.set_page_config(page_title="100% 오픈모델 RAG , made by DJ")
st.title("100% 오픈모델 LANGGRAPH RAG, by DJ")

with st.sidebar:
    file = st.file_uploader("PDF 파일 업로드", type=["pdf"])
    if file:
        FILE_PATH = upload_file(file)
        st.session_state["compression_retriever"] = creat_compression_retriever(
            FILE_PATH
        )  # 이 부분에서 retriever가 세션에 저장됨

        app = create_app()
        # 이 두 줄이 Streamlit 앱의 사이드바에 그래프를 시각화합니다.
        graph_bytes = app.get_graph().draw_mermaid_png()
        st.image(graph_bytes, caption="Chatbot Graph")
        st.session_state["app"] = app  # app 객체를 세션에 저장

# 이전 메시지가 출력력
print_messages()

user_input = st.chat_input()
if user_input:
    # 사용자 메시지 화면에 출력 및 세션에 추가
    st.chat_message("user").write(user_input)
    add_message("user", user_input)

    if "app" in st.session_state and "compression_retriever" in st.session_state:
        app = st.session_state["app"]
        compression_retriever = st.session_state["compression_retriever"]

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
    else:
        st.warning("PDF 파일을 먼저 업로드하고 그래프를 초기화해주세요.")
