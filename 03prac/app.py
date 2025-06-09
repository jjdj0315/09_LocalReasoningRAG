import streamlit as st
from IPython.display import Image, display

from utils.session import session_control
from utils.create_dir import create_dir
from utils.upload import upload_file
from utils.creat_compression_retriever import creat_compression_retriever
from utils.node import create_app

from utils.print_message import print_messages
from utils.add_message import add_message

from utils.state import RAGState # RAGState 임포트
from langchain_core.messages import HumanMessage, AIMessage # <-- 추가 임포트

# 세션 초기화
session_control()
# 디렉터리 생성
create_dir()

st.set_page_config(page_title="100% 오픈모델 RAG , made by DJ")
st.title("100% 오픈모델 LANGGRAPH RAG, by DJ")

with st.sidebar:
    file = st.file_uploader("PDF 파일 업로드", type=["pdf"])
    if file:
        # 파일이 업로드되었고, 아직 retriever가 세션에 없다면 (처음 업로드 시)
        if "compression_retriever" not in st.session_state:
            FILE_PATH = upload_file(file) 
            st.session_state["compression_retriever"] = creat_compression_retriever(FILE_PATH)
            st.success("PDF 파일 처리 및 임베딩 완료!")

        # app 객체도 한 번만 생성
        if "app" not in st.session_state:
            app = create_app()
            st.session_state["app"] = app
            graph_bytes = app.get_graph().draw_mermaid_png()
            st.image(graph_bytes, caption="Chatbot Graph")
        
#이전 메시지가 출력력
print_messages()

user_input = st.chat_input()
if user_input:
    # 사용자 메시지 화면에 출력 및 세션에 추가
    st.chat_message("user").write(user_input)
    add_message("user", user_input)

    # retriever와 app이 세션에 있는지 확인 (파일이 업로드되었는지 확인)
    if "app" in st.session_state and "compression_retriever" in st.session_state:
        app = st.session_state["app"]
        
        try:
            # st.session_state["messages"]에서 chat_history 생성
            # 마지막 사용자 입력은 query로 별도로 전달하므로, 이전 메시지만 포함
            current_chat_history = []
            for msg in st.session_state["messages"][:-1]: # 마지막 메시지(현재 사용자 입력) 제외
                if msg["role"] == "user":
                    current_chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "ai":
                    current_chat_history.append(AIMessage(content=msg["content"]))
            
            # LangGraph inputs에 chat_history 추가
            inputs = {
                "query": user_input,
                "documents": [],
                "thinking": "",
                "answer": "",
                "chat_history": current_chat_history # <-- chat_history 추가
            }
            
            # LangGraph 실행 시 config 전달 (thread_id는 session_control에서 관리)
            result = app.invoke(inputs, config=st.session_state["config"])
            ai_answer = result.get("answer", "답변을 생성하지 못했습니다.")
        except Exception as e:
            ai_answer = f"오류 발생: {e}"
            st.error(f"LangGraph 실행 중 오류 발생: {e}")

        # AI의 메시지가 출력력
        st.chat_message("ai").write(ai_answer)
        add_message("ai", ai_answer)
    else:
        st.warning("PDF 파일을 먼저 업로드하고 그래프를 초기화해주세요.")
