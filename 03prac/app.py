import streamlit as st


from utils.session import session_control
from utils.create_dir import create_dir
from utils.upload import upload_file
from utils.creat_compression_retriever import creat_compression_retriever


# from utils.create_rag_chain import create_rag_chain
from utils.print_message import print_messages
from utils.add_message import add_message
from utils.state import RAGState

# import os
# import torch
# import streamlit

# torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# # or simply:
# torch.classes.__path__ = []

session_control()
create_dir()

st.set_page_config(page_title="100% 오픈모델 RAG , made by DJ")
st.title("100% 오픈모델 LANGGRAPH RAG, by DJ")

with st.sidebar:
    file = st.file_uploader("PDF 파일 업로드", type=["pdf"])
    if file:
        FILE_PATH = upload_file(file) 
        answer_chain = creat_compression_retriever(FILE_PATH)
        st.session_state["chain"] = answer_chain

print_messages()

user_input = st.chat_input()
if user_input:
    if "chain" in st.session_state:
        chain = st.session_state["chain"]

        st.chat_message("user").write(user_input)

        # 상태 정의
        state: RAGState = {
            "query": user_input,
            "thinking": "",
            "documents": [],
            "answer": "",
            "messages": [],
            "mode": "",
        }

        # 스트리밍 실행
        response_stream = chain.stream(state)

        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for output in response_stream:
                if "answer" in output:
                    ai_answer = output["answer"]
                    container.markdown(ai_answer)

        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        st.warning("먼저 파일을 업로드해주세요.")
