import streamlit as st

from utils.session import session_control
from utils.create_dir import create_dir
from utils.upload import upload_file
from utils.create_rag_chain import create_rag_chain
from utils.print_messages import print_messages
from utils.add_message import add_message

session_control()
create_dir()

st.set_page_config(page_title="100% 오픈모델 RAG , made by DJ")
st.title("100% 오픈모델 RAG, by DJ")

with st.sidebar:
    file = st.file_uploader(
        "파일 업로드",
        type=["pdf"],
    )
    if file:
        file_path = upload_file(file)
        rag_chain = create_rag_chain(file_path)
        st.session_state["chain"] = rag_chain

#메시지 출력
print_messages()

user_input = st.chat_input()
if user_input:
    if "chain" in st.session_state and st.session_state["chain"] is not None:
        chain = st.session_state["chain"]

        #사용자 입력
        st.chat_message("user").write(user_input)
        
        #스트리밍 호출
        responese = chain.stream(user_input)
        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for token in responese:
                ai_answer += token
                container.markdown(ai_answer)
                
        #대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        st.write("파일을 업로드해주세요.")        