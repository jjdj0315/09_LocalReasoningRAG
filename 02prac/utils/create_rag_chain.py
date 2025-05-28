import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.storage import LocalFileStore
from langchain_ollama import OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import load_prompt
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from utils.format_docs import format_docs


@st.cache_resource(show_spinner="파일을 처리중입니다. 잠시만 기다려주세요.")
def create_rag_chain(file_path):
    # spliter설정
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # 문서 로드
    loader = PDFPlumberLoader(file_path)
    docs = loader.load_and_split(text_splitter=text_spliter)

    # 캐싱지원 임베딩 설정
    cache_dir = LocalFileStore(f".cache/embeddings")

    EMBEDDING_MODEL = "bge-m3"
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir, namespace=EMBEDDING_MODEL
    )

    # 벡터디비저장
    vectorstore = FAISS.from_documents(docs, embedding=cached_embeddings)

    # 검색기 설정
    retriever = vectorstore.as_retriever()

    # 프롬프트 로드
    prompt = load_prompt("utils/prompts/rag-exaone.yaml", encoding="utf-8")

    llm = ChatOllama(model="mistral", temperature=0)

    # 체인 생성
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
