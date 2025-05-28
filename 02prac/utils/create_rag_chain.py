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
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import MarkdownHeaderTextSplitter
from utils.format_docs import format_docs
from utils.format_docs import split_documents
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import RetrievalMode
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import START, StateGraph, END

@st.cache_resource(show_spinner="파일을 처리중입니다. 잠시만 기다려주세요.")
def create_rag_chain(file_path):
    
    reasoning_llm = ChatOllama(
    model="deepseek-r1:7b",
    stop=["</think>"]
    )

    answer_llm = ChatOllama(
    model="exaone3.5",
    temperature=0,
    )
    
    
    # spliter설정
    text_spliter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ],
    )
    splits = split_documents(docs,text_spliter)    
    # text_spliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # 문서 로드
    loader = DoclingLoader(
        file_path=file_path,
        export_type=ExportType.MARKDOWN
        )

    docs = loader.load()

    # loader = PDFPlumberLoader(file_path)
    # docs = loader.load_and_split(text_splitter=text_spliter)

    # 캐싱지원 임베딩 설정
    cache_dir = LocalFileStore(f".cache/embeddings")

    EMBEDDING_MODEL = "bge-m3:latest",
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir, namespace=EMBEDDING_MODEL
    )

    # 벡터디비저장
    vectorstore = QdrantVectorStore.from_documents(
                documents=splits,
                embedding=embeddings,
                location=":memory:",
                collection_name="rag_collection_0228",
                retrieval_mode=RetrievalMode.DENSE
            )

    # 검색기 설정
    retriever = vectorstore.as_retriever(search_kwargs = {'k':10})

    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

    compressor = CrossEncoderReranker(model=model, top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    
    
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
