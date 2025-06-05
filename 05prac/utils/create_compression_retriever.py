import streamlit as st
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import RetrievalMode
from .retriever import doc_load
from .model import embeddings


@st.cache_resource(show_spinner=False)
def create_compression_retriever(FILE_PATH, selected_loader):
    # 로드
    st.info("문서 로딩중..")
    splits = doc_load(FILE_PATH, selected_loader)

    # 캐싱지원 임베딩 설정
    cache_dir = LocalFileStore(f".cache/embeddings")

    EMBEDDING_MODEL = "bge-m3"
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir, namespace=EMBEDDING_MODEL
    )

    # 벡터디비
    st.info("벡터저장소 구축 중")

    vector_store = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=cached_embeddings,
        location=":memory",
        collection_name="rag_collection_0228",
        retriever_mode=RetrievalMode.DENSE,
    )
    return compression_retriever
