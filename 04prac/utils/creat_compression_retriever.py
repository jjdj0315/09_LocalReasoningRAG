import streamlit as st
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import RetrievalMode
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from .retriever import doc_load
from .model import embeddings

# from .retriever import doc_split


# from .qdrant import client
@st.cache_resource(show_spinner=False)
def creat_compression_retriever(FILE_PATH, selected_loader):
    # 로드
    st.info("🔍 문서 로딩 중...")
    splits = doc_load(FILE_PATH, selected_loader)

    # 캐싱지원 임베딩 설정
    cache_dir = LocalFileStore(f".cache/embeddings")

    EMBEDDING_MODEL = "bge-m3"
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir, namespace=EMBEDDING_MODEL
    )

    # 벡터디비
    st.info("📦 벡터 저장소 구축 중...")
    vector_store = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=cached_embeddings,
        location=":memory:",
        # location=".cache/qdrant",
        # client = client,
        collection_name="rag_collection_0228",
        retrieval_mode=RetrievalMode.DENSE,
    )
    # 리트리버
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # 리랭커
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

    compressor = CrossEncoderReranker(model=model, top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    st.success("✅ 압축 리트리버 준비 완료!")
    return compression_retriever
