from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import RetrievalMode
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


from .retriever import doc_load
from .model import embeddings

# from .retriever import doc_split


# from .qdrant import client
def creat_compression_retriever(FILE_PATH, selected_loader):
    # 로드
    splits = doc_load(FILE_PATH, selected_loader)

    # 벡터디비
    vector_store = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
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
    return compression_retriever
