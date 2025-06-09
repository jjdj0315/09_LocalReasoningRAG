from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader

from .model import embeddings
from .retriever import doc_split

def loader_load(selected_loader,docs):
    if selected_loader == "docling":
        loader = QdrantVectorStore.from_documents(
        documents=doc_split(docs),
        embedding=embeddings,
        location=":memory:",
        # location=".cache/qdrant",
        # client = client,
        collection_name="rag_collection_0228",
        retrieval_mode=RetrievalMode.DENSE,
    )
    else:
        selected_loader == "PDFPlumber":
        loader = QdrantVectorStore.from_documents(
        loader = PDFPlumberLoader(docs)
        documents=doc_split(docs),
        embedding=embeddings,
        location=":memory:",
        # location=".cache/qdrant",
        # client = client,
        collection_name="rag_collection_0228",
        retrieval_mode=RetrievalMode.DENSE,
    )
##
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

##
    return loader

QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        location=":memory:",
        # location=".cache/qdrant",
        # client = client,
        collection_name="rag_collection_0228",
        retrieval_mode=RetrievalMode.DENSE,
    )


class parser:
    def __init__(self, parser):
        self.parser = parser

    def aa