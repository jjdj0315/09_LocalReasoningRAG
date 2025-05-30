from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import RetrievalMode
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utils.model import reasoning_llm, answer_llm
from langgraph.graph import START, StateGraph, END

from utils.state import RAGState
from .model import embeddings
from .retriever import doc_load
from .retriever import doc_split

def app(FILE_PATH):
    #로드
    docs = doc_load(FILE_PATH)
    print(docs)
    #스필릿릿
    splits = doc_split(docs)
    print(splits[1])
    #벡터디비
    vector_store = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        location=":memory:",
        collection_name="rag_collection_0228",
        retrieval_mode=RetrievalMode.DENSE
    )        
    #리트리버
    retriever = vector_store.as_retriever(search_kwargs = {'k':10})


    #리랭커
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

    compressor = CrossEncoderReranker(model=model, top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    ) 
    
    
    
    return app


