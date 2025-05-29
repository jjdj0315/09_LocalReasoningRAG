from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from .model import embeddings

FILE_PATH = "https://arxiv.org/pdf/2408.09869"

loader = DoclingLoader(
    file_path=FILE_PATH,
    export_type=ExportType.MARKDOWN
    )

docs = loader.load()

from langchain_text_splitters import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ],
)
splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]

for d in splits[:3]:
    print(f"- {d.page_content=}")
print("...")


from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import RetrievalMode

vector_store = QdrantVectorStore.from_documents(
    documents=splits,
    embedding=embeddings,
    location=":memory:",
    collection_name="rag_collection_0228",
    retrieval_mode=RetrievalMode.DENSE
)

retriever = vector_store.as_retriever(search_kwargs = {'k':10})