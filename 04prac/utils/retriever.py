from langchain_docling import DoclingLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_docling.loader import ExportType


def doc_load(FILE_PATH):
    loader = DoclingLoader(file_path=FILE_PATH, export_type=ExportType.MARKDOWN)
    docs = loader.load()
    return docs


def doc_split(docs):
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ],
    )
    splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]
    return splits
