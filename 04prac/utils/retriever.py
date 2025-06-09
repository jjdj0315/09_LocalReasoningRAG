from langchain_docling import DoclingLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_docling.loader import ExportType
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader


def doc_load(FILE_PATH, selected_loader):
    if selected_loader == "docling":
        loader = DoclingLoader(file_path=FILE_PATH, export_type=ExportType.MARKDOWN)
        docs = loader.load()
        print("******docling실행중******")
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ],
        )
        splits = [
            split for doc in docs for split in splitter.split_text(doc.page_content)
        ]
        return splits

    else:
        loader = PDFPlumberLoader(file_path=FILE_PATH)
        docs = loader.load()
        print("******PDFPlumberLoader실행중******")
        # spliter설정
        text_spliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = loader.load_and_split(text_splitter=text_spliter)
        return splits


# def doc_split(docs):
#     splitter = MarkdownHeaderTextSplitter(
#         headers_to_split_on=[
#             ("#", "Header_1"),
#             ("##", "Header_2"),
#             ("###", "Header_3"),
#         ],
#     )
#     splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]
#     return splits
