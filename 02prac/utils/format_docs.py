# 문서 포맷팅
# def format_docs(docs):
#     return "\n\n".join(
#         f"<document><content>{doc.page_content}</content><page>{doc.metadata['page']}</page><source>{doc.metadata['source']}</source></document>"
#         for doc in docs
#     )




def split_documents(docs, splitter):
    return [split for doc in docs for split in splitter.split_text(doc.page_content)]