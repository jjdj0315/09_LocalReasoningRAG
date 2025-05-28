# 문서 포맷팅
def format_docs(docs):
    return "\n\n".join(
        f"<document><content>{doc.page_content}</content><page>{doc.metadata['page']}</page><source>{doc.metadata['source']}</source></document>"
        for doc in docs
    )
