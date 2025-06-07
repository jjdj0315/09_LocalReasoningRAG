from langgraph.graph import StateGraph
from utils.state import RAGState


workflow = StateGraph(RAGState)


def create_app():
    workflow = StateGraph(RAGState)

    # 노드추가
workflow.add_node("classify", classify_node)
    app = ddd
    return app
