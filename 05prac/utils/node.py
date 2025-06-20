from langgraph.graph import StateGraph,START, END
from langgraph.checkpoint.memory import MemorySaver
from utils.state import RAGState
from .graph import classify_node,reasoning,retrieve,generate,route_by_mode

workflow = StateGraph(RAGState)


def create_app():
    workflow = StateGraph(RAGState)

    # 노드추가
    workflow.add_node("classify", classify_node)
    workflow.add_node("reasoning",reasoning )
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    
    #엣지 추가
    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify",
        route_by_mode,
        {"retrieve" : retrieve, "geneerate" : generate}
    )
    workflow.add_edge("retrieve", "reasoning")
    workflow.add_edge("resoning", "generate")

    workflow.add_edge("generate", END)
    memory = MemorySaver()

    app = workflow.compile(checkpointer=memory)
    return app
