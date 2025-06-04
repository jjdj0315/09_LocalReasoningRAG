# utils/handler.py

import streamlit as st
import json
from langchain_core.messages import ChatMessage
from langchain_core.documents import Document
from utils.dataclass import ChatMessageWithType


def format_search_result(results):
    """
    RAG 검색 결과를 마크다운 문자열로 포맷팅합니다.
    """
    if isinstance(results, list) and all(isinstance(r, Document) for r in results):
        answer = ""
        for i, doc in enumerate(results):
            # 출처에 파일명만 표시되도록 수정
            source_name = doc.metadata.get("source", "출처 없음").split("/")[
                -1
            ]  # 파일 경로에서 파일명만 추출
            answer += f"**[{i+1}] {source_name}**\n"
            answer += (
                f"```\n{doc.page_content[:300]}...\n```\n\n"  # 코드 블록으로 내용 표시
            )
        return answer
    else:
        return str(results)


# stream_handler의 인자를 변경하여 각 컨테이너를 명시적으로 받음
def stream_handler(
    node_status_placeholder,
    retrieved_docs_expander_placeholder,  # 검색 결과 expander를 위한 placeholder
    thinking_placeholder,
    answer_placeholder,
    langgraph_app_stream_generator,
):
    """
    LangGraph 앱의 스트리밍 결과를 처리하고 Streamlit UI에 실시간으로 표시합니다.
    각 노드의 실행 상태를 표시하고, 검색된 문서와 최종 답변을 스트리밍합니다.

    Args:
        node_status_placeholder (st.empty): 노드 상태 메시지를 표시할 컨테이너.
        retrieved_docs_expander_placeholder (st.empty): 검색 문서 expander를 표시할 컨테이너.
        thinking_placeholder (st.empty): 추론 과정을 스트리밍할 컨테이너.
        answer_placeholder (st.empty): 최종 답변을 스트리밍할 컨테이너.
        langgraph_app_stream_generator (iterator): app.stream()이 반환하는 제너레이터.

    Returns:
        tuple: (list_of_tool_results, final_answer_string, final_thinking_string)
    """
    current_agent_answer_buffer = ""
    current_thinking_buffer = ""
    retrieved_documents = []

    # 노드별로 상태 메시지를 매핑
    node_messages = {
        "classify": "🔍 질문 분류 중...",
        "retrieve": "📚 문서 검색 중...",
        "reasoning": "🤔 추론 과정 생성 중...",
        "generate": "💬 답변 생성 중...",
    }

    for chunk in langgraph_app_stream_generator:
        # chunk는 { "node_name": state_delta } 형태
        for node_name, node_output in chunk.items():
            # 노드 상태 메시지 업데이트
            if node_name in node_messages:
                node_status_placeholder.info(node_messages[node_name])

            if node_name == "retrieve":
                if "documents" in node_output and node_output["documents"]:
                    retrieved_documents = node_output["documents"]
                    with (
                        retrieved_docs_expander_placeholder
                    ):  # 전달받은 placeholder 사용
                        with st.expander("✅ 검색된 문서"):
                            st.markdown(format_search_result(retrieved_documents))
                    node_status_placeholder.empty()  # 검색 완료 후 상태 메시지 제거

            elif node_name == "reasoning":
                if "thinking" in node_output:
                    current_thinking_buffer = node_output["thinking"]
                    with thinking_placeholder:  # 추론 과정 전용 플레이스홀더에 스트리밍
                        st.markdown(f"**🧠 추론 과정:**\n{current_thinking_buffer}")

            elif node_name == "generate":
                if "answer" in node_output:
                    current_agent_answer_buffer = node_output["answer"]
                    answer_placeholder.markdown(
                        current_agent_answer_buffer
                    )  # 최종 답변 플레이스홀더에 스트리밍

            # 모든 노드 완료 시 상태 메시지 제거 (generate 노드 이후에 발생)
            if node_name == "__end__":
                node_status_placeholder.empty()
                thinking_placeholder.empty()  # 추론 과정 플레이스홀더도 제거
                # answer_placeholder는 답변이 최종 출력되므로 비우지 않음.

    # 스트리밍이 모두 끝난 후, 최종 결과들을 반환합니다.
    return retrieved_documents, current_agent_answer_buffer, current_thinking_buffer
