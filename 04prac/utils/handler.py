# utils/handler.py

import streamlit as st
import json
from langchain_core.messages import ChatMessage
from langchain_core.documents import Document  # Document 타입 임포트
from utils.dataclass import ChatMessageWithType  # dataclass.py에서 정의된 클래스


def format_search_result(results):
    """
    RAG 검색 결과를 마크다운 문자열로 포맷팅합니다.
    """
    if isinstance(results, list) and all(isinstance(r, Document) for r in results):
        answer = ""
        for i, doc in enumerate(results):
            answer += f'**[{i+1}] {doc.metadata.get("source", "출처 없음")}**\n'
            answer += (
                f"```\n{doc.page_content[:300]}...\n```\n\n"  # 코드 블록으로 내용 표시
            )
        return answer
    else:
        return str(results)  # 예상치 못한 타입의 경우 문자열로 변환


def stream_handler(streamlit_container, langgraph_app_stream_generator):
    """
    LangGraph 앱의 스트리밍 결과를 처리하고 Streamlit UI에 실시간으로 표시합니다.
    각 노드의 실행 상태를 표시하고, 검색된 문서와 최종 답변을 스트리밍합니다.

    Args:
        streamlit_container (streamlit.empty): 스트리밍 메시지를 표시할 Streamlit 컨테이너.
        langgraph_app_stream_generator (iterator): app.stream()이 반환하는 제너레이터.

    Returns:
        tuple: (list_of_tool_results, final_answer_string, final_thinking_string)
        tool_results는 LangGraph의 "documents" 노드에서 생성된 Document 객체 리스트
    """
    current_agent_answer_buffer = ""  # LLM 답변 버퍼 (스트리밍 용)
    retrieved_documents = []  # 검색된 문서 (최종 session_state 저장을 위함)
    thinking_process = ""  # 추론 과정 (최종 session_state 저장을 위함)

    # 스트리밍된 LLM 답변을 표시할 Streamlit.empty() 객체.
    # 이 객체는 stream_handler 외부에서 app.py가 생성하여 넘겨주므로, 이 함수 내에서 다시 생성할 필요 없음.
    # if not hasattr(streamlit_container, 'markdown'):
    #     st.error("스트림 핸들러에 유효한 Streamlit 컨테이너가 전달되지 않았습니다.")
    #     return [], "", "" # 오류 발생 시 빈 값 반환

    # 각 노드별 상태 메시지를 표시할 임시 컨테이너
    node_status_container = st.empty()

    # 스트리밍 루프
    for chunk in langgraph_app_stream_generator:
        # chunk는 { "node_name": state_delta } 형태
        for node_name, node_output in chunk.items():
            # print(f"Chunk from node '{node_name}': {node_output}") # 디버깅용

            # --- 노드 상태 메시지 업데이트 ---
            if node_name == "classify":
                node_status_container.info("🔍 질문 분류 중...")
            elif node_name == "retrieve":
                node_status_container.info("📚 문서 검색 중...")
                if "documents" in node_output and node_output["documents"]:
                    retrieved_documents = node_output["documents"]  # 문서 저장
                    # 문서 검색 결과는 별도의 expander에 표시
                    with streamlit_container:  # 주 스트리밍 컨테이너 내에 expander 생성
                        with st.expander("✅ 검색된 문서"):
                            st.markdown(format_search_result(retrieved_documents))
                    node_status_container.empty()  # 검색 완료 후 상태 메시지 제거
            elif node_name == "reasoning":
                node_status_container.info("🤔 추론 과정 생성 중...")
                if "thinking" in node_output and node_output["thinking"]:
                    thinking_process = node_output["thinking"]  # 추론 과정 저장
                    # 추론 과정도 expander에 표시
                    with streamlit_container:
                        with st.expander("🧠 추론 과정"):
                            st.markdown(thinking_process)
                    node_status_container.empty()  # 추론 완료 후 상태 메시지 제거
            elif node_name == "generate":
                node_status_container.info("💬 답변 생성 중...")
                # LLM 답변 스트리밍은 'answer' 필드에 누적된 내용이 들어옵니다.
                if "answer" in node_output:
                    current_agent_answer_buffer = node_output["answer"]
                    streamlit_container.markdown(
                        current_agent_answer_buffer
                    )  # 실시간 업데이트
            elif node_name == "__end__":
                node_status_container.empty()  # 모든 작업 완료 후 상태 메시지 제거

    # 스트리밍이 모두 끝난 후, 최종 결과들을 반환합니다.
    return retrieved_documents, current_agent_answer_buffer, thinking_process
