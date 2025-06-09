# utils/handler.py
import streamlit as st
import json
import time  # time 모듈 추가
from langchain_core.documents import Document  # Document 타입 임포트


def format_search_result(results):
    """
    RAG 검색 결과를 마크다운 문자열로 포맷팅하거나,
    이전 Perplexity 앱의 웹 검색 결과(JSON 문자열)를 파싱하여 포맷팅합니다.
    """
    # RAG 문서 검색 결과 (List[Document]) 처리
    if isinstance(results, list) and all(isinstance(r, Document) for r in results):
        answer = ""
        for i, doc in enumerate(results):
            source_name = doc.metadata.get("source", "출처 없음")
            # 파일 경로일 경우 파일명만 추출
            if isinstance(source_name, str) and "/" in source_name:
                source_name = source_name.split("/")[-1]

            answer += f"**[{i+1}] {source_name}**\n"
            # 문서 내용을 최대 300자로 자르고 줄바꿈 추가
            answer += f"```\n{doc.page_content[:300]}...\n```\n\n"
        return answer
    else:
        # 이전 Perplexity 앱에서 사용하던 JSON 문자열 검색 결과 처리 (웹 검색)
        try:
            results_json = json.loads(results)
            answer = ""
            for result in results_json:
                answer += f'**[{result["title"]}]({result["url"]})**\n\n'
                answer += f'{result["content"]}\n\n'
                answer += f'신뢰도: {result["score"]}\n\n'
                answer += "\n-----\n"
            return answer
        except (json.JSONDecodeError, TypeError):
            # JSON이 아니거나 예상치 못한 형식일 경우 그대로 반환
            return str(results)


# get_current_tool_message 함수는 이 RAG 앱에서는 사용되지 않으므로 제거하거나 그대로 두어도 됩니다.
# (Perplexity 클론 앱에서 사용하던 함수)


def stream_handler(
    node_status_placeholder,
    retrieved_docs_expander_placeholder,
    thinking_placeholder,
    answer_placeholder,
    langgraph_app_stream_generator,
):
    """
    LangGraph 앱의 결과를 처리하고 Streamlit UI에 점진적으로 표시합니다.
    각 노드의 실행 상태를 표시하고, 검색된 문서와 최종 답변을 스트리밍합니다.
    (invoke()로 한 번에 받은 후 문자열을 한 글자씩 출력하여 스트리밍 효과를 냅니다.)

    Args:
        node_status_placeholder (st.empty): 노드 상태 메시지를 표시할 컨테이너.
        retrieved_docs_expander_placeholder (st.empty): 검색 문서 expander를 표시할 컨테이너.
        thinking_placeholder (st.empty): 추론 과정을 스트리밍할 컨테이너.
        answer_placeholder (st.empty): 최종 답변을 스트리밍할 컨테이너.
        langgraph_app_stream_generator (iterator): app.stream()이 반환하는 제너레이터.

    Returns:
        tuple: (list_of_retrieved_documents, final_answer_string, final_thinking_string)
    """
    current_agent_answer = ""
    current_thinking_content = ""
    retrieved_documents = []  # Document 객체 리스트를 저장할 변수

    for chunk in langgraph_app_stream_generator:
        # print(f"DEBUG: Handler received chunk: {chunk}") # 디버깅용
        for node_name, node_output in chunk.items():
            # 노드 상태 메시지 업데이트
            if node_name == "classify":
                node_status_placeholder.info("🔍 질문 분류 중...")
            elif node_name == "retrieve":
                node_status_placeholder.info("📚 문서 검색 중...")
            elif node_name == "reasoning":
                node_status_placeholder.info("🤔 추론 과정 생성 중...")
            elif node_name == "generate":
                node_status_placeholder.info("💬 답변 생성 중...")
            elif node_name == "__end__":
                node_status_placeholder.empty()  # 모든 노드 완료 후 상태 메시지 지우기

            # 각 노드별 처리
            if node_name == "retrieve":
                if "documents" in node_output and node_output["documents"]:
                    retrieved_documents = node_output["documents"]
                    with retrieved_docs_expander_placeholder.expander(
                        "📚 검색된 문서"
                    ):  # expander 안에 문서 내용 표시
                        st.markdown(format_search_result(retrieved_documents))
                    node_status_placeholder.success("✅ 문서 검색 완료")

            elif node_name == "reasoning":
                if "thinking" in node_output:
                    current_thinking_content = node_output["thinking"]
                    # 추론 과정을 한 글자씩 스트리밍 효과
                    full_thinking_text = ""
                    for char in current_thinking_content:
                        full_thinking_text += char
                        thinking_placeholder.markdown(
                            f"**🧠 추론 과정:**\n{full_thinking_text}"
                        )
                        time.sleep(0.02)  # 글자마다 지연 시간 (조절 가능)
                    node_status_placeholder.success("✅ 추론 과정 완료")

            elif node_name == "generate":
                if "answer" in node_output:
                    current_agent_answer = node_output["answer"]
                    # 최종 답변을 한 글자씩 스트리밍 효과
                    full_answer_text = ""
                    for char in current_agent_answer:
                        full_answer_text += char
                        answer_placeholder.markdown(full_answer_text)
                        time.sleep(0.02)  # 글자마다 지연 시간 (조절 가능)
                    node_status_placeholder.success("✅ 답변 생성 완료")

    return retrieved_documents, current_agent_answer, current_thinking_content
