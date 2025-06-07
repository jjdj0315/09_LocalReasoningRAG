from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .state import RAGState
from .model import reasoning_llm

classify_llm_chain = ChatPromptTemplate.from_template(
    """
다음 사용자 질문이 외부 문서의 정보가 필요한지 판단하세요.

- 특정 문서, 보고서, 사용자 업로드 파일, 수치, 통계, 날짜 등 **문서 기반 정보가 필요한 경우**: 'retrieve'
- 일반 상식, 대화, 감정, 일상적인 질문처럼 **문서 없이도 대답 가능한 경우**: 'generate'

오직 아래 두 단어 중 하나만 출력하세요: retrieve 또는 generate  
그 외 문장은 절대 출력하지 마세요.

사용자 질문: {query}

판단:
    """
    | reasoning_llm
    | StrOutputParser
)


# 1. 질문 분류 함수 - 중요: 여기서는 상태를 업데이트하는 노드 함수
def classify_node(state: RAGState):
    """질문을 분류하여 처리모드를 결저합니다."""
    query = state["query"]
    print("질문 분류 시작 : {query}")

    # 정의된 classify_llm_chain을 사용하야 질문 의도 분류
    classification = classify_llm_chain.invoke({"query": query})

    print(f"====질문 분류 완료 : {classification}====")

    # 분류 결과에 따라 mode상태를 설정합니다.
    if "retrieve" in classification.lower():
        mode = "retreive"
    else:
        mode = "generate"

    print("모드 결정 {mode}")
    return {"mode": mode}
