from typing import Any, Dict

from graph.state import GraphState
from graph.chains.retrieval_grader import retrieval_grader

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the user's question.
    If any document is not relevant, we will set a flag to run web search
    
    Arguments:
        state: The current graph state
    Returns:
        state: Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for doc in documents:

        res = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )

        grade = res.binary_score

        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT REVELANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
    }