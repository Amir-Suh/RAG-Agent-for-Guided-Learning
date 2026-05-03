from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class TutorState(TypedDict):
    messages: Annotated[list, add_messages]
    course_id: str
    current_context: str
    current_question: str
    student_answer: str
    is_correct: bool
    feedback: str
    mode: str  # NEW: Tracks if we are routing to QUIZ, LECTURE, or SKIP