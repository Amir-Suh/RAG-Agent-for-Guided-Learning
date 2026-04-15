import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from llama_index.llms.google_genai import GoogleGenAI

from src.agent.state import TutorState
from src.tools.retriever import GroundedRetriever
from src.tools.grader import evaluate_student

load_dotenv()

# --- 1. ACTION NODES ---

def ask_topic_node(state: TutorState):
    """Starts the loop dynamically, acting like a real human."""
    llm = GoogleGenAI(model="models/gemini-2.5-flash")
    last_input = state.get("student_answer", "")
    
    prompt = f"""
    You are a friendly, conversational AI tutor. 
    The student just said: "{last_input}"
    
    If the student hasn't said anything yet, just greet them and ask what they want to study.
    If they just asked to move on, briefly acknowledge it (e.g., "Sounds good!"). 
    Then, ask them what concept, topic, or slide number they would like to tackle next. 
    Vary your phrasing so you don't sound repetitive. Keep it to one short sentence.
    """
    question = llm.complete(prompt).text.strip()
    
    return {"current_question": question, "current_context": ""} 

def lecture_node(state: TutorState):
    """Retrieves the requested concept, explains it, and offers practice naturally."""
    retriever = GroundedRetriever(state["course_id"])
    llm = GoogleGenAI(model="models/gemini-2.5-flash")
    
    topic_request = state["student_answer"]
    
    query = f"Find information regarding: {topic_request}. Provide a comprehensive summary."
    context_response = retriever.ask(query)
    current_context = str(context_response)

    prompt = f"""
    You are a helpful, conversational tutor. Based on this retrieved curriculum data: 
    {current_context}
    
    1. Give a clear, engaging explanation of the concept to the student.
    2. At the very end of your response, naturally ask if they would like to try a practice question to test their understanding. Vary your phrasing (e.g., "Ready to try a problem?", "Want to test this out?").
    
    Do NOT ask them a quiz question yet. Just explain and offer the practice.
    """
    lecture = llm.complete(prompt).text

    return {
        "current_context": current_context,
        "current_question": lecture
    }

def quiz_generation_node(state: TutorState):
    """Generates a practice question based on the current slide."""
    llm = GoogleGenAI(model="models/gemini-2.5-flash")
    
    prompt = f"""
    Based on this curriculum context: {state['current_context']}
    Generate a thoughtful, Socratic practice question to test the student's understanding.
    """
    question = llm.complete(prompt).text
    
    return {"current_question": question}

def grade_answer_node(state: TutorState):
    """Evaluates the student's answer and asks what to do next naturally."""
    llm = GoogleGenAI(model="models/gemini-2.5-flash")
    
    grade_result = evaluate_student(
        question=state["current_question"],
        student_answer=state["student_answer"],
        ground_truth=state["current_context"]
    )
    
    if grade_result.is_correct:
        prompt = f"""
        The student got the answer correct. Mathematical Feedback: {grade_result.feedback}
        Rewrite this feedback to be conversational and encouraging. 
        At the end, ask them naturally if they want another practice question on this, or if they are ready to move on to a new concept.
        """
    else:
        prompt = f"""
        The student got the answer wrong. Mathematical Feedback: {grade_result.feedback}
        Rewrite this feedback to be gentle and helpful. 
        At the end, ask them if they want to try another question on this topic, or if they prefer to move on.
        """
        
    final_feedback = llm.complete(prompt).text
        
    return {
        "is_correct": grade_result.is_correct,
        "current_question": final_feedback
    }

# --- 2. THE ROUTER NODE ---

def intent_router_node(state: TutorState):
    """Reads the user's input to determine the next phase of the workflow."""
    llm = GoogleGenAI(model="models/gemini-2.5-flash")
    
    last_prompt = state.get("current_question", "")
    user_input = state.get("student_answer", "")
    
    prompt = f"""
    The agent just asked the user: "{last_prompt}"
    The user replied: "{user_input}"
    
    Categorize the user's intent into exactly ONE of these words:
    - 'NEW_TOPIC': They explicitly stated a specific concept, slide, or topic they want to learn next (e.g., "Let's do joint CDFs", "Slide 4", "Explain marginal density"). Prioritize this if they name a subject.
    - 'START_QUIZ': They answered 'yes' to wanting a practice question, or asked for another one.
    - 'ANSWERING': They are attempting to answer a practice question you just asked.
    - 'MOVE_ON': They said 'no' to practice, or explicitly said they want to move on, BUT did NOT specify what topic to move to.
    
    Category:
    """
    intent = llm.complete(prompt).text.strip().upper()
    
    valid_intents = ["NEW_TOPIC", "START_QUIZ", "ANSWERING", "MOVE_ON"]
    if intent not in valid_intents:
        intent = "MOVE_ON" 

    return {"mode": intent}

# --- 3. BUILD GRAPH ---

def build_tutor_graph():
    workflow = StateGraph(TutorState)
    
    workflow.add_node("Ask_Topic", ask_topic_node)
    workflow.add_node("Lecture", lecture_node)
    workflow.add_node("Router", intent_router_node)
    workflow.add_node("Quizzer", quiz_generation_node)
    workflow.add_node("Grader", grade_answer_node)
    
    workflow.set_entry_point("Ask_Topic")
    
    workflow.add_edge("Ask_Topic", "Router")
    workflow.add_edge("Lecture", "Router")
    workflow.add_edge("Quizzer", "Router")
    workflow.add_edge("Grader", "Router")
    
    workflow.add_conditional_edges(
        "Router",
        lambda x: x["mode"],
        {
            "NEW_TOPIC": "Lecture",
            "START_QUIZ": "Quizzer",
            "ANSWERING": "Grader",
            "MOVE_ON": "Ask_Topic"
        }
    )
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_before=["Router"])