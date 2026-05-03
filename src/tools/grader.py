import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.prompts import PromptTemplate

# Load environment variables safely
load_dotenv()

# 1. DEFINE THE OUTPUT SCHEMA
# This forces the LLM to return exactly these fields, no more, no less.
class GraderResult(BaseModel):
    is_correct: bool = Field(description="True if the student's answer is fundamentally correct, even if phrased differently.")
    score: int = Field(description="Score from 0 to 100 based on accuracy and completeness.")
    feedback: str = Field(description="Encouraging feedback explaining what was right, what was missing, and a gentle correction if needed.")

def evaluate_student(question: str, student_answer: str, ground_truth: str) -> GraderResult:
    """
    Acts as a strict but encouraging professor. 
    Compares the student's input against the Pinecone curriculum facts.
    """
    
    # 2. INITIALIZE THE EVALUATOR LLM
    # We use a low temperature (0.1) so the grader is deterministic and strict, not creative.
    llm = GoogleGenAI(model="models/gemini-2.5-flash", temperature=0.5)

    # 3. CONSTRUCT THE EVALUATION PROMPT
    prompt_str = """
    You are an expert probability and statistics professor grading a student.
    
    Question Asked: {question}
    Ground Truth (From Slides): {ground_truth}
    Student's Answer: {student_answer}
    
    Evaluate the student's answer based ONLY on the ground truth provided. 
    Do not penalize them for grammar, focus entirely on the mathematical concepts.
    Given an example response, not straight off from the slides, that covers all parts of the question. 
    """
    
    prompt = PromptTemplate(prompt_str)
    
    # 4. EXECUTE STRUCTURED PREDICTION
    # This guarantees the output will match our GraderResult class
    result = llm.structured_predict(
        GraderResult, 
        prompt, 
        question=question, 
        ground_truth=ground_truth, 
        student_answer=student_answer
    )
    
    return result

# --- TERMINAL TESTING LOOP ---
if __name__ == "__main__":
    print("--- Initializing Socratic Grader ---")
    
    # Mock data that would normally come from your Retriever
    test_question = "What is the difference between a marginal and joint probability density function?"
    test_ground_truth = "A joint PDF describes the probability distribution of two or more continuous random variables simultaneously. A marginal PDF describes the probability distribution of a single variable contained within a larger joint distribution, obtained by integrating the joint PDF over the other variables."
    
    print(f"\nQuestion: {test_question}")
    student_input = input("\nYou are the student. Answer the question: ")
    
    print("\nGrading response...")
    grade = evaluate_student(
        question=test_question,
        student_answer=student_input,
        ground_truth=test_ground_truth
    )
    
    print("\n--- Grading Results ---")
    print(f"Score:     {grade.score}/100")
    print(f"Feedback:  {grade.feedback}")