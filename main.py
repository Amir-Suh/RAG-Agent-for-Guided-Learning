import sys
import os
import uuid

# Ensure Python can find the src folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.agent.graph import build_tutor_graph

def run_tutor():
    print("Initializing User-Directed Tutor Agent...")
    graph = build_tutor_graph()
    
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "course_id": "probability-and-stats-101",
        "messages": []
    }

    # Start the engine (Runs Ask_Topic -> Pauses at Router)
    for event in graph.stream(initial_state, config):
        if "Ask_Topic" in event:
            print(f"\n[Agent]: {event['Ask_Topic']['current_question']}")

    # The Endless Loop
    while True:
        user_input = input("\n[You]: ")
        
        if user_input.lower() in ["quit", "exit"]:
            print("\nClass dismissed. Good luck studying!")
            break

        # Inject user input into the paused Router
        graph.update_state(config, {"student_answer": user_input})

        # Resume the engine. 
        for event in graph.stream(None, config):
            for node_name, node_data in event.items():
                # Print whatever the active node generated as the next question/response
                if node_name != "Router" and "current_question" in node_data:
                    print(f"\n[Agent]: {node_data['current_question']}")

if __name__ == "__main__":
    run_tutor()