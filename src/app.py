import streamlit as st
import uuid
import sys
import os

# Ensure Python can find the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.graph import build_tutor_graph
from src.data_pipeline.ingestion import ingest_curriculum_to_pinecone

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Socratic AI Tutor", layout="wide")
st.title("Interactive Socratic Tutor")

# --- INITIALIZE SESSION STATE ---
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

if "graph" not in st.session_state:
    st.session_state.graph = build_tutor_graph()

# --- SIDEBAR: CONVERSATION LIST ---
with st.sidebar:
    st.header("Your Classes")
    
    if st.button("New Class Study Session", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {
            "title": f"New Session {len(st.session_state.chats) + 1}",
            "messages": [],
            "course_id": "default-class", # Each chat gets its own namespace
            "ingested": False
        }
        st.session_state.active_chat_id = new_id
        st.rerun()

    st.divider()

    for chat_id, chat_data in st.session_state.chats.items():
        is_active = (chat_id == st.session_state.active_chat_id)
        # Use the course name as the button label
        label = f"{chat_data['course_id']} ({chat_data['title']})"
        if st.button(label, key=f"btn_{chat_id}", 
                     use_container_width=True, 
                     type="primary" if is_active else "secondary"):
            st.session_state.active_chat_id = chat_id
            st.rerun()

# --- MAIN INTERFACE ---
if st.session_state.active_chat_id:
    active_id = st.session_state.active_chat_id
    current_chat = st.session_state.chats[active_id]
    config = {"configurable": {"thread_id": active_id}}

    # --- SESSION CONFIGURATION AREA ---
    # This section allows you to set the class name and upload files for THIS chat
    with st.expander("Class Configuration", expanded=not current_chat["ingested"]):
        col1, col2 = st.columns([2, 1])
        with col1:
            current_chat["course_id"] = st.text_input(
                "Enter Class Name (e.g., machine-learning-101)", 
                value=current_chat["course_id"],
                key=f"input_{active_id}"
            )
        
        uploaded_files = st.file_uploader(
            f"Upload slides for {current_chat['course_id']}", 
            accept_multiple_files=True,
            key=f"upload_{active_id}"
        )
        
        if st.button("Ingest and Start Studying", key=f"ingest_{active_id}"):
            if uploaded_files:
                temp_dir = "temp_data"
                os.makedirs(temp_dir, exist_ok=True)
                for f in uploaded_files:
                    with open(os.path.join(temp_dir, f.name), "wb") as buffer:
                        buffer.write(f.read())
                
                with st.spinner(f"Teaching the tutor about {current_chat['course_id']}..."):
                    # Pass the chat-specific course_id to the ingestion script
                    ingest_curriculum_to_pinecone(temp_dir, current_chat["course_id"])
                    current_chat["ingested"] = True
                    st.success("Ingestion complete. You can now begin.")
                    st.rerun()
            else:
                st.warning("Please upload at least one PDF for this class.")

    # --- CHAT INTERFACE ---
    if current_chat["ingested"]:
        # Initial greeting if session just started
        if not current_chat["messages"]:
            initial_state = {"course_id": current_chat["course_id"], "messages": []}
            with st.spinner("Initializing class materials..."):
                for event in st.session_state.graph.stream(initial_state, config):
                    if "Ask_Topic" in event:
                        greeting = event["Ask_Topic"]["current_question"]
                        current_chat["messages"].append({"role": "assistant", "content": greeting})

        # Render chat history
        for message in current_chat["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input processing
        if prompt := st.chat_input("Ask a question about this class..."):
            current_chat["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Ensure the graph uses the course_id bound to this chat
            st.session_state.graph.update_state(
                config, 
                {"student_answer": prompt, "course_id": current_chat["course_id"]}
            )

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                with st.spinner("Searching class curriculum..."):
                    events = st.session_state.graph.stream(None, config)
                    for event in events:
                        for node_name, node_data in event.items():
                            if node_name != "Router" and "current_question" in node_data:
                                full_response = node_data["current_question"]
                
                message_placeholder.markdown(full_response)
                current_chat["messages"].append({"role": "assistant", "content": full_response})
else:
    st.info("Start a new class study session in the sidebar to begin.")