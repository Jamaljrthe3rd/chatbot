import streamlit as st
# Change the faiss import to use the standard faiss-cpu package
import faiss
import numpy as np
import os
import pandas as pd
from mistralai import Mistral, UserMessage
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Use the provided API key or fall back to environment variable
api_key = "fPOyqAqecQ9f4ekvTizLD3v4JbE9fG30"  # Using the provided API key

# Set page configuration
st.set_page_config(
    page_title="UDST Policy Chatbot",
    page_icon="📚",
    layout="wide"
)

# Load Sentence Transformer for Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define policies directly in the code instead of loading from CSV
policies = {
    "Academic Integrity Policy": "Students must maintain academic integrity in all coursework and research. Plagiarism, cheating, and falsification of data are strictly prohibited.",
    "Attendance Policy": "Students are expected to attend all scheduled classes. Absences exceeding 15% of total class hours may result in course failure.",
    "Grading Policy": "Grades are assigned on a scale of A to F. A minimum GPA of 2.0 is required to maintain good academic standing.",
    "Research Ethics Policy": "All research involving human subjects must be approved by the Ethics Committee before commencement.",
    "Student Conduct Policy": "Students must behave respectfully towards faculty, staff, and peers. Harassment and discrimination are not tolerated.",
    "Technology Use Policy": "University computing resources are for academic and administrative purposes only. Unauthorized access is prohibited.",
    "Tuition Payment Policy": "Tuition fees must be paid by the specified deadline each semester. Late payments incur additional charges.",
    "Library Resources Policy": "Library materials must be returned by the due date. Overdue items result in fines and possible loss of borrowing privileges.",
    "Campus Safety Policy": "All campus safety incidents must be reported immediately to security personnel.",
    "Accommodation Policy": "Students with disabilities may request reasonable accommodations through the Student Support Services office."
}

# Initialize FAISS Index
d = 384  # Embedding size
database = faiss.IndexFlatL2(d)

# Embed Policies and Store in FAISS
policy_keys = list(policies.keys())
policy_embeddings = model.encode(list(policies.values()))
database.add(np.array(policy_embeddings, dtype=np.float32))

# Initialize Mistral API
mistral_client = Mistral(api_key=api_key)

def retrieve_policy(query):
    query_embedding = model.encode([query])
    D, I = database.search(np.array(query_embedding, dtype=np.float32), k=1)
    return policy_keys[I[0][0]], policies[policy_keys[I[0][0]]]

def generate_response(query):
    policy_title, policy_content = retrieve_policy(query)
    messages = [
        UserMessage(content=f"Context: {policy_content}\nUser Query: {query}\nAnswer: ")
    ]
    response = mistral_client.chat.complete(model="mistral-large-latest", messages=messages)
    return policy_title, response.choices[0].message.content

# Streamlit UI
st.title("UDST Policy Chatbot 🤖")
st.markdown("Ask questions about UDST policies and get accurate answers!")

# Create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.sidebar.header("UDST Policies")
    selected_policy = st.sidebar.selectbox("Select a policy to view", list(policies.keys()))
    
    # Display all available policies
    st.sidebar.subheader("Available Policies")
    for i, policy in enumerate(policies.keys(), 1):
        st.sidebar.markdown(f"{i}. {policy}")

with col2:
    # Display selected policy
    st.subheader(f"Selected Policy: {selected_policy}")
    st.markdown(policies[selected_policy])
    
    # Chat interface
    st.markdown("---")
    st.subheader("Ask a Question")
    query = st.text_input("Enter your question about UDST policies:")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    if st.button("Get Answer"):
        if query:
            with st.spinner("Generating answer..."):
                title, answer = generate_response(query)
                # Add to chat history
                st.session_state.chat_history.append({"query": query, "title": title, "answer": answer})
        else:
            st.warning("Please enter a question!")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("Chat History")
        for i, chat in enumerate(st.session_state.chat_history):
            st.markdown(f"**Question {i+1}:** {chat['query']}")
            st.markdown(f"**Policy Referenced:** {chat['title']}")
            st.markdown(f"**Answer:** {chat['answer']}")
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("UDST Policy Chatbot - Powered by Mistral AI and Streamlit")