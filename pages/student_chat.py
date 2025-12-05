"""
Streamlit Student Chat Interface
Students can ask questions and get instant answers with intelligent escalation
"""

import streamlit as st
import requests
import os
from datetime import datetime

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Student Chat - Enrollment FAQ",
    page_icon="💬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        text-align: right;
    }
    .bot-message {
        background: #f0f0f0;
        color: #333;
    }
    .escalation-notice {
        background: #ff6b6b;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .confidence-high {
        background: #d4edda;
        color: #155724;
    }
    .confidence-medium {
        background: #fff3cd;
        color: #856404;
    }
    .confidence-low {
        background: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.student_id = f"streamlit_student_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    # Add welcome message
    st.session_state.messages.append({
        "role": "bot",
        "content": """**Hello! 👋**

I'm your enrollment assistant. I can help answer questions about:
• Competency-based education model
• Programs and tuition
• How to apply and start
• Transfer credits and financial aid

What would you like to know?""",
        "confidence": None,
        "escalated": False
    })

# Header
st.title("💬 Student Chat")
st.markdown("Ask me anything about enrollment, programs, or tuition!")

# Sidebar with quick questions
st.sidebar.title("Quick Questions")
st.sidebar.markdown("Click to ask:")

quick_questions = [
    "What is competency-based education?",
    "How much does tuition cost?",
    "What programs do you offer?",
    "How do I apply?",
    "Will my credits transfer?",
    "Do you accept GI Bill?",
    "When can I start?",
    "Can I speak to a person?"
]

for question in quick_questions:
    if st.sidebar.button(question, key=f"quick_{question}"):
        st.session_state.pending_message = question

# Chat container
chat_container = st.container()

with chat_container:
    # Display all messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                👤 <strong>You:</strong><br>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        elif msg["role"] == "bot":
            confidence_badge = ""
            if msg.get("confidence") is not None:
                conf = msg["confidence"]
                if conf >= 0.9:
                    badge_class = "confidence-high"
                elif conf >= 0.7:
                    badge_class = "confidence-medium"
                else:
                    badge_class = "confidence-low"
                confidence_badge = f'<span class="confidence-badge {badge_class}">{int(conf * 100)}% confident</span>'

            st.markdown(f"""
            <div class="chat-message bot-message">
                🤖 <strong>Bot:</strong>{confidence_badge}<br>
                {msg["content"].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

            if msg.get("escalated"):
                st.markdown(f"""
                <div class="escalation-notice">
                    ⚠️ <strong>Escalation Triggered:</strong> {msg.get("escalation_reason", "Connecting you to an advisor...")}
                </div>
                """, unsafe_allow_html=True)

            # Show follow-up questions
            if msg.get("follow_up_questions"):
                st.markdown("**You might also want to know:**")
                cols = st.columns(min(len(msg["follow_up_questions"]), 3))
                for idx, fq in enumerate(msg["follow_up_questions"][:3]):
                    with cols[idx]:
                        if st.button(fq, key=f"followup_{len(st.session_state.messages)}_{idx}"):
                            st.session_state.pending_message = fq
                            st.rerun()

# Input area
st.markdown("---")
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Your question:",
        key="user_input",
        placeholder="Type your question here...",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("Send", type="primary", use_container_width=True)

# Handle pending message from quick questions
if "pending_message" in st.session_state:
    user_input = st.session_state.pending_message
    send_button = True
    del st.session_state.pending_message

# Process message
if send_button and user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "confidence": None,
        "escalated": False
    })

    # Call API
    try:
        with st.spinner("🤖 Thinking..."):
            response = requests.post(
                f"{API_BASE_URL}/api/v1/chat",
                json={
                    "student_id": st.session_state.student_id,
                    "message": user_input
                },
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()

                # Add bot response
                st.session_state.messages.append({
                    "role": "bot",
                    "content": data.get("message", ""),
                    "confidence": data.get("confidence"),
                    "escalated": data.get("escalation_suggested", False),
                    "escalation_reason": data.get("escalation_reason"),
                    "follow_up_questions": data.get("follow_up_questions", [])
                })
            else:
                st.error(f"Error: Unable to connect to chatbot (Status {response.status_code})")

    except requests.exceptions.ConnectionError:
        st.error("""
        ❌ **Cannot connect to the chatbot API**

        Make sure the FastAPI backend is running:
        ```bash
        cd enrollment_chatbot
        python main.py
        ```

        The API should be accessible at: `http://localhost:8000`
        """)
    except Exception as e:
        st.error(f"Error: {str(e)}")

    # Rerun to display new messages
    st.rerun()

# Sidebar stats
st.sidebar.markdown("---")
st.sidebar.subheader("Session Info")
st.sidebar.info(f"Student ID: {st.session_state.student_id}")
st.sidebar.info(f"Messages: {len([m for m in st.session_state.messages if m['role'] == 'user'])}")

# Check API status
try:
    health_response = requests.get(f"{API_BASE_URL}/health", timeout=2)
    if health_response.status_code == 200:
        st.sidebar.success("✅ Chatbot API Online")
    else:
        st.sidebar.error("❌ Chatbot API Error")
except:
    st.sidebar.error("❌ Chatbot API Offline")

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.session_state.student_id = f"streamlit_student_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    st.rerun()
