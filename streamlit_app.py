"""
Streamlit Main App - Enrollment FAQ Chatbot
Provides navigation to Student Chat and Advisor Dashboard
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Enrollment FAQ Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #667eea;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 Enrollment FAQ Chatbot</h1>
    <p>Complete prototype with Human-in-the-Loop functionality</p>
</div>
""", unsafe_allow_html=True)

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Interface:",
    ["Home", "Student Chat 💬", "Advisor Dashboard 👨‍💼"]
)

if page == "Home":
    st.header("Welcome to the Enrollment Chatbot Demo")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>💬 Student Chat</h3>
            <p>Experience the chatbot from a student's perspective. Ask questions about programs, tuition, and admissions.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>👨‍💼 Advisor Dashboard</h3>
            <p>View escalated conversations, manage the queue, and see full context for each student inquiry.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Statistics</h3>
            <p>Real-time monitoring of bot performance, queue length, and resolution rates.</p>
        </div>
        """, unsafe_allow_html=True)

    st.header("✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Student Experience
        - 🤖 **Automated FAQ** - Instant answers to common questions
        - 🎯 **Smart Escalation** - Confidence-based routing to humans
        - ⚡ **24/7 Availability** - Students get help anytime
        - 📱 **Mobile-Friendly** - Works on any device
        """)

    with col2:
        st.markdown("""
        ### Advisor Experience
        - 💾 **Context Preservation** - Full conversation history
        - 📋 **Queue Management** - Real-time escalation tracking
        - 🎯 **Smart Routing** - See escalation reasons
        - 📊 **Performance Metrics** - Bot resolution rates
        """)

    st.header("💰 Business Impact")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Annual Savings", "$237,500", "70% automation")

    with col2:
        st.metric("ROI", "7.9x", "First year")

    with col3:
        st.metric("Bot Resolution Rate", "70%", "Target")

    st.info("👈 Use the sidebar to navigate to Student Chat or Advisor Dashboard")

    st.header("🚀 Quick Start")

    with st.expander("Setup Instructions"):
        st.markdown("""
        ### Running the Backend API

        The Streamlit app connects to the FastAPI backend. Make sure it's running:

        ```bash
        cd enrollment_chatbot
        python main.py
        ```

        The API should be running on: `http://localhost:8000`

        ### For Streamlit Deployment

        If deploying to Streamlit Cloud, you'll need to:
        1. Deploy the FastAPI backend separately (e.g., on Heroku, AWS, or Railway)
        2. Update the `API_BASE_URL` in the Streamlit pages

        ### Environment Variables

        You can set the API URL via environment variable:
        ```bash
        export API_BASE_URL=https://your-api-url.com
        ```
        """)

elif page == "Student Chat 💬":
    # Import and run student chat
    import pages.student_chat

elif page == "Advisor Dashboard 👨‍💼":
    # Import and run advisor dashboard
    import pages.advisor_dashboard
