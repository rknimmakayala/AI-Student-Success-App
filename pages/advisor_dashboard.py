"""
Streamlit Advisor Dashboard
View escalated conversations and manage the queue
"""

import streamlit as st
import requests
import os
from datetime import datetime
import time

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Advisor Dashboard",
    page_icon="👨‍💼",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .queue-item {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
    }
    .conversation-message {
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .user-msg {
        background: #e3f2fd;
        border-left: 3px solid #2196f3;
    }
    .bot-msg {
        background: #f1f8e9;
        border-left: 3px solid #8bc34a;
    }
    .system-msg {
        background: #fff3cd;
        border-left: 3px solid #ffc107;
    }
    .escalation-badge {
        background: #ff6b6b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("👨‍💼 Advisor Dashboard")
st.markdown("Manage escalated conversations and provide personalized support")

# Auto-refresh toggle
col1, col2 = st.columns([4, 1])
with col2:
    auto_refresh = st.checkbox("Auto-refresh (10s)", value=False)

if auto_refresh:
    time.sleep(10)
    st.rerun()

# Fetch statistics
try:
    stats_response = requests.get(f"{API_BASE_URL}/api/v1/stats", timeout=5)
    if stats_response.status_code == 200:
        stats = stats_response.json()

        # Display statistics
        st.subheader("📊 Real-Time Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "⏳ Queue Length",
                stats.get("current_queue_length", 0),
                help="Conversations waiting for advisor"
            )

        with col2:
            st.metric(
                "💬 Total Conversations",
                stats.get("total_conversations", 0),
                help="All conversations today"
            )

        with col3:
            st.metric(
                "✅ Bot Handled",
                stats.get("bot_handled", 0),
                help="Successfully resolved by bot"
            )

        with col4:
            st.metric(
                "📊 Bot Resolution Rate",
                f"{stats.get('bot_resolution_rate', 0):.0f}%",
                help="Percentage of conversations handled by bot"
            )

    else:
        st.error(f"Error fetching statistics (Status {stats_response.status_code})")

except requests.exceptions.ConnectionError:
    st.error("""
    ❌ **Cannot connect to the chatbot API**

    Make sure the FastAPI backend is running:
    ```bash
    cd enrollment_chatbot
    python main.py
    ```
    """)
    st.stop()
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.stop()

st.markdown("---")

# Fetch queue
try:
    queue_response = requests.get(f"{API_BASE_URL}/api/v1/advisor/queue", timeout=5)

    if queue_response.status_code == 200:
        queue_data = queue_response.json()
        queue = queue_data.get("queue", [])

        # Display queue
        st.subheader(f"🔔 Escalated Conversations Queue ({len(queue)})")

        if len(queue) == 0:
            st.success("✅ **All caught up!** No conversations waiting for assistance.")
        else:
            # Create tabs for each conversation
            for idx, item in enumerate(queue):
                with st.expander(
                    f"👤 Student: {item['student_id']} | ⏱️ {item['wait_time_minutes']} min wait | 📋 {item['messages_count']} messages",
                    expanded=(idx == 0)  # Expand first item
                ):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"""
                        **Channel:** {item['channel'].upper()}
                        **Escalated:** {item['escalated_at']}
                        **Reason:** <span class="escalation-badge">{item['escalation_reason']}</span>
                        """, unsafe_allow_html=True)

                    with col2:
                        if st.button("📋 View Full Conversation", key=f"view_{item['conversation_id']}"):
                            st.session_state.viewing_conversation = item['conversation_id']

                    # If this conversation is being viewed, show full details
                    if st.session_state.get("viewing_conversation") == item['conversation_id']:
                        st.markdown("---")
                        st.subheader("📜 Full Conversation Context")

                        # Fetch full conversation
                        try:
                            conv_response = requests.get(
                                f"{API_BASE_URL}/api/v1/advisor/conversation/{item['conversation_id']}",
                                timeout=5
                            )

                            if conv_response.status_code == 200:
                                conv_data = conv_response.json()

                                # Conversation info
                                st.markdown(f"""
                                **Conversation ID:** `{conv_data['conversation_id']}`
                                **Status:** **{conv_data['status'].upper()}**
                                **Created:** {conv_data['student_profile']['created_at']}
                                """)

                                # Messages
                                st.markdown("### 💬 Message History")
                                for msg in conv_data['messages']:
                                    role_icons = {
                                        'user': '👤 STUDENT',
                                        'bot': '🤖 BOT',
                                        'system': '⚙️ SYSTEM'
                                    }
                                    role_class = {
                                        'user': 'user-msg',
                                        'bot': 'bot-msg',
                                        'system': 'system-msg'
                                    }

                                    confidence_info = ""
                                    if msg.get('confidence') is not None:
                                        confidence_info = f" (Confidence: {msg['confidence'] * 100:.0f}%)"

                                    st.markdown(f"""
                                    <div class="conversation-message {role_class.get(msg['role'], '')}">
                                        <strong>{role_icons.get(msg['role'], msg['role'].upper())}</strong>{confidence_info}<br>
                                        {msg['content'].replace(chr(10), '<br>')}
                                    </div>
                                    """, unsafe_allow_html=True)

                                # Bot notes
                                if conv_data.get('bot_notes'):
                                    st.markdown("### 🎯 Bot Notes for Advisor")
                                    notes = conv_data['bot_notes']

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.info(f"**Topics Discussed:** {', '.join(notes.get('topics_discussed', []))}")
                                        st.info(f"**Questions Asked:** {notes.get('questions_asked', 0)}")

                                    with col2:
                                        st.warning(f"**Escalation Reason:** {notes.get('escalation_reason', 'N/A')}")

                                    if notes.get('recommended_actions'):
                                        st.markdown("**Recommended Actions:**")
                                        for action in notes['recommended_actions']:
                                            st.markdown(f"→ {action}")

                                # Close button
                                if st.button("✖️ Close Conversation View", key=f"close_{item['conversation_id']}"):
                                    del st.session_state.viewing_conversation
                                    st.rerun()

                            else:
                                st.error(f"Error loading conversation (Status {conv_response.status_code})")

                        except Exception as e:
                            st.error(f"Error loading conversation: {str(e)}")

    else:
        st.error(f"Error fetching queue (Status {queue_response.status_code})")

except Exception as e:
    st.error(f"Error: {str(e)}")

# Sidebar
st.sidebar.markdown("### 🔄 Actions")
if st.sidebar.button("Refresh Queue"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.info("""
**Context Preservation:**
Every escalated conversation includes the complete message history, so you never need to ask students to repeat themselves.

**Prioritization:**
Conversations are shown with wait times to help you prioritize who to help first.

**Bot Notes:**
The bot provides recommended actions to help you quickly assist each student.
""")

# Footer
st.markdown("---")
st.caption("Auto-refresh enabled: Queue updates every 10 seconds" if auto_refresh else "Enable auto-refresh to see real-time updates")
