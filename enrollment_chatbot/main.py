"""
Enrollment FAQ Chatbot Backend with Human-in-the-Loop
Production-ready FastAPI application for competency-based university enrollment

Features:
- Automated FAQ responses
- Intelligent escalation to human advisors
- Context preservation during handoff
- Advisor dashboard and queue management
- Multi-channel support (web, SMS, WhatsApp)
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import json
import logging
import uuid
import os
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Enrollment FAQ Chatbot API",
    description="Chatbot with Human-in-the-Loop for university enrollment",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Data Models
# ============================================================================

class MessageRole(str, Enum):
    USER = "user"
    BOT = "bot"
    ADVISOR = "advisor"
    SYSTEM = "system"

class ConversationStatus(str, Enum):
    BOT_HANDLING = "bot_handling"
    ESCALATED = "escalated"
    ADVISOR_ACTIVE = "advisor_active"
    RESOLVED = "resolved"

class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence: Optional[float] = None

class StudentProfile(BaseModel):
    student_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    program_interest: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    student_id: str
    message: str
    student_name: Optional[str] = None
    channel: str = "web"  # web, sms, whatsapp

class ChatResponse(BaseModel):
    message: str
    role: MessageRole
    confidence: float
    escalation_suggested: bool = False
    escalation_reason: Optional[str] = None
    follow_up_questions: List[str] = []
    sources: List[str] = []

class EscalationRequest(BaseModel):
    conversation_id: str
    reason: str

class AdvisorHandoffResponse(BaseModel):
    advisor_id: str
    advisor_name: str
    conversation_history: List[Message]
    student_profile: StudentProfile
    bot_notes: Dict[str, Any]

# ============================================================================
# In-Memory Storage (Production: Use Redis/PostgreSQL)
# ============================================================================

class ChatbotState:
    def __init__(self):
        self.conversations: Dict[str, Dict] = {}
        self.advisor_queue: deque = deque()
        self.active_advisors: Dict[str, str] = {}  # advisor_id -> conversation_id
        self.faq_database = self._load_faq_database()

    def _load_faq_database(self) -> Dict:
        """Load FAQ database"""
        return {
            "what_is_cbe": {
                "question": "What is competency-based education?",
                "keywords": ["competency", "cbe", "competency-based", "self-paced"],
                "response": """Great question! Competency-based education (CBE) is a flexible, self-paced approach where you progress by demonstrating mastery of skills, not by spending time in class.

**Key differences from traditional college:**
• **Self-paced**: Move as fast or slow as you need
• **No semesters**: Start monthly, work on your own schedule
• **Prior knowledge counts**: If you already know material, prove it and move on
• **Subscription model**: Pay per 6-month term, take as many courses as you can complete
• **Real-world skills**: Focus on what employers need

**Example:** If you already know accounting from work, take the assessment. Pass it, and you're done with that course - no need to sit through lectures!

Want to learn more about how assessments work?""",
                "confidence": 0.95,
                "category": "competency_model"
            },
            "tuition_cost": {
                "question": "How much does tuition cost?",
                "keywords": ["cost", "tuition", "price", "expensive", "afford", "how much"],
                "response": """Our tuition structure is designed to be affordable and predictable:

**Subscription Model:**
• $3,000 per 6-month term (undergraduate)
• $4,000 per 6-month term (graduate)
• Take as many courses as you can complete in that term
• No per-credit fees

**Additional costs:**
• $50 application fee (often waived during promotions)
• ~$200-300 for course materials per term
• Some programs have proctoring fees ($25-50 per assessment)

**Financial aid available:**
• Federal student aid (FAFSA)
• Military benefits (GI Bill, TA)
• Employer tuition assistance
• Payment plans

**Example:** If you complete 4 courses in 6 months = $3,000 total. That's $750 per course!

Would you like information about financial aid or payment options?""",
                "confidence": 0.95,
                "category": "tuition"
            },
            "transfer_credits": {
                "question": "Will my credits transfer?",
                "keywords": ["transfer", "credits", "previous college", "community college"],
                "response": """Good question! Credit transfer depends on several factors:

**Generally transferable:**
• Regionally accredited colleges/universities
• Grades of C or better
• Courses relevant to your program
• Completed within last 10 years (for most programs)

**We'll evaluate:**
• Official transcripts from all institutions
• Course descriptions and syllabi
• Alignment with our competency outcomes

**To get started:**
1. Request transcripts from previous schools
2. Submit them with your application
3. We'll provide a transfer credit evaluation (usually within 2 weeks)

Many students transfer 30-60 credits! Want me to connect you with an enrollment advisor who can give you a preliminary assessment?""",
                "confidence": 0.85,
                "escalation_trigger": "multiple_schools",
                "category": "transfer_credits"
            },
            "work_fulltime": {
                "question": "Can I work full-time and attend?",
                "keywords": ["work full-time", "working", "job", "balance", "flexible"],
                "response": """Absolutely! Most of our students work full-time. Here's how it works:

**Flexibility built for working adults:**
• 100% online - study anytime, anywhere
• No scheduled class meetings
• Self-paced - work ahead on weekends or slow weeks
• Mobile-friendly platform for learning on-the-go

**Typical time commitment:**
• 15-20 hours per week for full-time progress
• Can do more in busy work periods, catch up in slow periods
• Average student completes bachelor's in 2-3 years (part-time)

**Real student example:**
Maria, a retail manager, studied early mornings (5-7am) and evenings (9-11pm). She completed her bachelor's in 2.5 years while working 40+ hours per week.

**Support available:**
• Program mentors (academic coaches)
• Flexible assessment scheduling
• Career-relevant coursework you can apply at work

Want to talk to an enrollment advisor about creating a plan that fits your schedule?""",
                "confidence": 0.90,
                "category": "program_format"
            },
            "gi_bill": {
                "question": "Do you accept GI Bill?",
                "keywords": ["gi bill", "military", "veteran", "va benefits", "jst"],
                "response": """Yes! We proudly support military and veterans with their education benefits:

**Accepted Benefits:**
• GI Bill® (Chapters 30, 33, 1606)
• Vocational Rehabilitation (Chapter 31)
• Tuition Assistance (TA)
• Yellow Ribbon Program participant

**Military Credit:**
• Joint Services Transcript (JST) evaluated for credit
• ACE-recommended military training accepted
• Many veterans transfer 20-40 credits from military experience

**How it works:**
1. Apply for admission
2. Submit Certificate of Eligibility (COE)
3. We certify your enrollment to VA
4. Start classes while VA processes payment

**Special for veterans:**
• Priority registration
• Veteran student support services
• Military-friendly scheduling

Want to speak with our military benefits coordinator for a personalized evaluation?""",
                "confidence": 0.88,
                "category": "financial_aid"
            },
            "admission_requirements": {
                "question": "What are admission requirements?",
                "keywords": ["admission", "requirements", "apply", "qualify", "eligible"],
                "response": """Our admission requirements are straightforward:

**Undergraduate Programs:**
• High school diploma or GED
• No SAT/ACT required
• $50 application fee (often waived)
• Official high school transcript or GED scores

**Graduate Programs:**
• Bachelor's degree from accredited institution
• Minimum 2.5 GPA (some programs 3.0)
• Official transcripts from all colleges attended
• Resume or CV
• No GRE/GMAT required for most programs

**Application Process:**
1. Complete online application (15-20 minutes)
2. Submit transcripts
3. Admission decision within 2 weeks
4. Start date: Monthly cohorts (next start in 3 weeks!)

**No application essay required!**

Ready to start your application?""",
                "confidence": 0.93,
                "category": "admissions"
            },
            "programs_offered": {
                "question": "What programs do you offer?",
                "keywords": ["programs", "degrees", "majors", "what can i study"],
                "response": """We offer a wide range of programs designed for career advancement:

**Undergraduate (Bachelor's):**
• Business Administration
• Information Technology
• Healthcare Administration
• Cybersecurity
• Computer Science
• Accounting
• Marketing
• Human Resources Management

**Graduate (Master's):**
• MBA (Master of Business Administration)
• M.S. in Information Technology Management
• M.S. in Cybersecurity & Information Assurance
• M.S. in Data Analytics
• M.Ed. in Educational Leadership

**All programs are:**
• 100% online
• Self-paced (competency-based)
• Accredited by regional accrediting body
• Designed for working professionals

Which program interests you? I can provide more details about curriculum, career outcomes, and time to completion!""",
                "confidence": 0.95,
                "category": "programs"
            },
            "how_long_to_complete": {
                "question": "How long does it take to complete a degree?",
                "keywords": ["how long", "time", "duration", "fast", "quick"],
                "response": """Great question! Time to completion varies based on YOUR pace:

**Undergraduate (Bachelor's):**
• Average: 2-3 years (part-time, working full-time)
• Fast track: 12-18 months (if you have transfer credits and can study 25+ hrs/week)
• Standard: 3-4 years (15-20 hrs/week study time)

**Graduate (Master's):**
• Average: 18-24 months
• Fast track: 12 months (intensive)
• Standard: 2 years

**Factors that speed up completion:**
• Transfer credits (can reduce time by 50%)
• Prior learning assessment (PLA) - get credit for work experience
• More study time per week
• Prior knowledge in subject area

**Real Examples:**
• James (IT professional): Completed bachelor's in 14 months with 45 transfer credits
• Maria (retail manager): Completed bachelor's in 2.5 years, working full-time

Want to discuss how quickly YOU could finish based on your situation?""",
                "confidence": 0.87,
                "category": "program_format"
            }
        }

    def create_conversation(self, student_id: str, channel: str = "web") -> str:
        """Create new conversation"""
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = {
            "id": conversation_id,
            "student_id": student_id,
            "channel": channel,
            "status": ConversationStatus.BOT_HANDLING,
            "messages": [],
            "student_profile": {
                "student_id": student_id,
                "created_at": datetime.now().isoformat()
            },
            "bot_context": {
                "questions_asked": 0,
                "topics_discussed": [],
                "escalation_triggers": []
            },
            "created_at": datetime.now().isoformat()
        }
        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation by ID"""
        return self.conversations.get(conversation_id)

    def add_message(self, conversation_id: str, role: MessageRole, content: str, confidence: Optional[float] = None):
        """Add message to conversation"""
        if conversation_id in self.conversations:
            self.conversations[conversation_id]["messages"].append({
                "role": role.value,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "confidence": confidence
            })

    def escalate_to_human(self, conversation_id: str, reason: str):
        """Escalate conversation to human advisor"""
        if conversation_id in self.conversations:
            conv = self.conversations[conversation_id]
            conv["status"] = ConversationStatus.ESCALATED
            conv["escalation_reason"] = reason
            conv["escalated_at"] = datetime.now().isoformat()
            self.advisor_queue.append(conversation_id)
            logger.info(f"Conversation {conversation_id} escalated: {reason}")

# Initialize state
state = ChatbotState()

# ============================================================================
# FAQ Matching Engine
# ============================================================================

class FAQMatcher:
    @staticmethod
    def find_best_match(question: str, faq_db: Dict) -> tuple[Optional[Dict], float]:
        """Find best matching FAQ"""
        question_lower = question.lower()
        best_match = None
        best_score = 0.0

        for faq_id, faq_data in faq_db.items():
            score = 0.0
            keyword_matches = 0

            # Check keyword matches
            for keyword in faq_data.get("keywords", []):
                if keyword.lower() in question_lower:
                    keyword_matches += 1
                    score += 0.2

            # Boost score if multiple keywords match
            if keyword_matches >= 2:
                score += 0.3

            # Check question similarity (simple approach)
            if faq_data["question"].lower() in question_lower or question_lower in faq_data["question"].lower():
                score += 0.4

            if score > best_score:
                best_score = min(score, 1.0)  # Cap at 1.0
                best_match = faq_data

        return best_match, best_score

    @staticmethod
    def should_escalate(question: str, confidence: float, conversation_context: Dict) -> tuple[bool, Optional[str]]:
        """Determine if question should be escalated"""

        # Low confidence threshold
        if confidence < 0.70:
            return True, "Low confidence in answer"

        # Check for explicit human request
        human_keywords = ["speak to person", "talk to human", "real person", "advisor", "counselor"]
        if any(keyword in question.lower() for keyword in human_keywords):
            return True, "Student requested human advisor"

        # Check for frustration indicators
        frustration_keywords = ["frustrated", "not helping", "doesn't answer", "waste of time"]
        if any(keyword in question.lower() for keyword in frustration_keywords):
            return True, "Student expressed frustration"

        # Check for complex situations
        complex_keywords = ["multiple schools", "3 colleges", "several institutions", "complicated situation"]
        if any(keyword in question.lower() for keyword in complex_keywords):
            return True, "Complex situation requiring personalized guidance"

        # Check if student asked many follow-ups on same topic
        if conversation_context["questions_asked"] >= 3:
            recent_topics = conversation_context.get("topics_discussed", [])
            if len(recent_topics) >= 3 and len(set(recent_topics[-3:])) == 1:
                return True, "Multiple follow-up questions on same topic"

        return False, None

# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - handles student questions
    """
    try:
        logger.info(f"Received message from {request.student_id}: {request.message[:50]}...")

        # Find or create conversation
        conversation_id = None
        for conv_id, conv in state.conversations.items():
            if conv["student_id"] == request.student_id and conv["status"] != ConversationStatus.RESOLVED:
                conversation_id = conv_id
                break

        if not conversation_id:
            conversation_id = state.create_conversation(request.student_id, request.channel)

        conversation = state.get_conversation(conversation_id)

        # Add user message
        state.add_message(conversation_id, MessageRole.USER, request.message)
        conversation["bot_context"]["questions_asked"] += 1

        # Find best FAQ match
        faq_match, confidence = FAQMatcher.find_best_match(request.message, state.faq_database)

        if faq_match:
            response_text = faq_match["response"]
            category = faq_match.get("category", "general")
            conversation["bot_context"]["topics_discussed"].append(category)
        else:
            response_text = """I want to make sure you get the best answer. Let me connect you with an enrollment advisor who can help with your specific question.

They'll see our conversation so you won't need to repeat anything. One moment..."""
            confidence = 0.50

        # Check if should escalate
        should_escalate, escalation_reason = FAQMatcher.should_escalate(
            request.message,
            confidence,
            conversation["bot_context"]
        )

        if should_escalate:
            state.escalate_to_human(conversation_id, escalation_reason)
            state.add_message(conversation_id, MessageRole.SYSTEM, f"Escalated: {escalation_reason}", confidence)

        # Add bot response
        state.add_message(conversation_id, MessageRole.BOT, response_text, confidence)

        # Generate follow-up questions
        follow_ups = []
        if faq_match and confidence > 0.85:
            if category == "tuition":
                follow_ups = [
                    "Would you like information about financial aid?",
                    "Want to know about payment plans?",
                    "Interested in scholarship opportunities?"
                ]
            elif category == "competency_model":
                follow_ups = [
                    "Want to learn how assessments work?",
                    "Curious about how fast you can complete?",
                    "Interested in Prior Learning Assessment (PLA)?"
                ]
            elif category == "transfer_credits":
                follow_ups = [
                    "Want to speak with an advisor for a preliminary evaluation?",
                    "Interested in how Prior Learning Assessment works?",
                    "Need help requesting transcripts?"
                ]

        return ChatResponse(
            message=response_text,
            role=MessageRole.BOT,
            confidence=confidence,
            escalation_suggested=should_escalate,
            escalation_reason=escalation_reason,
            follow_up_questions=follow_ups,
            sources=[faq_match.get("category", "general")] if faq_match else []
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/escalate")
async def escalate(request: EscalationRequest):
    """
    Manually escalate conversation to human advisor
    """
    conversation = state.get_conversation(request.conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    state.escalate_to_human(request.conversation_id, request.reason)

    return {
        "status": "escalated",
        "conversation_id": request.conversation_id,
        "position_in_queue": len(state.advisor_queue)
    }


@app.get("/api/v1/advisor/queue")
async def get_advisor_queue():
    """
    Get current advisor queue (for advisor dashboard)
    """
    queue_data = []
    for conv_id in list(state.advisor_queue):
        conv = state.get_conversation(conv_id)
        if conv:
            queue_data.append({
                "conversation_id": conv_id,
                "student_id": conv["student_id"],
                "channel": conv["channel"],
                "escalation_reason": conv.get("escalation_reason"),
                "messages_count": len(conv["messages"]),
                "escalated_at": conv.get("escalated_at"),
                "wait_time_minutes": (datetime.now() - datetime.fromisoformat(conv.get("escalated_at", conv["created_at"]))).seconds // 60
            })

    return {
        "queue": queue_data,
        "count": len(queue_data)
    }


@app.get("/api/v1/advisor/conversation/{conversation_id}")
async def get_conversation_for_advisor(conversation_id: str):
    """
    Get full conversation context for advisor handoff
    """
    conversation = state.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Format messages for advisor
    messages = [
        {
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": msg["timestamp"],
            "confidence": msg.get("confidence")
        }
        for msg in conversation["messages"]
    ]

    # Generate advisor notes
    bot_notes = {
        "questions_asked": conversation["bot_context"]["questions_asked"],
        "topics_discussed": list(set(conversation["bot_context"]["topics_discussed"])),
        "escalation_reason": conversation.get("escalation_reason"),
        "recommended_actions": []
    }

    # Add recommendations based on topics
    topics = conversation["bot_context"]["topics_discussed"]
    if "transfer_credits" in topics:
        bot_notes["recommended_actions"].append("Discuss transcript evaluation process")
        bot_notes["recommended_actions"].append("Mention Prior Learning Assessment (PLA)")
    if "financial_aid" in topics:
        bot_notes["recommended_actions"].append("Ask about FAFSA completion")
        bot_notes["recommended_actions"].append("Discuss payment options")
    if "competency_model" in topics:
        bot_notes["recommended_actions"].append("Clarify self-paced vs. traditional")
        bot_notes["recommended_actions"].append("Share student success stories")

    return {
        "conversation_id": conversation_id,
        "student_profile": conversation["student_profile"],
        "messages": messages,
        "bot_notes": bot_notes,
        "status": conversation["status"]
    }


@app.post("/api/v1/advisor/claim/{conversation_id}")
async def claim_conversation(conversation_id: str, advisor_id: str, advisor_name: str):
    """
    Advisor claims a conversation from queue
    """
    conversation = state.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Remove from queue
    if conversation_id in state.advisor_queue:
        state.advisor_queue.remove(conversation_id)

    # Update conversation status
    conversation["status"] = ConversationStatus.ADVISOR_ACTIVE
    conversation["advisor_id"] = advisor_id
    conversation["advisor_name"] = advisor_name
    conversation["claimed_at"] = datetime.now().isoformat()

    # Add system message
    state.add_message(
        conversation_id,
        MessageRole.SYSTEM,
        f"Conversation claimed by advisor {advisor_name}"
    )

    return {
        "status": "claimed",
        "conversation_id": conversation_id,
        "advisor_name": advisor_name
    }


@app.get("/api/v1/stats")
async def get_stats():
    """
    Get chatbot statistics
    """
    total_conversations = len(state.conversations)
    bot_handled = sum(1 for c in state.conversations.values() if c["status"] == ConversationStatus.BOT_HANDLING)
    escalated = sum(1 for c in state.conversations.values() if c["status"] in [ConversationStatus.ESCALATED, ConversationStatus.ADVISOR_ACTIVE])
    resolved = sum(1 for c in state.conversations.values() if c["status"] == ConversationStatus.RESOLVED)

    return {
        "total_conversations": total_conversations,
        "bot_handled": bot_handled,
        "escalated": escalated,
        "resolved": resolved,
        "current_queue_length": len(state.advisor_queue),
        "bot_resolution_rate": round((bot_handled / total_conversations * 100) if total_conversations > 0 else 0, 1)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "conversations_active": len(state.conversations),
        "queue_length": len(state.advisor_queue)
    }


# ============================================================================
# Web Interface - Static Files
# ============================================================================

# Serve static files (HTML/CSS/JS)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def read_root():
    """Serve the main web interface"""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Enrollment FAQ Chatbot API", "docs": "/docs"}


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("🚀 Starting Enrollment FAQ Chatbot with HITL...")
    logger.info(f"✓ Loaded {len(state.faq_database)} FAQs")
    logger.info("✓ Services initialized")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
