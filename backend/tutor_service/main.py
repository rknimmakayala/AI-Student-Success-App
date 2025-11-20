"""
AI Tutor Backend Service - Core Implementation
FastAPI backend with LangChain RAG for AI Strategy tutoring
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Tutor API",
    description="AI-powered tutor for AI Strategy graduate course",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    question: str = Field(..., description="Student's question", min_length=1)
    student_id: str = Field(..., description="Unique student identifier")
    context: Optional[Dict] = Field(default={}, description="Conversation context")
    topic: Optional[str] = Field(None, description="Current course topic/module")

class Source(BaseModel):
    """Source reference from course materials"""
    title: str
    module: str
    relevance_score: float

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str = Field(..., description="AI tutor's response")
    sources: List[Source] = Field(..., description="Course material references")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    follow_up_questions: List[str] = Field(..., description="Suggested follow-up questions")
    timestamp: datetime = Field(default_factory=datetime.now)

class CaseStudyRequest(BaseModel):
    """Request model for case study feedback"""
    case_id: str
    student_id: str
    analysis: str = Field(..., min_length=100, description="Student's case analysis")

class CaseStudyResponse(BaseModel):
    """Response model for case study feedback"""
    feedback: str
    score: int = Field(..., ge=0, le=100)
    strengths: List[str]
    improvements: List[str]
    recommendations: List[str]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: datetime

# ============================================================================
# RAG Chain Setup
# ============================================================================

class RAGService:
    """Service for Retrieval Augmented Generation"""

    def __init__(self):
        self.chain = None
        self._initialize_chain()

    def _initialize_chain(self):
        """Initialize LangChain RAG chain"""
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from langchain_pinecone import PineconeVectorStore
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema.runnable import RunnablePassthrough, RunnableMap
            from langchain.schema.output_parser import StrOutputParser

            # Initialize LLM
            llm = ChatOpenAI(
                model="gpt-4-turbo-preview",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY")
            )

            # Initialize vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = PineconeVectorStore(
                index_name="ai-tutor-course-content",
                embedding=embeddings
            )
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )

            # Create prompt template
            template = """You are an expert AI Strategy tutor for graduate students.
Your goal is to help students understand AI strategy concepts through clear explanations and real-world examples.

**Relevant Course Materials:**
{context}

**Student Question:**
{question}

**Student Background:**
- Completed modules: {completed_modules}
- Current knowledge level: {student_level}

**Instructions:**
1. Provide a clear, graduate-level explanation (200-300 words)
2. Include a relevant real-world business example (Google, Amazon, Microsoft, etc.)
3. Reference specific course materials when appropriate
4. Suggest 2-3 follow-up questions to deepen understanding
5. Use professional but approachable tone

**Response Format:**
**Explanation:**
[Your detailed explanation here]

**Real-World Example:**
[Concrete business example here]

**Follow-up Questions:**
1. [Question to check understanding]
2. [Question to encourage deeper thinking]
3. [Question to apply concept]
"""

            prompt = ChatPromptTemplate.from_template(template)

            # Create chain
            self.chain = (
                RunnableMap({
                    "context": lambda x: "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(x["question"])]),
                    "question": lambda x: x["question"],
                    "completed_modules": lambda x: ", ".join(x.get("completed_modules", [])),
                    "student_level": lambda x: x.get("student_level", "Intermediate")
                })
                | prompt
                | llm
                | StrOutputParser()
            )

            logger.info("✓ RAG chain initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG chain: {e}")
            self.chain = None

    async def generate_response(self, question: str, student_profile: Dict) -> str:
        """Generate response using RAG chain"""
        if not self.chain:
            raise HTTPException(status_code=503, detail="AI service not available")

        try:
            response = await self.chain.ainvoke({
                "question": question,
                **student_profile
            })
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize RAG service
rag_service = RAGService()

# ============================================================================
# Student Profile Service (Simplified)
# ============================================================================

class StudentProfileService:
    """Service for managing student profiles and progress"""

    def __init__(self):
        # In production, this would connect to PostgreSQL
        # For MVP, using in-memory storage
        self.profiles = {}

    def get_profile(self, student_id: str) -> Dict:
        """Get student profile"""
        if student_id not in self.profiles:
            # Create default profile
            self.profiles[student_id] = {
                "student_id": student_id,
                "completed_modules": [],
                "student_level": "Intermediate",
                "interaction_count": 0,
                "topics_mastered": []
            }
        return self.profiles[student_id]

    def update_interaction(self, student_id: str, question: str, topic: Optional[str]):
        """Update student profile after interaction"""
        profile = self.get_profile(student_id)
        profile["interaction_count"] += 1

        if topic and topic not in profile["completed_modules"]:
            # Simple logic: after 5 questions on a topic, mark as exposed
            # In production, this would be based on mastery assessments
            pass

    def get_mastery_level(self, student_id: str) -> str:
        """Determine student's mastery level"""
        profile = self.get_profile(student_id)

        if profile["interaction_count"] < 10:
            return "Novice"
        elif profile["interaction_count"] < 50:
            return "Intermediate"
        else:
            return "Advanced"

student_service = StudentProfileService()

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now()
    )

@app.post("/api/v1/tutor/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handle student question and return AI tutor response

    This endpoint:
    1. Retrieves student profile
    2. Queries knowledge base for relevant content
    3. Generates personalized response using RAG
    4. Updates student interaction history
    5. Returns response with sources and follow-ups
    """
    try:
        logger.info(f"Received question from student {request.student_id}: {request.question[:50]}...")

        # Get student profile
        student_profile = student_service.get_profile(request.student_id)
        student_profile["student_level"] = student_service.get_mastery_level(request.student_id)

        # Generate response using RAG
        response_text = await rag_service.generate_response(
            request.question,
            student_profile
        )

        # Update student profile
        student_service.update_interaction(
            request.student_id,
            request.question,
            request.topic
        )

        # Parse response (simplified - in production, use structured output)
        # Extract follow-up questions from response
        follow_ups = []
        if "Follow-up Questions:" in response_text:
            lines = response_text.split("\n")
            in_followups = False
            for line in lines:
                if "Follow-up Questions:" in line:
                    in_followups = True
                    continue
                if in_followups and line.strip().startswith(("1.", "2.", "3.")):
                    question = line.strip()[2:].strip()
                    if question:
                        follow_ups.append(question)

        # Mock sources (in production, get from retriever)
        sources = [
            Source(
                title="AI Strategy Fundamentals",
                module="Module 1",
                relevance_score=0.92
            ),
            Source(
                title="Competitive Advantage through AI",
                module="Module 3",
                relevance_score=0.85
            )
        ]

        return ChatResponse(
            answer=response_text,
            sources=sources,
            confidence=0.87,  # In production, calculate from retrieval scores
            follow_up_questions=follow_ups if follow_ups else [
                "How would you apply this concept in your organization?",
                "What challenges might you encounter?",
                "Can you think of other examples?"
            ]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/tutor/case-study", response_model=CaseStudyResponse)
async def analyze_case_study(request: CaseStudyRequest):
    """
    Analyze student's case study submission and provide feedback

    This endpoint:
    1. Retrieves case study details and rubric
    2. Analyzes student's submission
    3. Generates detailed feedback
    4. Provides score and recommendations
    """
    try:
        # In production, retrieve case study from database
        # For MVP, return mock feedback

        return CaseStudyResponse(
            feedback="""Your analysis demonstrates a good understanding of AI strategy fundamentals.
You correctly identified the key business objectives and aligned AI capabilities accordingly.

Your discussion of organizational readiness was particularly strong, showing awareness of
change management challenges.

However, the analysis could be strengthened by deeper consideration of competitive dynamics
and more specific implementation timelines.""",
            score=82,
            strengths=[
                "Clear articulation of business objectives",
                "Strong understanding of organizational change management",
                "Good use of relevant frameworks (AI Transformation Playbook)"
            ],
            improvements=[
                "Include competitive analysis using Porter's 5 Forces + AI lens",
                "Provide more specific implementation timeline with milestones",
                "Discuss potential risks and mitigation strategies in more detail"
            ],
            recommendations=[
                "Review Module 4 on Competitive Strategy with AI",
                "Practice creating detailed implementation roadmaps",
                "Consider reading case studies on failed AI implementations"
            ]
        )

    except Exception as e:
        logger.error(f"Error in case study endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/analytics/student/{student_id}/progress")
async def get_student_progress(student_id: str):
    """Get student progress and analytics"""
    try:
        profile = student_service.get_profile(student_id)

        return {
            "student_id": student_id,
            "overall_progress": min(profile["interaction_count"] * 2, 100),  # Simplified
            "knowledge_level": student_service.get_mastery_level(student_id),
            "interactions": profile["interaction_count"],
            "topics_mastered": profile["topics_mastered"],
            "completed_modules": profile["completed_modules"],
            "engagement_metrics": {
                "total_questions": profile["interaction_count"],
                "avg_questions_per_week": profile["interaction_count"] / 4  # Simplified
            }
        }

    except Exception as e:
        logger.error(f"Error in progress endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting AI Tutor API...")
    logger.info("✓ Services initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down AI Tutor API...")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
