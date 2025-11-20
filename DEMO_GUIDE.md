# AI Tutor Demo Guide

This guide shows you how to explore all the generated outputs and run a working demo of the AI tutor system.

## 📁 Part 1: View Generated Documentation

All specifications and code are in your repository. Here's what to look at:

### Main Documentation Files

```bash
# Navigate to your project
cd AI-Student-Success-App

# View the complete specification (50+ pages)
cat AI_TUTOR_SPECIFICATION.md
# OR open in your editor/browser for better formatting

# View executive presentation (for stakeholders)
cat STAKEHOLDER_PRESENTATION.md

# View implementation guide
cat IMPLEMENTATION_QUICK_START.md

# View multi-agent system overview
cat AGENTIC_SYSTEM_OVERVIEW.md
```

### Generated JSON Specifications

```bash
# View detailed requirements from Product Manager agent
cat agents/examples/ai_tutor_pm_requirements.json | jq '.'

# View architecture from System Architect agent
cat agents/examples/ai_tutor_architecture.json | jq '.'

# View AI/LLM strategy from AI Engineer agent
cat agents/examples/ai_tutor_ai_strategy.json | jq '.'

# View complete workflow state
cat agents/examples/ai_tutor_complete_workflow.json | jq '.'
```

**Key Sections to Review:**

1. **User Personas** (in pm_requirements.json):
   - Sarah (Working Professional Student)
   - Dr. Martinez (Course Instructor)

2. **Core Requirements** (in pm_requirements.json):
   - 8 functional requirements
   - 5 non-functional requirements
   - User stories with acceptance criteria

3. **System Architecture** (in architecture.json):
   - 9 microservices
   - API contracts
   - Technology stack
   - Security design

4. **AI Strategy** (in ai_strategy.json):
   - 3 reasoning chains
   - Prompt templates
   - Evaluation framework

---

## 🎬 Part 2: Run the Multi-Agent Workflow

See the agents in action by running the workflow that generated everything:

```bash
cd agents

# Run the AI Tutor workflow
python examples/ai_tutor_workflow.py
```

**What you'll see:**
```
====================================================================================================
🎓 AI TUTOR FOR AI STRATEGY COURSE - MULTI-AGENT WORKFLOW
====================================================================================================

✓ Initialized orchestrator and 7 specialized agents

====================================================================================================
📋 PHASE 1: DISCOVERY
====================================================================================================

📝 Task Created: Analyze AI Tutor Requirements for AI Strategy Course
   Assigned to: Product Manager

✅ Product Manager Analysis Complete
   Deliverable: requirements_document
   User Personas: 2
   Functional Requirements: 8
   ...

====================================================================================================
🏗️ PHASE 2: ARCHITECTURE
====================================================================================================

... (continues through all phases)
```

This workflow:
- Creates 3 tasks (PM analysis, Architecture design, AI strategy)
- Executes them in sequence with dependencies
- Generates complete specifications
- Saves outputs to JSON files

---

## 🚀 Part 3: Run the Backend API (Live Demo)

### Prerequisites

Install dependencies:
```bash
# Install Python 3.11+
python --version  # Should be 3.11 or higher

# Install jq for JSON formatting (optional)
# macOS: brew install jq
# Ubuntu: sudo apt-get install jq
# Windows: https://stedolan.github.io/jq/download/
```

### Option A: Quick Demo (Without Real LLM)

Run the API with mock responses (no API keys needed):

```bash
cd backend/tutor_service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install minimal dependencies
pip install fastapi uvicorn pydantic python-dotenv

# Run the server
python main.py
```

**Test the API:**
```bash
# In a new terminal, test health endpoint
curl http://localhost:8000/health | jq '.'

# Expected output:
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00"
}

# Test chat endpoint (will use mock data)
curl -X POST http://localhost:8000/api/v1/tutor/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is AI-first strategy?",
    "student_id": "demo_student"
  }' | jq '.'

# Test student progress endpoint
curl http://localhost:8000/api/v1/analytics/student/demo_student/progress | jq '.'
```

### Option B: Full Demo (With Real LLM)

To see the actual AI tutor in action with GPT-4:

**Step 1: Set up environment**
```bash
cd backend/tutor_service

# Create .env file with your API keys
cat > .env << EOF
OPENAI_API_KEY=sk-your-actual-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
DATABASE_URL=postgresql://tutor_admin:dev_password@localhost:5432/ai_tutor
REDIS_URL=redis://localhost:6379
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=us-east-1-aws
EOF
```

**Step 2: Start infrastructure**
```bash
cd ../../infrastructure/docker
docker-compose up -d

# Wait for services to start (30 seconds)
sleep 30

# Verify services are running
docker-compose ps
```

**Step 3: Install full dependencies**
```bash
cd ../../backend/tutor_service
source venv/bin/activate
pip install -r requirements.txt
```

**Step 4: Run the API**
```bash
uvicorn main:app --reload --log-level info
```

**Step 5: Test with real AI**
```bash
# Ask a complex question
curl -X POST http://localhost:8000/api/v1/tutor/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain the difference between AI-first and AI-enabled strategy with examples from tech companies",
    "student_id": "test_student_123",
    "context": {}
  }' | jq '.'
```

**Expected response structure:**
```json
{
  "answer": "**Explanation:**\nAI-first strategy means...\n\n**Real-World Example:**\nGoogle's transformation...",
  "sources": [
    {
      "title": "AI Strategy Fundamentals",
      "module": "Module 1",
      "relevance_score": 0.92
    }
  ],
  "confidence": 0.87,
  "follow_up_questions": [
    "How would you apply this in your organization?",
    "What challenges might you encounter?"
  ],
  "timestamp": "2024-01-15T10:30:00"
}
```

---

## 🗄️ Part 4: View Database

If you started Docker containers, you can view the database:

### Using pgAdmin (Web UI)

1. Open browser: http://localhost:5050
2. Login:
   - Email: `admin@aitutor.com`
   - Password: `admin`
3. Add server:
   - Name: `AI Tutor DB`
   - Host: `postgres`
   - Port: `5432`
   - Database: `ai_tutor`
   - Username: `tutor_admin`
   - Password: `dev_password_change_in_prod`

4. Browse tables:
   - `students` - Student profiles
   - `modules` - Course modules (5 sample modules)
   - `student_progress` - Progress tracking
   - `interactions` - Q&A history
   - `case_studies` - Case study assignments
   - `case_submissions` - Student submissions

### Using psql (Command Line)

```bash
# Connect to database
docker exec -it ai_tutor_db psql -U tutor_admin -d ai_tutor

# View modules
SELECT * FROM modules;

# View schema
\dt

# Exit
\q
```

---

## 📊 Part 5: Interactive API Documentation

Once the API is running, visit:

**http://localhost:8000/docs**

This gives you:
- **Interactive Swagger UI**
- All API endpoints documented
- "Try it out" button to test each endpoint
- Request/response examples
- Schema definitions

**Key endpoints to try:**
1. `GET /health` - Health check
2. `POST /api/v1/tutor/chat` - Ask a question
3. `POST /api/v1/tutor/case-study` - Submit case analysis
4. `GET /api/v1/analytics/student/{student_id}/progress` - View progress

---

## 🎥 Part 6: Demo Scenarios

### Scenario 1: Student Asks a Question

```bash
# Student asks about AI strategy
curl -X POST http://localhost:8000/api/v1/tutor/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key components of a successful AI transformation strategy?",
    "student_id": "sarah_student_001",
    "context": {},
    "topic": "Module 5: AI Implementation Frameworks"
  }' | jq '.'
```

### Scenario 2: Case Study Analysis

```bash
# Student submits case analysis
curl -X POST http://localhost:8000/api/v1/tutor/case-study \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "netflix_ai_case",
    "student_id": "sarah_student_001",
    "analysis": "Netflix has successfully implemented AI across their platform, primarily in their recommendation system which drives 80% of viewing. Their AI-first approach includes personalized thumbnails, content creation insights, and operational optimization. Key success factors include strong data infrastructure, executive support, and iterative deployment."
  }' | jq '.'
```

**Expected response:**
```json
{
  "feedback": "Your analysis demonstrates good understanding...",
  "score": 82,
  "strengths": [
    "Clear articulation of business objectives",
    "Good use of relevant frameworks"
  ],
  "improvements": [
    "Include competitive analysis",
    "Provide more specific implementation timeline"
  ],
  "recommendations": [
    "Review Module 4 on Competitive Strategy",
    "Practice creating detailed roadmaps"
  ]
}
```

### Scenario 3: Track Student Progress

```bash
# View student progress over time
curl http://localhost:8000/api/v1/analytics/student/sarah_student_001/progress | jq '.'
```

**Expected response:**
```json
{
  "student_id": "sarah_student_001",
  "overall_progress": 45,
  "knowledge_level": "Intermediate",
  "interactions": 23,
  "topics_mastered": [],
  "completed_modules": [],
  "engagement_metrics": {
    "total_questions": 23,
    "avg_questions_per_week": 5.75
  }
}
```

---

## 🔍 Part 7: Explore Agent Outputs in Detail

### View User Personas

```bash
cat agents/examples/ai_tutor_pm_requirements.json | jq '.user_personas'
```

**Output:**
```json
[
  {
    "name": "Sarah - Working Professional Student",
    "age": 32,
    "role": "Product Manager at tech company",
    "goals": [
      "Understand how to implement AI strategy in her organization",
      "Balance coursework with full-time job",
      "Get practical, applicable knowledge"
    ],
    "pain_points": [
      "Limited time for synchronous learning",
      "Needs quick clarification on complex concepts",
      "Wants real-world examples, not just theory"
    ]
  },
  ...
]
```

### View Requirements

```bash
cat agents/examples/ai_tutor_pm_requirements.json | jq '.core_requirements.functional'
```

### View Architecture Components

```bash
cat agents/examples/ai_tutor_architecture.json | jq '.components[] | {name, purpose, technology}'
```

### View AI Reasoning Chains

```bash
cat agents/examples/ai_tutor_ai_strategy.json | jq '.reasoning_chains'
```

### View Prompt Templates

```bash
cat agents/examples/ai_tutor_ai_strategy.json | jq '.prompt_templates.concept_explanation.template'
```

---

## 📱 Part 8: Visual Demo (Screenshots)

While we don't have a frontend yet, here's what the API responses look like:

### Health Check Response
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

### Chat Response (Actual AI Tutor)
```json
{
  "answer": "**Explanation:**\nAI-first strategy is an organizational approach where artificial intelligence is positioned as the primary driver of business value creation...\n\n**Real-World Example:**\nConsider Google's transformation under Sundar Pichai. They shifted from 'mobile-first' to 'AI-first'...\n\n**Follow-up Questions:**\n1. How would you assess if your organization is ready for AI-first?\n2. What are the risks of AI-first vs. AI-enabled approaches?",
  "sources": [
    {
      "title": "AI Strategy Fundamentals",
      "module": "Module 1",
      "relevance_score": 0.92
    }
  ],
  "confidence": 0.87,
  "follow_up_questions": [
    "How would you assess if your organization is ready for AI-first?",
    "What are the risks of AI-first vs. AI-enabled approaches?"
  ],
  "timestamp": "2024-01-15T10:30:15.678901"
}
```

---

## 🧪 Part 9: Testing the System

### Run Unit Tests (Optional)

```bash
cd backend/tutor_service

# Install test dependencies
pip install pytest pytest-asyncio httpx

# Create simple test file
cat > test_api.py << 'EOF'
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_chat_endpoint():
    response = client.post(
        "/api/v1/tutor/chat",
        json={
            "question": "What is AI strategy?",
            "student_id": "test_student"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "confidence" in data

def test_student_progress():
    response = client.get("/api/v1/analytics/student/test_student/progress")
    assert response.status_code == 200
    data = response.json()
    assert "student_id" in data
    assert "overall_progress" in data
EOF

# Run tests
pytest test_api.py -v
```

---

## 📈 Part 10: Performance Testing

### Load Test (Simple)

```bash
# Install Apache Bench (if not installed)
# macOS: brew install httpd
# Ubuntu: sudo apt-get install apache2-utils

# Test health endpoint (1000 requests, 10 concurrent)
ab -n 1000 -c 10 http://localhost:8000/health

# Test chat endpoint
ab -n 100 -c 5 -p chat_request.json -T application/json http://localhost:8000/api/v1/tutor/chat
```

**Create test request file:**
```bash
cat > chat_request.json << 'EOF'
{
  "question": "What is AI strategy?",
  "student_id": "load_test_student"
}
EOF
```

---

## 🎯 Quick Reference: See Everything

### 1-Minute Demo
```bash
# View main specification
cat AI_TUTOR_SPECIFICATION.md | head -200

# View requirements
cat agents/examples/ai_tutor_pm_requirements.json | jq '.core_requirements.functional[0:3]'

# Run workflow
python agents/examples/ai_tutor_workflow.py | head -50
```

### 5-Minute Demo
```bash
# Start services
cd infrastructure/docker && docker-compose up -d

# Start API
cd ../../backend/tutor_service
python main.py &

# Test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/tutor/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AI strategy?", "student_id": "demo"}' | jq '.'
```

### 30-Minute Deep Dive
1. Read `AI_TUTOR_SPECIFICATION.md` (15 min)
2. Run multi-agent workflow (5 min)
3. Start backend and test API (5 min)
4. Explore database and API docs (5 min)

---

## 🐛 Troubleshooting

### API won't start
```bash
# Check if port 8000 is in use
lsof -i :8000
# Kill process if needed
kill -9 <PID>
```

### Docker services won't start
```bash
# Check Docker is running
docker --version

# View logs
docker-compose logs

# Restart services
docker-compose down
docker-compose up -d
```

### Dependencies won't install
```bash
# Upgrade pip
pip install --upgrade pip

# Install one by one
pip install fastapi
pip install uvicorn
pip install pydantic
```

### Missing API keys
The system will work without API keys (uses mock data) for basic testing. For full AI functionality, you need:
- OpenAI API key (from platform.openai.com)
- Pinecone API key (from pinecone.io)

---

## 📞 Next Steps

1. **Review Documentation**: Start with `AI_TUTOR_SPECIFICATION.md`
2. **Run Workflow**: See agents in action with `python agents/examples/ai_tutor_workflow.py`
3. **Start API**: Follow Option A (Quick Demo) above
4. **Explore API**: Visit http://localhost:8000/docs
5. **Test Endpoints**: Use curl commands or Swagger UI
6. **Review Code**: Check out `backend/tutor_service/main.py`

---

**All outputs are in your repository and ready to explore!** 🚀
