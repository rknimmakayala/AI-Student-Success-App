# AI Tutor Implementation Quick Start Guide

This guide will get you from specifications to a working prototype in the shortest time possible.

## Prerequisites Checklist

Before you begin, ensure you have:

- [ ] AWS account with admin access
- [ ] GitHub repository created
- [ ] OpenAI API key (for GPT-4)
- [ ] Anthropic API key (optional, for Claude fallback)
- [ ] Python 3.11+ installed
- [ ] Node.js 18+ installed
- [ ] Docker installed (for local development)
- [ ] Terraform installed (for infrastructure)

## Week 1: Environment Setup

### Day 1: Repository and Access

```bash
# Clone the repository
git clone https://github.com/your-org/ai-tutor.git
cd ai-tutor

# Create project structure
mkdir -p backend/{tutor_service,knowledge_base,student_profile,analytics}
mkdir -p frontend/{src,public}
mkdir -p infrastructure/{terraform,docker}
mkdir -p docs
```

Create `.env` file:
```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
AWS_REGION=us-east-1
DATABASE_URL=postgresql://user:password@localhost:5432/ai_tutor
REDIS_URL=redis://localhost:6379
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east-1-aws
```

### Day 2-3: Infrastructure Setup

**Option A: Local Development (Quickest)**

```bash
# Use docker-compose for local development
cd infrastructure/docker

# Create docker-compose.yml (see template below)
docker-compose up -d

# This starts:
# - PostgreSQL (port 5432)
# - Redis (port 6379)
# - pgAdmin (port 5050)
```

**Option B: AWS Cloud (Production-like)**

```bash
cd infrastructure/terraform

# Initialize Terraform
terraform init

# Plan infrastructure
terraform plan -out=tfplan

# Apply (creates: RDS, ElastiCache, ECS, etc.)
terraform apply tfplan
```

### Day 4-5: Core Service Development

Start with the **Tutor Service** - the heart of the system.

## Week 2-3: MVP Development

### Phase 1: Tutor Service (Core AI Logic)

**Install dependencies:**
```bash
cd backend/tutor_service
pip install -r requirements.txt
```

**requirements.txt:**
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
langchain==0.1.0
langchain-openai==0.0.5
langchain-anthropic==0.0.1
langchain-pinecone==0.0.1
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0
redis==5.0.1
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
alembic==1.13.0
```

**Create basic API structure:** (see `backend/tutor_service/main.py` template)

### Phase 2: Knowledge Base Setup

**Step 1: Prepare Course Materials**

```bash
# Create course content directory
mkdir -p data/course_materials

# Structure:
# data/course_materials/
#   ├── module_1_ai_strategy_fundamentals/
#   │   ├── content.md
#   │   ├── slides.pdf
#   │   └── readings.json
#   ├── module_2_organizational_adoption/
#   └── ...
```

**Step 2: Generate Embeddings**

```python
# scripts/ingest_course_materials.py
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import pinecone

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))

# Load documents
loader = DirectoryLoader('data/course_materials', glob="**/*.md")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# Generate embeddings and store
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore.from_documents(
    chunks,
    embeddings,
    index_name="ai-tutor-course-content"
)

print(f"✓ Ingested {len(chunks)} chunks into Pinecone")
```

Run ingestion:
```bash
python scripts/ingest_course_materials.py
```

### Phase 3: Implement RAG Chain

```python
# backend/tutor_service/chains/concept_explanation.py
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def create_concept_explanation_chain():
    """Create RAG chain for concept explanations"""

    # Initialize components
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.7)
    vectorstore = PineconeVectorStore(index_name="ai-tutor-course-content")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Prompt template
    template = """You are an expert AI Strategy tutor for graduate students.

Relevant Course Materials:
{context}

Student Question: {question}

Student Background:
- Completed modules: {completed_modules}
- Current level: {student_level}

Provide a clear, graduate-level explanation with:
1. Concept explanation (200-300 words)
2. Real-world business example (Google, Amazon, etc.)
3. 2-3 follow-up questions to check understanding

Format:
**Explanation:**
[Your explanation]

**Example:**
[Business example]

**Follow-up Questions:**
1. [Question]
2. [Question]
"""

    prompt = ChatPromptTemplate.from_template(template)

    # Create chain
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "completed_modules": lambda x: x.get("completed_modules", []),
            "student_level": lambda x: x.get("student_level", "Intermediate")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
```

### Phase 4: API Endpoints

```python
# backend/tutor_service/routers/chat.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from chains.concept_explanation import create_concept_explanation_chain

router = APIRouter(prefix="/api/v1/tutor", tags=["tutor"])

class ChatRequest(BaseModel):
    question: str
    student_id: str
    context: dict = {}

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    follow_up_questions: list[str]

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handle student question and return AI tutor response
    """
    try:
        # Get student profile (mock for now)
        student_profile = {
            "completed_modules": ["module_1", "module_2"],
            "student_level": "Intermediate"
        }

        # Create chain
        chain = create_concept_explanation_chain()

        # Get response
        response = await chain.ainvoke({
            "question": request.question,
            **student_profile
        })

        # Parse response (simplified)
        return ChatResponse(
            answer=response,
            sources=["Module 1: AI Strategy Fundamentals"],
            confidence=0.85,
            follow_up_questions=[
                "How would you apply this in your organization?",
                "What challenges might you face?"
            ]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Phase 5: Frontend (Simple Chat UI)

```bash
# Create React app
cd frontend
npx create-react-app . --template typescript
npm install @tailwindcss/forms axios
```

**Simple Chat Component:**
```typescript
// frontend/src/components/ChatInterface.tsx
import React, { useState } from 'react';
import axios from 'axios';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages([...messages, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/api/v1/tutor/chat', {
        question: input,
        student_id: 'student_123',
        context: {}
      });

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.data.answer
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-4">
      <div className="bg-white rounded-lg shadow-lg p-6 mb-4 h-96 overflow-y-auto">
        {messages.map((msg, idx) => (
          <div key={idx} className={`mb-4 ${msg.role === 'user' ? 'text-right' : 'text-left'}`}>
            <div className={`inline-block p-3 rounded-lg ${
              msg.role === 'user'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-800'
            }`}>
              {msg.content}
            </div>
          </div>
        ))}
        {loading && <div className="text-center text-gray-500">Thinking...</div>}
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Ask a question about AI Strategy..."
          className="flex-1 p-3 border rounded-lg"
        />
        <button
          onClick={sendMessage}
          disabled={loading}
          className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </div>
  );
};
```

## Week 4: Testing & Deployment

### Testing the System

**Test 1: Basic Q&A**
```bash
# Start backend
cd backend/tutor_service
uvicorn main:app --reload

# Test endpoint
curl -X POST http://localhost:8000/api/v1/tutor/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is AI-first strategy?",
    "student_id": "test_student",
    "context": {}
  }'
```

**Test 2: RAG Quality**
```python
# scripts/test_rag.py
from chains.concept_explanation import create_concept_explanation_chain

chain = create_concept_explanation_chain()

test_questions = [
    "What is AI-first strategy?",
    "How do companies achieve competitive advantage through AI?",
    "What are the key challenges in AI adoption?"
]

for question in test_questions:
    print(f"\nQ: {question}")
    response = chain.invoke({"question": question})
    print(f"A: {response}\n")
    print("-" * 80)
```

### Deploy to AWS

**Quick Deploy with Docker:**

```bash
# Build and push Docker image
docker build -t ai-tutor-backend:latest .
docker tag ai-tutor-backend:latest <your-ecr-repo>/ai-tutor-backend:latest
docker push <your-ecr-repo>/ai-tutor-backend:latest

# Deploy to ECS (using Terraform)
cd infrastructure/terraform
terraform apply -var="image_tag=latest"
```

## Monitoring & Maintenance

### Set up monitoring:

```python
# Add to main.py
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Metrics available at /metrics
```

### Cost Tracking:

```python
# backend/tutor_service/middleware/cost_tracking.py
from functools import wraps
import redis

redis_client = redis.Redis.from_url(os.getenv("REDIS_URL"))

def track_llm_cost(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        response = await func(*args, **kwargs)

        # Estimate cost (simplified)
        tokens = len(response) / 4  # rough estimate
        cost = tokens * 0.00003  # GPT-4 pricing

        # Store in Redis
        redis_client.incrbyfloat("total_llm_cost", cost)

        return response
    return wrapper
```

## Success Checklist

After 4 weeks, you should have:

- [ ] Backend API running locally
- [ ] Course materials ingested into Pinecone
- [ ] RAG chain returning accurate answers
- [ ] Simple frontend chat interface
- [ ] Student can ask questions and get responses < 3 seconds
- [ ] Responses reference course materials
- [ ] Basic analytics (request count, response times)
- [ ] Deployed to AWS (staging environment)

## Next Steps

**Week 5-8: Add Features**
- [ ] Student profile service (track progress)
- [ ] Case study analysis chain
- [ ] Instructor analytics dashboard
- [ ] LMS integration (Canvas LTI 1.3)

**Week 9-12: Pilot**
- [ ] Onboard 50 pilot students
- [ ] Collect feedback
- [ ] Iterate on prompts based on instructor review
- [ ] Measure success metrics

## Common Issues & Solutions

**Issue: Slow response times (>5 seconds)**
- Solution: Implement caching with Redis for common questions
- Solution: Use GPT-3.5 for simple queries

**Issue: Inaccurate answers**
- Solution: Improve chunking strategy for course materials
- Solution: Add confidence threshold (reject low-confidence responses)

**Issue: High LLM costs**
- Solution: Cache FAQ responses for 24 hours
- Solution: Optimize prompts to reduce token usage
- Solution: Route simple questions to cheaper models

**Issue: RAG not finding relevant content**
- Solution: Improve document chunking (smaller chunks, more overlap)
- Solution: Add metadata filters (module number, topic)
- Solution: Experiment with different embedding models

## Resources

- **LangChain Docs**: https://python.langchain.com/docs/get_started/introduction
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Pinecone Docs**: https://docs.pinecone.io/
- **OpenAI API Docs**: https://platform.openai.com/docs/api-reference

## Support

For issues or questions:
1. Check the detailed specifications in `/docs`
2. Review architecture diagrams in `AI_TUTOR_SPECIFICATION.md`
3. Run multi-agent workflow for clarifications: `python agents/examples/ai_tutor_workflow.py`

---

**Ready to build!** Start with Week 1 setup and work through each phase. The MVP can be completed in 4 weeks with a small team.
