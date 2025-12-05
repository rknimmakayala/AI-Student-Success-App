# 🎓 Enrollment FAQ Chatbot - Complete Demo Guide

**Everything You Need to See the Chatbot in Action**

---

## 📦 What You Have

All artifacts for a working enrollment FAQ chatbot with Human-in-the-Loop:

### 1. **Working Backend Code** (`enrollment_chatbot/main.py` - 650+ lines)
- Complete FastAPI application
- 8 pre-loaded FAQs with intelligent matching
- Automatic escalation logic
- Advisor queue management
- Real-time statistics
- **Ready to run immediately**

### 2. **Demo Script** (`enrollment_chatbot/demo.py`)
- 5 test scenarios
- Interactive walkthrough
- Shows bot + HITL in action
- Automated testing

### 3. **Complete Specifications**
- `ENROLLMENT_CHATBOT_SUMMARY.md` - Full specification
- `enrollment_chatbot_pm_requirements.json` - Detailed requirements
- `enrollment_chatbot_workflow.json` - Workflow state

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
cd enrollment_chatbot

# Install required packages
pip install fastapi uvicorn pydantic requests

# Or use requirements file
pip install -r requirements.txt
```

### Step 2: Start the Chatbot API

```bash
python main.py
```

**You'll see:**
```
🚀 Starting Enrollment FAQ Chatbot with HITL...
✓ Loaded 8 FAQs
✓ Services initialized
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Run the Demo (In New Terminal)

```bash
# In a new terminal
cd enrollment_chatbot
python demo.py
```

**Follow the interactive prompts to see:**
- Bot answering simple FAQs
- Escalation to human advisors
- Context preservation
- Statistics and analytics

---

## 🎯 What the Demo Shows

### Scenario 1: Simple FAQ (Bot Handles) ✅

**Student**: "What is competency-based education?"

**Bot Response** (confidence: 95%):
```
Great question! Competency-based education (CBE) is a flexible,
self-paced approach where you progress by demonstrating mastery
of skills, not by spending time in class.

Key differences from traditional college:
• Self-paced: Move as fast or slow as you need
• No semesters: Start monthly, work on your own schedule
• Prior knowledge counts: If you already know material, prove it
• Subscription model: Pay per 6-month term
...
```

**Result**: Bot handles completely, no human needed

---

### Scenario 2: Tuition Cost (Bot Handles) ✅

**Student**: "How much does it cost?"

**Bot Response** (confidence: 95%):
```
Our tuition structure is designed to be affordable:

Subscription Model:
• $3,000 per 6-month term (undergraduate)
• $4,000 per 6-month term (graduate)
• Take as many courses as you can complete
• No per-credit fees

Additional costs:
• $50 application fee (often waived)
• ~$200-300 for course materials per term
...
```

**Result**: Clear pricing provided, student informed

---

### Scenario 3: Complex Transfer Credits (Escalation) ⚠️

**Student**: "Will my credits transfer?"

**Bot Response** (confidence: 85%):
```
Good question! Credit transfer depends on several factors:

Generally transferable:
• Regionally accredited colleges/universities
• Grades of C or better
• Courses relevant to your program
...
```

**Student**: "I have credits from 3 different colleges and military training from 8 years ago"

**Bot Response** (confidence: 65%):
```
I want to make sure you get the best answer. Let me connect you
with an enrollment advisor who can help with your specific situation.

They'll see our conversation so you won't need to repeat anything.
```

**🚨 ESCALATION TRIGGERED**: Complex situation requiring personalized guidance

**Result**: Bot recognizes complexity, escalates to human

---

### Scenario 4: Explicit Human Request (Immediate Escalation) ⚠️

**Student**: "Can I speak to a real person please?"

**Bot Response** (confidence: 50%):
```
I want to make sure you get the best answer. Let me connect you
with an enrollment advisor who can help with your specific question.
```

**🚨 ESCALATION TRIGGERED**: Student requested human advisor

**Result**: Immediate handoff, no questions asked

---

### Scenario 5: Military Benefits (Bot Handles) ✅

**Student**: "Do you accept GI Bill for veterans?"

**Bot Response** (confidence: 88%):
```
Yes! We proudly support military and veterans:

Accepted Benefits:
• GI Bill® (Chapters 30, 33, 1606)
• Vocational Rehabilitation (Chapter 31)
• Tuition Assistance (TA)
• Yellow Ribbon Program participant

Military Credit:
• Joint Services Transcript (JST) evaluated for credit
• ACE-recommended military training accepted
• Many veterans transfer 20-40 credits
...
```

**Result**: Military-specific information provided

---

## 🔍 Testing the API Manually

### Using cURL (Command Line)

**1. Health Check**
```bash
curl http://localhost:8000/health | jq '.'
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-12-05T15:45:23",
  "conversations_active": 5,
  "queue_length": 2
}
```

**2. Ask a Question**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "test_student_123",
    "message": "What is competency-based education?"
  }' | jq '.'
```

**Response:**
```json
{
  "message": "Great question! Competency-based education...",
  "role": "bot",
  "confidence": 0.95,
  "escalation_suggested": false,
  "escalation_reason": null,
  "follow_up_questions": [
    "Want to learn how assessments work?",
    "Curious about how fast you can complete?"
  ],
  "sources": ["competency_model"]
}
```

**3. Get Statistics**
```bash
curl http://localhost:8000/api/v1/stats | jq '.'
```

**Response:**
```json
{
  "total_conversations": 10,
  "bot_handled": 7,
  "escalated": 3,
  "resolved": 0,
  "current_queue_length": 3,
  "bot_resolution_rate": 70.0
}
```

**4. View Advisor Queue**
```bash
curl http://localhost:8000/api/v1/advisor/queue | jq '.'
```

**Response:**
```json
{
  "queue": [
    {
      "conversation_id": "abc-123",
      "student_id": "student_456",
      "channel": "web",
      "escalation_reason": "Complex situation requiring personalized guidance",
      "messages_count": 4,
      "escalated_at": "2024-12-05T15:40:12",
      "wait_time_minutes": 5
    }
  ],
  "count": 1
}
```

**5. Get Conversation for Advisor Handoff**
```bash
curl http://localhost:8000/api/v1/advisor/conversation/abc-123 | jq '.'
```

**Response:**
```json
{
  "conversation_id": "abc-123",
  "student_profile": {
    "student_id": "student_456",
    "created_at": "2024-12-05T15:35:00"
  },
  "messages": [
    {
      "role": "user",
      "content": "Will my credits transfer?",
      "timestamp": "2024-12-05T15:35:10",
      "confidence": null
    },
    {
      "role": "bot",
      "content": "Good question! Credit transfer depends...",
      "timestamp": "2024-12-05T15:35:11",
      "confidence": 0.85
    },
    {
      "role": "user",
      "content": "I have credits from 3 different colleges",
      "timestamp": "2024-12-05T15:38:45",
      "confidence": null
    }
  ],
  "bot_notes": {
    "questions_asked": 2,
    "topics_discussed": ["transfer_credits"],
    "escalation_reason": "Complex situation",
    "recommended_actions": [
      "Discuss transcript evaluation process",
      "Mention Prior Learning Assessment (PLA)"
    ]
  },
  "status": "escalated"
}
```

---

## 🌐 Using the Interactive API Docs

**Open in browser:** http://localhost:8000/docs

You'll see **Swagger UI** with:
- All endpoints documented
- "Try it out" buttons
- Request/response examples
- Live testing

**Try it:**
1. Click on **POST /api/v1/chat**
2. Click **"Try it out"**
3. Edit the request body:
```json
{
  "student_id": "web_test_user",
  "message": "How much does it cost?",
  "student_name": "John Doe",
  "channel": "web"
}
```
4. Click **"Execute"**
5. See the response!

---

## 📊 Understanding the Responses

### Bot Response Structure

```json
{
  "message": "The actual response text",
  "role": "bot",
  "confidence": 0.95,        // 0.0-1.0 (1.0 = 100% confident)
  "escalation_suggested": false,
  "escalation_reason": null,
  "follow_up_questions": [...],
  "sources": ["category"]
}
```

### Confidence Levels

- **0.90-1.00**: Very confident, simple FAQ
- **0.70-0.89**: Confident, may suggest human for follow-up
- **< 0.70**: Low confidence, **automatic escalation**

### Escalation Triggers

Bot automatically escalates when:
- Confidence < 70%
- Student explicitly requests human
- Student shows frustration
- Complex situation detected (e.g., "3 colleges", "multiple schools")
- 3+ follow-up questions on same topic

---

## 🎨 Customizing the Chatbot

### Add New FAQs

Edit `main.py`, find `_load_faq_database()` method:

```python
"new_faq_id": {
    "question": "Your question here",
    "keywords": ["keyword1", "keyword2", "phrase"],
    "response": """Your detailed response here.

Can use **markdown** formatting!

Bullet points:
• Point 1
• Point 2
    """,
    "confidence": 0.90,
    "category": "your_category"
}
```

### Adjust Escalation Criteria

Edit `FAQMatcher.should_escalate()` method:

```python
# Change confidence threshold
if confidence < 0.70:  # Change to 0.60 for more bot handling
    return True, "Low confidence in answer"

# Add new escalation keywords
complex_keywords = ["multiple schools", "3 colleges", "YOUR NEW KEYWORD"]
```

### Change Follow-up Questions

Edit the `chat()` endpoint:

```python
if category == "your_category":
    follow_ups = [
        "Your follow-up question 1?",
        "Your follow-up question 2?",
        "Your follow-up question 3?"
    ]
```

---

## 📈 Monitoring and Analytics

### Real-Time Statistics

**Endpoint:** `GET /api/v1/stats`

**Metrics tracked:**
- Total conversations
- Bot resolution rate
- Escalation count
- Current queue length

**Example:**
```bash
watch -n 5 'curl -s http://localhost:8000/api/v1/stats | jq "."'
```
Updates every 5 seconds

### Advisor Queue Management

**Endpoint:** `GET /api/v1/advisor/queue`

**Shows:**
- Waiting conversations
- Escalation reasons
- Wait times
- Student info

---

## 🧪 Test Scenarios to Try

### Test 1: Happy Path (Bot Success)
```
Student: "What programs do you offer?"
Expected: Bot lists programs, no escalation
```

### Test 2: Price Inquiry
```
Student: "How expensive is this?"
Expected: Bot provides tuition breakdown
```

### Test 3: Working Full-Time
```
Student: "Can I work and attend?"
Expected: Bot explains flexibility
```

### Test 4: Transfer Credits (Simple)
```
Student: "Do you accept transfer credits?"
Expected: Bot provides general info
```

### Test 5: Transfer Credits (Complex)
```
Student: "I have credits from 3 different community colleges from 10 years ago, plus military training"
Expected: Bot escalates due to complexity
```

### Test 6: Frustration
```
Student: "This doesn't answer my question. I need to talk to someone"
Expected: Immediate escalation
```

### Test 7: Military Benefits
```
Student: "Do you take GI Bill?"
Expected: Bot provides military benefits info
```

### Test 8: Unknown Question
```
Student: "What's the weather like on campus?"
Expected: Bot escalates (low confidence, irrelevant)
```

---

## 🔧 Troubleshooting

### API Won't Start

**Problem:** Port 8000 already in use

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or use different port
uvicorn main:app --port 8001
```

### Dependencies Missing

**Problem:** `ModuleNotFoundError`

**Solution:**
```bash
pip install fastapi uvicorn pydantic requests
```

### Demo Script Can't Connect

**Problem:** `Connection refused`

**Solution:**
1. Make sure API is running (`python main.py`)
2. Check it's on port 8000
3. Visit http://localhost:8000/health in browser

### No Escalations Happening

**Problem:** Bot isn't escalating when it should

**Solution:** Check escalation criteria in code:
- Confidence threshold (default: 0.70)
- Keyword matches
- Question count

---

## 📁 File Structure

```
enrollment_chatbot/
├── main.py                 # Backend API (650+ lines)
├── demo.py                 # Interactive demo script
├── requirements.txt        # Dependencies
└── README.md              # This file

agents/examples/
├── enrollment_chatbot_pm_requirements.json    # Full requirements
├── enrollment_chatbot_workflow.json          # Workflow state
└── enrollment_chatbot_workflow.py            # Workflow generator

ENROLLMENT_CHATBOT_SUMMARY.md     # Non-technical specification
```

---

## 🎯 API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/api/v1/chat` | POST | Send message, get response |
| `/api/v1/escalate` | POST | Manually escalate conversation |
| `/api/v1/advisor/queue` | GET | View escalation queue |
| `/api/v1/advisor/conversation/{id}` | GET | Get full conversation for advisor |
| `/api/v1/advisor/claim/{id}` | POST | Advisor claims conversation |
| `/api/v1/stats` | GET | Get statistics |

---

## ✨ Features Demonstrated

### ✅ Automated FAQ Handling
- 8 pre-loaded FAQs
- Keyword matching
- Confidence scoring
- Follow-up suggestions

### ✅ Human-in-the-Loop
- Automatic escalation criteria
- Queue management
- Context preservation
- Advisor recommendations

### ✅ Conversation Management
- Multi-turn conversations
- Message history
- Student profiles
- Topic tracking

### ✅ Analytics
- Resolution rates
- Queue lengths
- Performance metrics
- Real-time statistics

---

## 🚀 Next Steps

### To Enhance:
1. **Add more FAQs** - Currently 8, target 70+
2. **Integrate CRM** - Connect to Salesforce/HubSpot
3. **Add channels** - SMS, WhatsApp, Facebook
4. **Build advisor dashboard** - Web UI for advisors
5. **Add AI/LLM** - Use GPT-4 for natural language understanding

### To Deploy:
1. **Add database** - PostgreSQL for persistence
2. **Add Redis** - For session management
3. **Add monitoring** - Datadog, CloudWatch
4. **Containerize** - Docker + Kubernetes
5. **Add CI/CD** - GitHub Actions

---

## 💡 Tips for Demo

1. **Run scenarios in order** - Shows progression from simple to complex
2. **Point out confidence scores** - Show how bot knows its limits
3. **Highlight escalation** - Key differentiator vs. basic chatbots
4. **Show advisor view** - Full context, no repeating questions
5. **Check statistics** - Demonstrate 70%+ resolution rate

---

## 🎉 Summary

You now have a **complete, working chatbot** with:

✅ 8 pre-loaded FAQs covering key enrollment topics
✅ Intelligent escalation logic (confidence + keywords + context)
✅ Advisor queue management
✅ Context preservation for handoff
✅ Real-time statistics and monitoring
✅ Complete API documentation
✅ Interactive demo script

**Everything runs locally, no external dependencies needed!**

---

**Ready to demo?** 🚀

1. `python main.py` - Start the API
2. `python demo.py` - Run the demo
3. Watch the magic happen!

*All code is in `/enrollment_chatbot/` and ready to run.*
