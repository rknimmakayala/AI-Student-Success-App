# Quick Start Guide: Enrollment FAQ Chatbot
## Get Up and Running in 5 Minutes

---

## Prerequisites

You need:
- ✅ Python 3.8 or higher installed
- ✅ A web browser (Chrome, Firefox, Safari, Edge)
- ✅ Terminal/Command Prompt access

Check Python version:
```bash
python --version
# or
python3 --version
```

---

## Step 1: Install Dependencies (1 minute)

Navigate to the project directory and install requirements:

```bash
cd enrollment_chatbot
pip install -r requirements.txt
```

You should see:
```
Successfully installed fastapi-0.104.1 uvicorn-0.24.0 pydantic-2.5.0 ...
```

---

## Step 2: Start the Server (30 seconds)

Run the main application:

```bash
python main.py
```

Wait for this message:
```
🚀 Starting Enrollment FAQ Chatbot with HITL...
✓ Loaded 8 FAQs
✓ Services initialized
Uvicorn running on http://0.0.0.0:8000
```

**That's it! The server is running.**

---

## Step 3: Open the Prototype (30 seconds)

Open your web browser and go to:

```
http://localhost:8000
```

You'll see the main landing page with three options:
1. **Student Chat** - Try the chatbot as a student
2. **Advisor Dashboard** - See the advisor interface
3. **API Documentation** - View technical docs

---

## Step 4: Try the Student Chat (2 minutes)

### Click "Student Chat" or go to:
```
http://localhost:8000/static/student_chat.html
```

### Test these scenarios:

**Scenario A: Simple Question (Bot Handles It)**
1. Click the "What is CBE?" quick button
2. Watch the bot respond with 100% confidence
3. Notice no escalation needed

**Scenario B: Complex Question (Auto-Escalation)**
1. Type: "Will my credits transfer?"
2. Bot provides general info
3. Type: "I have credits from 3 different colleges"
4. Watch the escalation notice appear!
5. Bot recognizes it needs a human

**Scenario C: Request Human**
1. Type: "Can I speak to a real person?"
2. Immediate escalation - no questions asked

---

## Step 5: Check the Advisor Dashboard (2 minutes)

### Open in a new tab:
```
http://localhost:8000/static/advisor_dashboard.html
```

You'll see:
- 📊 **Statistics** at the top (queue length, bot resolution rate)
- 📋 **Escalated conversations** from your testing
- 🔍 **View Full Conversation** buttons

### Try this:
1. Click "View Full Conversation" on any escalated item
2. See the complete message history
3. Notice bot notes and recommended actions
4. See why it was escalated

**This is what advisors see when taking over from the bot!**

---

## Common Commands

### Start the server:
```bash
cd enrollment_chatbot
python main.py
```

### Stop the server:
- Press `Ctrl+C` in the terminal

### Restart the server (fresh data):
1. Stop it with `Ctrl+C`
2. Run `python main.py` again

### View API documentation:
```
http://localhost:8000/docs
```

---

## Quick Troubleshooting

### Problem: "Port already in use"
**Solution:** Another process is using port 8000.

Option 1 - Find and kill the process:
```bash
# On Mac/Linux:
lsof -ti:8000 | xargs kill -9

# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F
```

Option 2 - Use a different port:
Edit `main.py` line 720 to use port 8001:
```python
uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
```

### Problem: "Module not found"
**Solution:** Dependencies not installed.
```bash
cd enrollment_chatbot
pip install -r requirements.txt
```

### Problem: Page shows "Unable to connect"
**Solution:** Make sure the server is running.
- Check the terminal for the "Uvicorn running" message
- Try http://localhost:8000/health to check server status

### Problem: Empty advisor queue
**Solution:** Create some test conversations in the student chat first!

---

## File Structure

Understanding what's what:

```
enrollment_chatbot/
│
├── main.py                     # The backend API server (start this!)
├── demo.py                     # Automated demo script
├── requirements.txt            # Python dependencies
│
└── static/                     # Web interfaces
    ├── index.html             # Main landing page
    ├── student_chat.html      # Student chat interface
    └── advisor_dashboard.html # Advisor queue management
```

---

## URLs at a Glance

Once the server is running:

| Interface | URL |
|-----------|-----|
| **Main Page** | http://localhost:8000 |
| **Student Chat** | http://localhost:8000/static/student_chat.html |
| **Advisor Dashboard** | http://localhost:8000/static/advisor_dashboard.html |
| **API Docs** | http://localhost:8000/docs |
| **Health Check** | http://localhost:8000/health |
| **Statistics** | http://localhost:8000/api/v1/stats |

---

## Testing the API Directly

Want to test the API with command line?

### Send a chat message:
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "test_123",
    "message": "What is competency-based education?"
  }'
```

### Check the advisor queue:
```bash
curl http://localhost:8000/api/v1/advisor/queue
```

### Get statistics:
```bash
curl http://localhost:8000/api/v1/stats
```

---

## Demo Tips

### For a Live Demo to Stakeholders:

**Before they arrive:**
1. Start the server
2. Open all three tabs in your browser
3. Run a few test conversations to populate the queue
4. Keep the terminal visible (shows real-time activity)

**During the demo:**
1. Show the student chat first (most relatable)
2. Create an escalation while they watch
3. Switch to advisor dashboard
4. Show the escalated conversation with full context
5. Emphasize "no information loss"

**What to highlight:**
- ⚡ Instant responses for simple questions
- 🎯 Intelligent escalation for complex cases
- 📋 Complete context preservation
- 📊 Real-time statistics and monitoring

---

## Next Steps

### Want to customize it?

**Add a new FAQ:**
1. Open `main.py`
2. Find the `_load_faq_database()` function (around line 115)
3. Add a new FAQ following the existing pattern
4. Restart the server

**Change the confidence threshold:**
1. Open `main.py`
2. Find line 464: `if confidence < 0.70:`
3. Change 0.70 to your preferred threshold
4. Restart the server

**Modify the UI:**
1. Edit the HTML files in `enrollment_chatbot/static/`
2. Refresh your browser (no restart needed)

---

## Getting Help

### Documentation Available:
- `VIDEO_WALKTHROUGH_SCRIPT.md` - Detailed demo script
- `STAKEHOLDER_PRESENTATION.md` - Full presentation slides
- `ENROLLMENT_CHATBOT_DEMO_GUIDE.md` - Technical guide
- `ENROLLMENT_CHATBOT_SUMMARY.md` - Non-technical overview

### Check Server Logs:
The terminal where you ran `python main.py` shows all activity:
- Requests received
- Conversations created
- Escalations triggered

### Test Server Health:
```bash
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "conversations_active": 0,
  "queue_length": 0
}
```

---

## Production Deployment (Future)

When you're ready to deploy for real students:

1. **Expand FAQs** - Add 70+ questions
2. **Integrate CRM** - Connect to Salesforce/HubSpot
3. **Add Authentication** - Secure the advisor dashboard
4. **Deploy to Cloud** - AWS, Azure, or Google Cloud
5. **Add Monitoring** - Error tracking and analytics
6. **Enable HTTPS** - Secure communications
7. **Add Channels** - SMS, WhatsApp, Facebook

But for now, **just run it locally and impress your stakeholders!**

---

## Cheat Sheet

| Task | Command |
|------|---------|
| Start server | `cd enrollment_chatbot && python main.py` |
| Stop server | `Ctrl+C` |
| Open main page | http://localhost:8000 |
| Open student chat | http://localhost:8000/static/student_chat.html |
| Open advisor dashboard | http://localhost:8000/static/advisor_dashboard.html |
| Check health | http://localhost:8000/health |
| View API docs | http://localhost:8000/docs |
| Get statistics | http://localhost:8000/api/v1/stats |

---

## Success Checklist

Before your demo, verify:

- [ ] Server starts without errors
- [ ] Main page loads (http://localhost:8000)
- [ ] Student chat interface works
- [ ] Advisor dashboard loads
- [ ] Can send messages and get responses
- [ ] Escalation triggers appear
- [ ] Advisor queue shows escalated conversations
- [ ] "View Full Conversation" opens correctly
- [ ] Statistics are displayed
- [ ] Browser zoom is at 100%
- [ ] Notifications are disabled (DND mode)

**You're ready to demo!**

---

## Quick Demo Script (2 Minutes)

If you need to show it fast:

1. **Open student chat** (30 sec)
   - "This is what students see"
   - Ask: "What is competency-based education?"
   - Show instant response

2. **Trigger escalation** (30 sec)
   - Ask: "I have credits from 3 colleges"
   - Show escalation notice
   - "Bot recognizes it needs human help"

3. **Show advisor dashboard** (1 min)
   - "This is what advisors see"
   - Click "View Full Conversation"
   - "Complete context - no repeating information"
   - "This is Human-in-the-Loop"

**Done! Questions?**

---

**Remember:** This is a fully functional prototype. Everything you see actually works!

**Have fun demoing!** 🎉
