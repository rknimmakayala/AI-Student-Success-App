# Streamlit Deployment Guide
## Deploy Enrollment FAQ Chatbot to Streamlit Cloud

---

## 🚀 Quick Start - Run Locally

### Step 1: Start the FastAPI Backend

The Streamlit app needs the FastAPI backend to be running.

```bash
# Terminal 1 - Start the API
cd enrollment_chatbot
python main.py
```

Wait for:
```
✓ Loaded 8 FAQs
✓ Services initialized
Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Start the Streamlit App

```bash
# Terminal 2 - Start Streamlit
streamlit run streamlit_app.py
```

The app will open in your browser at: `http://localhost:8501`

---

## 📦 What's Included

### File Structure:
```
AI-Student-Success-App/
├── streamlit_app.py              # Main Streamlit app (navigation)
├── pages/
│   ├── student_chat.py          # Student chat interface
│   └── advisor_dashboard.py     # Advisor dashboard
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── requirements-streamlit.txt    # Streamlit dependencies
└── enrollment_chatbot/
    └── main.py                   # FastAPI backend (must be running)
```

### Features:
✅ **Student Chat Page** - Beautiful chat interface with quick questions
✅ **Advisor Dashboard Page** - Queue management and conversation viewer
✅ **Home Page** - Navigation and feature overview
✅ **Real-time updates** - Auto-refresh option for advisor dashboard
✅ **Responsive design** - Works on mobile and desktop

---

## ☁️ Deploy to Streamlit Cloud (Free)

### Option 1: Streamlit Frontend Only (Recommended for Demo)

**Deploy just the Streamlit app and run the API locally:**

1. **Push your code to GitHub** (already done!)

2. **Go to Streamlit Cloud:**
   - Visit: https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"

3. **Configure deployment:**
   - Repository: `rknimmakayala/AI-Student-Success-App`
   - Branch: `main` (or your branch name)
   - Main file path: `streamlit_app.py`
   - Advanced settings → Python version: `3.11`

4. **Set environment variable:**
   - In "Advanced settings" → "Secrets"
   - Add:
   ```toml
   API_BASE_URL = "http://YOUR-NGROK-URL"
   ```
   - See "Expose Local API" section below

5. **Click "Deploy"**

6. **Your app will be live at:**
   ```
   https://share.streamlit.io/rknimmakayala/ai-student-success-app/main/streamlit_app.py
   ```

### Option 2: Full Deployment (Frontend + Backend)

**Deploy both Streamlit frontend and FastAPI backend:**

#### A. Deploy FastAPI Backend First

Choose one platform for the backend:

**Option A1: Railway.app (Easiest)**

1. Go to: https://railway.app/
2. Click "Start a New Project" → "Deploy from GitHub"
3. Select your repository
4. Railway auto-detects FastAPI
5. Set start command: `cd enrollment_chatbot && python main.py`
6. Note your Railway URL: `https://your-app.railway.app`

**Option A2: Render.com (Free Tier)**

1. Go to: https://render.com/
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - Build Command: `pip install -r enrollment_chatbot/requirements.txt`
   - Start Command: `cd enrollment_chatbot && uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Note your Render URL: `https://your-app.onrender.com`

**Option A3: Heroku**

```bash
# Install Heroku CLI, then:
cd enrollment_chatbot
heroku create your-chatbot-api
git push heroku main
```

#### B. Deploy Streamlit Frontend

1. Follow "Option 1" steps above
2. In step 4, set `API_BASE_URL` to your backend URL:
   ```toml
   API_BASE_URL = "https://your-app.railway.app"
   ```

---

## 🔌 Expose Local API (for Testing)

If you want to test with Streamlit Cloud but keep the API running locally:

### Using ngrok (Free):

1. **Install ngrok:**
   ```bash
   # Mac
   brew install ngrok

   # Windows
   choco install ngrok

   # Or download from: https://ngrok.com/download
   ```

2. **Start your FastAPI backend:**
   ```bash
   cd enrollment_chatbot
   python main.py
   ```

3. **Expose it with ngrok:**
   ```bash
   ngrok http 8000
   ```

4. **Copy the ngrok URL:**
   ```
   Forwarding https://abc123.ngrok.io -> http://localhost:8000
   ```

5. **Update Streamlit secrets:**
   - In Streamlit Cloud settings → Secrets
   - Set:
   ```toml
   API_BASE_URL = "https://abc123.ngrok.io"
   ```

6. **Redeploy Streamlit app**

**Note:** ngrok URLs expire and change each time you restart. For permanent deployment, use Railway/Render.

---

## 🧪 Testing the Deployment

### Test Student Chat:
1. Navigate to "Student Chat 💬" in sidebar
2. Try a simple question: "What is competency-based education?"
3. Try a complex question: "I have credits from 3 colleges"
4. Watch for escalation notice

### Test Advisor Dashboard:
1. Navigate to "Advisor Dashboard 👨‍💼"
2. Check that statistics appear
3. Create escalations in Student Chat
4. Refresh Advisor Dashboard
5. Click "View Full Conversation"

### Test API Connection:
- Look for "✅ Chatbot API Online" in sidebar
- If offline, check your API URL and backend status

---

## 🐛 Troubleshooting

### "Cannot connect to chatbot API"

**Check 1: Is the backend running?**
```bash
curl http://localhost:8000/health
# Should return: {"status": "healthy", ...}
```

**Check 2: Is API_BASE_URL correct?**
- For local: `http://localhost:8000`
- For deployed: `https://your-backend-url.com`
- Check Streamlit secrets/environment variables

**Check 3: CORS issues?**
- The FastAPI backend already has CORS enabled
- Check browser console for errors

### "Streamlit app won't start"

**Check dependencies:**
```bash
pip install -r requirements-streamlit.txt
```

**Check Python version:**
```bash
python --version  # Should be 3.8+
```

### "Pages don't load"

**Check directory structure:**
```bash
ls pages/
# Should show: student_chat.py advisor_dashboard.py
```

### "Deployment failed on Streamlit Cloud"

**Check requirements file:**
- Make sure `requirements-streamlit.txt` exists
- Or rename it to `requirements.txt` for Streamlit Cloud

**Check file paths:**
- All paths should be relative, not absolute
- Use forward slashes (/) not backslashes (\)

---

## 🔐 Security Notes for Production

When deploying to production:

1. **Use environment variables for API URL:**
   ```python
   API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
   ```

2. **Add authentication:**
   - Use Streamlit's built-in authentication
   - Or add password protection:
   ```python
   import streamlit_authenticator as stauth
   ```

3. **Secure your API:**
   - Add API keys
   - Use HTTPS only
   - Enable rate limiting

4. **Don't expose sensitive data:**
   - Don't show student PII in URLs
   - Use session state, not query parameters

---

## 📊 Streamlit vs HTML Interfaces

### Streamlit Advantages:
✅ Easier deployment (Streamlit Cloud is free)
✅ Python-based (no JavaScript needed)
✅ Built-in state management
✅ Auto-refresh capabilities
✅ Mobile-responsive out of the box

### HTML Advantages:
✅ More customizable design
✅ Faster page loads
✅ Works without Python runtime
✅ Can be hosted anywhere (static site)

**Recommendation:** Use Streamlit for quick demos and internal tools, use HTML for customer-facing production apps.

---

## 🎯 Next Steps

### After Deployment:

1. **Share the URL** with stakeholders
2. **Monitor usage** via Streamlit analytics
3. **Collect feedback** from test users
4. **Iterate** on design and features

### Upgrade to Production:

1. Deploy backend to Railway/Render
2. Add authentication to Streamlit
3. Set up monitoring and logging
4. Configure custom domain
5. Add more FAQs to backend
6. Integrate with CRM

---

## 📚 Additional Resources

### Streamlit:
- Docs: https://docs.streamlit.io/
- Community Cloud: https://streamlit.io/cloud
- Gallery: https://streamlit.io/gallery

### Backend Deployment:
- Railway: https://railway.app/
- Render: https://render.com/
- Heroku: https://heroku.com/

### ngrok:
- Docs: https://ngrok.com/docs
- Dashboard: https://dashboard.ngrok.com/

---

## ✅ Deployment Checklist

Before going live:

- [ ] FastAPI backend is running and accessible
- [ ] Streamlit app starts without errors
- [ ] Student chat sends messages successfully
- [ ] Advisor dashboard shows queue and stats
- [ ] Escalations appear in advisor dashboard
- [ ] "View Full Conversation" works
- [ ] API connection status shows "Online"
- [ ] Environment variables are set correctly
- [ ] Mobile view works properly
- [ ] Shared URL is accessible to others

---

**You're all set! Your Streamlit app is ready to deploy.** 🚀

For questions or issues, check the troubleshooting section above or reach out for help.
