# Phase 1 MVP: Technical Specification

## Document Overview

**Version:** 1.0
**Last Updated:** 2026-01-08
**Status:** Draft for Review

This document provides the technical implementation details for the Phase 1 MVP of the AI Student Success App, including architecture, data models, API specifications, and deployment guidelines.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Technology Stack](#technology-stack)
3. [Database Schema](#database-schema)
4. [API Endpoints](#api-endpoints)
5. [Authentication & Security](#authentication--security)
6. [Safety System](#safety-system)
7. [Response Generation Engine](#response-generation-engine)
8. [File Storage](#file-storage)
9. [Frontend Components](#frontend-components)
10. [Deployment Guide](#deployment-guide)
11. [Testing Strategy](#testing-strategy)

---

## 1. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Client Layer                        │
│  ┌────────────────┐              ┌────────────────┐        │
│  │  Parent Web UI │              │   Kid Web UI   │        │
│  │   (React/JS)   │              │   (React/JS)   │        │
│  └────────────────┘              └────────────────┘        │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTPS
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Node.js + Express Server                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────┐    │  │
│  │  │  Auth    │  │  Safety  │  │   Response     │    │  │
│  │  │ Middleware│  │  Engine  │  │   Generator    │    │  │
│  │  └──────────┘  └──────────┘  └────────────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       Data Layer                            │
│  ┌──────────────┐              ┌────────────────┐          │
│  │   SQLite DB  │              │  File Storage  │          │
│  │  (Relational)│              │  (Audio Files) │          │
│  └──────────────┘              └────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

**Client Layer:**
- Parent UI: Dashboard, child management, check-in review, safety alerts
- Kid UI: Login, voice recording, check-in submission, response display

**Application Layer:**
- Express Server: Request routing, business logic, middleware orchestration
- Auth Middleware: Session management, role-based access control
- Safety Engine: Keyword detection, severity scoring, alert generation
- Response Generator: Rule-based template selection and personalization

**Data Layer:**
- SQLite: Relational data (users, profiles, check-ins, summaries)
- File Storage: Audio files (local filesystem in Phase 1)

---

## 2. Technology Stack

### Backend
```json
{
  "runtime": "Node.js 18+",
  "framework": "Express 4.x",
  "database": "SQLite 3.x (better-sqlite3 driver)",
  "session": "express-session + connect-sqlite3",
  "upload": "multer (multipart form data)",
  "password": "bcrypt (cost factor 10)",
  "validation": "express-validator",
  "security": "helmet (security headers)"
}
```

### Frontend
```json
{
  "framework": "React 18+ or Vanilla JS",
  "build": "Vite or Create React App",
  "audio": "MediaRecorder API (browser native)",
  "stt": "Web Speech API (optional, best-effort)",
  "http": "Fetch API or Axios"
}
```

### Development Tools
```json
{
  "linter": "ESLint",
  "formatter": "Prettier",
  "testing": "Jest + Supertest (backend), React Testing Library (frontend)",
  "version_control": "Git + GitHub"
}
```

### Deployment (Phase 1)
```json
{
  "platform": "Replit (recommended) or Railway/Render",
  "ssl": "Automatic HTTPS (platform-provided)",
  "env": ".env file + platform secrets"
}
```

---

## 3. Database Schema

### 3.1 Parents Table

```sql
CREATE TABLE parents (
  parent_id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  consent_at DATETIME NOT NULL,  -- ISO 8601 timestamp
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_parents_email ON parents(email);
```

**Constraints:**
- `email`: Must be valid email format, unique, max 255 chars
- `password_hash`: bcrypt hash (60 chars)
- `consent_at`: Cannot be null (consent required)

---

### 3.2 Children Table

```sql
CREATE TABLE children (
  child_id INTEGER PRIMARY KEY AUTOINCREMENT,
  parent_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  age INTEGER NOT NULL CHECK(age BETWEEN 9 AND 14),
  tone TEXT NOT NULL CHECK(tone IN ('calm', 'cheerful', 'coachy')),
  kid_code_hash TEXT UNIQUE NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (parent_id) REFERENCES parents(parent_id) ON DELETE CASCADE
);

CREATE INDEX idx_children_parent ON children(parent_id);
CREATE INDEX idx_children_code ON children(kid_code_hash);
```

**Constraints:**
- `name`: Max 100 chars
- `age`: 9-14 inclusive
- `tone`: Enum (calm, cheerful, coachy)
- `kid_code_hash`: bcrypt hash of 6-char alphanumeric code

**Kid Code Generation:**
```javascript
// Generate 6-char alphanumeric code (case-insensitive)
function generateKidCode() {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // Exclude I, O, 0, 1
  let code = '';
  for (let i = 0; i < 6; i++) {
    code += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return code;
}
```

---

### 3.3 JournalEntries Table

```sql
CREATE TABLE journal_entries (
  entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
  child_id INTEGER NOT NULL,
  audio_path TEXT NOT NULL,  -- Relative path to audio file
  transcript TEXT,           -- Optional, from STT or manual
  mood TEXT NOT NULL CHECK(mood IN ('happy', 'okay', 'tired', 'frustrated', 'nervous', 'other')),
  focus TEXT NOT NULL CHECK(focus IN ('batting', 'bowling', 'fielding', 'fitness', 'mindset', 'other')),
  proud TEXT,                -- Optional, max 200 chars
  try_next TEXT,             -- Optional, max 200 chars
  safety_flag BOOLEAN DEFAULT 0,
  safety_category TEXT,      -- E.g., 'self_harm', 'abuse', 'violence'
  safety_severity INTEGER DEFAULT 0,  -- 0=none, 1=low, 2=medium, 3=high
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (child_id) REFERENCES children(child_id) ON DELETE CASCADE
);

CREATE INDEX idx_entries_child ON journal_entries(child_id);
CREATE INDEX idx_entries_created ON journal_entries(created_at);
CREATE INDEX idx_entries_safety ON journal_entries(safety_flag, child_id);
```

**Constraints:**
- `audio_path`: Unique filename (UUID-based)
- `transcript`: Max 1000 chars
- `proud`, `try_next`: Max 200 chars each
- `mood`, `focus`: Enum values

---

### 3.4 WeeklySummaries Table

```sql
CREATE TABLE weekly_summaries (
  summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
  child_id INTEGER NOT NULL,
  week_start_date DATE NOT NULL,  -- Monday of the week
  checkin_count INTEGER NOT NULL,
  most_common_mood TEXT,
  most_common_focus TEXT,
  top_proud_moments TEXT,  -- JSON array of strings
  top_try_next TEXT,       -- JSON array of strings
  summary_text TEXT,       -- Human-readable summary
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (child_id) REFERENCES children(child_id) ON DELETE CASCADE
);

CREATE INDEX idx_summaries_child ON weekly_summaries(child_id);
CREATE INDEX idx_summaries_week ON weekly_summaries(week_start_date);
```

**Example JSON Fields:**
```json
{
  "top_proud_moments": ["Got 3 catches in fielding drill", "Bowled 5 overs without a break", "Helped teammate with stance"],
  "top_try_next": ["Work on cover drive", "Practice yorkers", "Build stamina"]
}
```

---

### 3.5 SafetyEvents Table

```sql
CREATE TABLE safety_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  child_id INTEGER NOT NULL,
  entry_id INTEGER,  -- Reference to journal entry
  category TEXT NOT NULL,  -- 'self_harm', 'abuse', 'violence', 'explicit'
  severity INTEGER NOT NULL CHECK(severity BETWEEN 0 AND 3),
  matched_keywords TEXT,  -- JSON array of detected keywords
  note TEXT,
  reviewed_by_parent BOOLEAN DEFAULT 0,
  reviewed_at DATETIME,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (child_id) REFERENCES children(child_id) ON DELETE CASCADE,
  FOREIGN KEY (entry_id) REFERENCES journal_entries(entry_id) ON DELETE SET NULL
);

CREATE INDEX idx_safety_child ON safety_events(child_id);
CREATE INDEX idx_safety_reviewed ON safety_events(reviewed_by_parent);
```

---

## 4. API Endpoints

### 4.1 Parent Authentication

#### POST `/api/parent/signup`
**Description:** Create new parent account

**Request Body:**
```json
{
  "email": "parent@example.com",
  "password": "SecurePass123!",
  "consent": true
}
```

**Validation:**
- `email`: Valid email format, unique
- `password`: Min 8 chars, at least 1 uppercase, 1 number
- `consent`: Must be `true`

**Response (201):**
```json
{
  "success": true,
  "parentId": 1,
  "message": "Account created successfully"
}
```

**Response (400 - Validation Error):**
```json
{
  "success": false,
  "errors": [
    { "field": "email", "message": "Email already exists" },
    { "field": "consent", "message": "Consent is required" }
  ]
}
```

---

#### POST `/api/parent/login`
**Description:** Parent login

**Request Body:**
```json
{
  "email": "parent@example.com",
  "password": "SecurePass123!"
}
```

**Response (200):**
```json
{
  "success": true,
  "parentId": 1
}
```

**Response (401):**
```json
{
  "success": false,
  "message": "Invalid email or password"
}
```

**Session:** Sets `parent_session` cookie (httpOnly, secure)

---

#### POST `/api/parent/logout`
**Description:** Parent logout

**Response (200):**
```json
{
  "success": true,
  "message": "Logged out successfully"
}
```

---

### 4.2 Child Management

#### POST `/api/parent/children`
**Description:** Create child profile (requires parent auth)

**Request Body:**
```json
{
  "name": "Priya",
  "age": 11,
  "tone": "cheerful"
}
```

**Response (201):**
```json
{
  "success": true,
  "childId": 1,
  "kidCode": "AB3X7K",
  "message": "Child profile created. Save this Kid Code - it will only be shown once."
}
```

**Note:** Kid Code is returned ONLY on creation, never again.

---

#### GET `/api/parent/children`
**Description:** Get all children for logged-in parent

**Response (200):**
```json
{
  "success": true,
  "children": [
    {
      "childId": 1,
      "name": "Priya",
      "age": 11,
      "tone": "cheerful",
      "createdAt": "2026-01-01T10:00:00Z",
      "hasUnreadAlerts": true
    }
  ]
}
```

---

#### PATCH `/api/parent/children/:childId`
**Description:** Update child profile

**Request Body:**
```json
{
  "name": "Priya Kumar",
  "age": 12,
  "tone": "calm"
}
```

**Response (200):**
```json
{
  "success": true,
  "message": "Profile updated"
}
```

---

#### DELETE `/api/parent/children/:childId`
**Description:** Delete child profile and all data

**Response (200):**
```json
{
  "success": true,
  "message": "Child profile and all associated data deleted permanently"
}
```

**Side Effects:**
- Deletes all journal entries
- Deletes all audio files
- Deletes all weekly summaries
- Deletes all safety events

---

### 4.3 Child Authentication

#### POST `/api/kid/login`
**Description:** Kid login with Kid Code

**Request Body:**
```json
{
  "kidCode": "ab3x7k"
}
```

**Validation:**
- Code is case-insensitive
- Rate limit: 10 attempts per 30 min per IP

**Response (200):**
```json
{
  "success": true,
  "childId": 1,
  "name": "Priya",
  "tone": "cheerful"
}
```

**Response (401):**
```json
{
  "success": false,
  "message": "Invalid Kid Code"
}
```

**Session:** Sets `kid_session` cookie (httpOnly, secure)

---

#### POST `/api/kid/logout`
**Description:** Kid logout

**Response (200):**
```json
{
  "success": true
}
```

---

### 4.4 Check-In Submission

#### POST `/api/kid/checkin`
**Description:** Submit voice check-in (multipart form)

**Headers:**
- `Content-Type: multipart/form-data`

**Form Fields:**
```
audio: [File] (required, max 5MB, .webm/.mp3/.wav)
transcript: [String] (optional, max 1000 chars)
mood: [String] (required, enum)
focus: [String] (required, enum)
proud: [String] (optional, max 200 chars)
tryNext: [String] (optional, max 200 chars)
```

**Processing Steps:**
1. Validate kid session
2. Save audio file with UUID filename
3. Run safety check on transcript
4. Generate response (normal or safety override)
5. Create journal entry record

**Response (200 - Normal):**
```json
{
  "success": true,
  "entryId": 42,
  "response": {
    "message": "Awesome! It's great to hear you had a happy practice working on your bowling! Your effort is really shining through. Here's a question: what's one thing you want to try next time to keep building on today? And don't forget—your coach is there for any technique questions!",
    "safetyFlagged": false
  }
}
```

**Response (200 - Safety Flagged):**
```json
{
  "success": true,
  "entryId": 43,
  "response": {
    "message": "I can hear that something difficult is happening. Please talk to a parent or trusted adult right now. You can also call the Kids Helpline at 1800-551-800 anytime.",
    "safetyFlagged": true
  }
}
```

**Response (400):**
```json
{
  "success": false,
  "errors": [
    { "field": "audio", "message": "Audio file is required" },
    { "field": "mood", "message": "Mood must be one of: happy, okay, tired, frustrated, nervous, other" }
  ]
}
```

---

### 4.5 Parent Dashboard

#### GET `/api/parent/children/:childId/entries`
**Description:** Get recent check-ins for a child

**Query Params:**
- `days`: Number of days to fetch (default: 7, max: 90)

**Response (200):**
```json
{
  "success": true,
  "entries": [
    {
      "entryId": 42,
      "createdAt": "2026-01-07T16:30:00Z",
      "mood": "happy",
      "focus": "bowling",
      "proud": "Got 3 wickets today!",
      "tryNext": "Work on slower balls",
      "audioUrl": "/api/parent/audio/42",
      "transcript": "Today was awesome...",
      "safetyFlag": false
    },
    {
      "entryId": 41,
      "createdAt": "2026-01-06T15:45:00Z",
      "mood": "tired",
      "focus": "fitness",
      "proud": null,
      "tryNext": "Get more sleep",
      "audioUrl": "/api/parent/audio/41",
      "transcript": null,
      "safetyFlag": false
    }
  ]
}
```

---

#### GET `/api/parent/audio/:entryId`
**Description:** Stream audio file for a specific entry

**Authorization:** Parent must own the child associated with entry

**Response (200):**
- `Content-Type: audio/webm` (or appropriate MIME type)
- Audio file stream

**Response (403):**
```json
{
  "success": false,
  "message": "Access denied"
}
```

---

#### POST `/api/parent/children/:childId/weekly-summary`
**Description:** Generate weekly summary

**Query Params:**
- `weekStart`: ISO date (default: last Monday)

**Response (200):**
```json
{
  "success": true,
  "summary": {
    "summaryId": 5,
    "weekStartDate": "2026-01-06",
    "checkinCount": 5,
    "mostCommonMood": "happy",
    "mostCommonFocus": "bowling",
    "topProudMoments": [
      "Got 3 wickets today!",
      "Ran 2km without stopping",
      "Helped teammate with grip"
    ],
    "topTryNext": [
      "Work on slower balls",
      "Practice cover drives",
      "Build leg strength"
    ],
    "summaryText": "Priya had a great week with 5 check-ins! She was mostly happy and focused on bowling. She's proud of her wicket-taking and fitness progress. Next week, she wants to work on slower balls and batting technique."
  }
}
```

---

#### GET `/api/parent/children/:childId/export`
**Description:** Export all child data as JSON

**Response (200):**
- `Content-Type: application/json`
- `Content-Disposition: attachment; filename="priya_export_2026-01-08.json"`

**Response Body:**
```json
{
  "child": {
    "childId": 1,
    "name": "Priya",
    "age": 11,
    "tone": "cheerful",
    "createdAt": "2026-01-01T10:00:00Z"
  },
  "entries": [ /* all journal entries */ ],
  "summaries": [ /* all weekly summaries */ ],
  "safetyEvents": [ /* all safety events */ ],
  "exportedAt": "2026-01-08T12:00:00Z"
}
```

---

#### GET `/api/parent/children/:childId/safety-alerts`
**Description:** Get safety alerts for a child

**Response (200):**
```json
{
  "success": true,
  "alerts": [
    {
      "eventId": 3,
      "category": "self_harm",
      "severity": 3,
      "createdAt": "2026-01-07T16:30:00Z",
      "reviewedByParent": false,
      "entryId": 42
    }
  ]
}
```

---

#### PATCH `/api/parent/safety-alerts/:eventId/review`
**Description:** Mark safety alert as reviewed

**Response (200):**
```json
{
  "success": true,
  "message": "Alert marked as reviewed"
}
```

---

## 5. Authentication & Security

### 5.1 Password Security

**Hashing:**
```javascript
const bcrypt = require('bcrypt');
const SALT_ROUNDS = 10;

// During signup
const passwordHash = await bcrypt.hash(password, SALT_ROUNDS);

// During login
const isValid = await bcrypt.compare(password, passwordHash);
```

**Requirements:**
- Min 8 characters
- At least 1 uppercase letter
- At least 1 number
- Special characters recommended (not enforced in Phase 1)

---

### 5.2 Session Management

**Implementation:**
```javascript
const session = require('express-session');
const SQLiteStore = require('connect-sqlite3')(session);

app.use(session({
  store: new SQLiteStore({ db: 'sessions.db' }),
  secret: process.env.SESSION_SECRET,
  resave: false,
  saveUninitialized: false,
  cookie: {
    httpOnly: true,
    secure: true,  // HTTPS only
    maxAge: 7 * 24 * 60 * 60 * 1000  // 7 days
  }
}));
```

**Session Types:**
- `parent_session`: Stores `parentId`
- `kid_session`: Stores `childId`

**Middleware:**
```javascript
// Require parent auth
function requireParent(req, res, next) {
  if (!req.session.parentId) {
    return res.status(401).json({ success: false, message: 'Unauthorized' });
  }
  next();
}

// Require kid auth
function requireKid(req, res, next) {
  if (!req.session.childId) {
    return res.status(401).json({ success: false, message: 'Unauthorized' });
  }
  next();
}
```

---

### 5.3 Rate Limiting

**Implementation:**
```javascript
const rateLimit = require('express-rate-limit');

// Parent login: 5 attempts per 15 min
const parentLoginLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 5,
  message: { success: false, message: 'Too many login attempts. Try again in 15 minutes.' }
});

// Kid login: 10 attempts per 30 min
const kidLoginLimiter = rateLimit({
  windowMs: 30 * 60 * 1000,
  max: 10,
  message: { success: false, message: 'Too many attempts. Try again later.' }
});

app.post('/api/parent/login', parentLoginLimiter, /* handler */);
app.post('/api/kid/login', kidLoginLimiter, /* handler */);
```

---

### 5.4 Security Headers

**Implementation:**
```javascript
const helmet = require('helmet');

app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'"],  // Adjust based on frontend
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", "data:"],
      mediaSrc: ["'self'", "blob:"],  // For audio playback
    }
  },
  hsts: { maxAge: 31536000 },
  frameguard: { action: 'deny' },
  noSniff: true
}));
```

---

## 6. Safety System

### 6.1 Keyword Detection

**Categories & Keywords:**

```javascript
const SAFETY_KEYWORDS = {
  self_harm: {
    keywords: [
      'hurt myself', 'kill myself', 'end my life', 'want to die',
      'suicide', 'cut myself', 'hate myself', 'self harm'
    ],
    severity: 3  // Critical
  },
  abuse: {
    keywords: [
      'hit me', 'hurt me', 'touched me', 'scared to go home',
      'makes me do', 'secret between us', 'don\'t tell anyone'
    ],
    severity: 3  // Critical
  },
  violence: {
    keywords: [
      'want to hurt', 'bring a weapon', 'kill someone',
      'going to fight', 'punch them'
    ],
    severity: 2  // High
  },
  explicit: {
    keywords: [
      /* age-inappropriate sexual content - define based on context */
    ],
    severity: 2  // High
  }
};
```

**Detection Function:**
```javascript
function detectSafetyConcerns(transcript) {
  if (!transcript) return { flagged: false };

  const lowerTranscript = transcript.toLowerCase();
  let flagged = false;
  let category = null;
  let severity = 0;
  let matchedKeywords = [];

  for (const [cat, config] of Object.entries(SAFETY_KEYWORDS)) {
    for (const keyword of config.keywords) {
      if (lowerTranscript.includes(keyword)) {
        flagged = true;
        category = cat;
        severity = Math.max(severity, config.severity);
        matchedKeywords.push(keyword);
      }
    }
  }

  return { flagged, category, severity, matchedKeywords };
}
```

---

### 6.2 Safety Response Override

**Safety Response Template:**
```javascript
function generateSafetyResponse(category, severity) {
  return {
    message: `I can hear that something difficult is happening. Please talk to a parent or trusted adult right now. You can also call the Kids Helpline at 1800-551-800 anytime.`,
    safetyFlagged: true,
    category,
    severity
  };
}
```

**Note:** Crisis hotline number should be region-appropriate (e.g., India: CHILDLINE 1098, Australia: Kids Helpline 1800-551-800, US: 988).

---

## 7. Response Generation Engine

### 7.1 Template Structure

**Template Categories:**
1. **Openers** (by tone)
2. **Mood Acknowledgments**
3. **Effort-Based Praise**
4. **Reflection Questions**
5. **Boundary Reminders**

---

### 7.2 Template Examples

```javascript
const RESPONSE_TEMPLATES = {
  openers: {
    calm: [
      "Thanks for checking in.",
      "I appreciate you taking time to reflect.",
      "It's good to hear from you."
    ],
    cheerful: [
      "Awesome!",
      "Great to hear from you!",
      "Yay, another check-in!"
    ],
    coachy: [
      "Good work checking in.",
      "Solid effort today.",
      "Nice to see you reflecting."
    ]
  },
  moodAcknowledgments: {
    happy: [
      "It sounds like you had a happy practice!",
      "I can hear the energy in your voice!",
      "It's great when practice feels good!"
    ],
    tired: [
      "It sounds like today was a big effort.",
      "Even when you're tired, showing up matters.",
      "Rest is important too—you earned it!"
    ],
    frustrated: [
      "Frustration is tough, but it shows you care.",
      "It's okay to feel frustrated sometimes.",
      "Even hard days teach us something."
    ],
    nervous: [
      "Nerves are normal—they show you're ready to try.",
      "It's okay to feel nervous before a challenge.",
      "Feeling nervous means it matters to you."
    ],
    okay: [
      "Okay is a good place to be.",
      "Not every day has to be amazing—steady progress counts.",
      "Okay days build consistency."
    ],
    other: [
      "I hear you.",
      "Thanks for sharing how you're feeling.",
      "Your feelings are valid."
    ]
  },
  praise: [
    "Your effort is really shining through.",
    "Showing up consistently is what matters most.",
    "You're building great habits.",
    "Keep putting in the work—it adds up.",
    "Effort is what leads to growth."
  ],
  questions: [
    "What's one thing you want to try next time?",
    "What felt good about your practice today?",
    "What's one small thing you did well today?",
    "What's something you're curious to work on next?",
    "How do you want to show up next practice?"
  ],
  boundaries: [
    "Remember, if you need help with technique, your coach is the best person to ask!",
    "Your coach is there for any technique questions!",
    "For cricket skills, your coach is your go-to expert!",
    "Don't forget—your coach is the best resource for technique work!"
  ]
};
```

---

### 7.3 Response Assembly

```javascript
function generateResponse(mood, tone, safetyCheck) {
  // If safety flagged, override with safety response
  if (safetyCheck.flagged) {
    return generateSafetyResponse(safetyCheck.category, safetyCheck.severity);
  }

  // Otherwise, assemble normal response
  const opener = randomChoice(RESPONSE_TEMPLATES.openers[tone]);
  const moodAck = randomChoice(RESPONSE_TEMPLATES.moodAcknowledgments[mood]);
  const praise = randomChoice(RESPONSE_TEMPLATES.praise);
  const question = randomChoice(RESPONSE_TEMPLATES.questions);
  const boundary = randomChoice(RESPONSE_TEMPLATES.boundaries);

  const message = `${opener} ${moodAck} ${praise} ${question} ${boundary}`;

  return {
    message,
    safetyFlagged: false
  };
}

function randomChoice(array) {
  return array[Math.floor(Math.random() * array.length)];
}
```

---

## 8. File Storage

### 8.1 Audio File Naming

**Strategy:** Use UUIDs to prevent enumeration and ensure uniqueness.

```javascript
const { v4: uuidv4 } = require('uuid');
const path = require('path');

function generateAudioFilename(originalFile) {
  const extension = path.extname(originalFile.originalname);  // e.g., .webm
  const uuid = uuidv4();
  return `${uuid}${extension}`;
}
```

**Storage Path:**
```
/uploads/audio/
  ├── a3f2e1b9-4c5d-6e7f-8a9b-0c1d2e3f4a5b.webm
  ├── b4g3f2c0-5d6e-7f8g-9b0c-1d2e3f4g5h6i.mp3
  └── ...
```

---

### 8.2 File Upload Configuration

```javascript
const multer = require('multer');
const path = require('path');

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, path.join(__dirname, 'uploads/audio'));
  },
  filename: (req, file, cb) => {
    const filename = generateAudioFilename(file);
    cb(null, filename);
  }
});

const upload = multer({
  storage,
  limits: {
    fileSize: 5 * 1024 * 1024,  // 5MB max
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['audio/webm', 'audio/mp3', 'audio/mpeg', 'audio/wav'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only audio files are allowed.'));
    }
  }
});

app.post('/api/kid/checkin', requireKid, upload.single('audio'), /* handler */);
```

---

### 8.3 File Cleanup on Deletion

When a child profile is deleted:
```javascript
const fs = require('fs').promises;

async function deleteChildData(childId) {
  // Get all audio paths
  const entries = db.prepare('SELECT audio_path FROM journal_entries WHERE child_id = ?').all(childId);

  // Delete audio files
  for (const entry of entries) {
    try {
      await fs.unlink(path.join(__dirname, 'uploads/audio', entry.audio_path));
    } catch (err) {
      console.error(`Failed to delete audio file: ${entry.audio_path}`, err);
    }
  }

  // Delete database records (cascade handles this if FK constraints are set)
  db.prepare('DELETE FROM children WHERE child_id = ?').run(childId);
}
```

---

## 9. Frontend Components

### 9.1 Parent UI Components

**Key Screens:**
1. **Signup/Login:** Email + password form, consent checkbox
2. **Child List:** Cards showing all children with status indicators
3. **Child Dashboard:** Selected child's recent check-ins, safety alerts
4. **Entry Detail:** Audio player, transcript, metadata, safety flag
5. **Weekly Summary:** Aggregated insights, download/regenerate options
6. **Settings:** Export data, delete child, logout

**Component Hierarchy:**
```
App
├── AuthProvider
│   ├── ParentLogin
│   ├── ParentSignup
│   └── ParentDashboard
│       ├── ChildList
│       ├── ChildDashboard
│       │   ├── EntryList
│       │   │   └── EntryCard
│       │   ├── SafetyAlerts
│       │   └── WeeklySummary
│       └── Settings
└── KidApp (separate routing)
```

---

### 9.2 Kid UI Components

**Key Screens:**
1. **Kid Login:** Kid Code input (6 chars, case-insensitive)
2. **Check-In:** Voice recorder, structured fields, submit button
3. **Response:** Display AI response after submission

**Component Hierarchy:**
```
App
├── KidLogin
└── KidCheckin
    ├── VoiceRecorder
    ├── TranscriptBox (optional)
    ├── StructuredFields
    └── ResponseDisplay
```

---

### 9.3 Voice Recorder Component

**Implementation (React Example):**
```javascript
import React, { useState, useRef } from 'react';

export default function VoiceRecorder({ onRecordingComplete }) {
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        onRecordingComplete(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);

      // Auto-stop at 120 seconds
      setTimeout(() => {
        if (mediaRecorder.state === 'recording') {
          stopRecording();
        }
      }, 120000);

    } catch (err) {
      alert('Microphone access denied. Please allow microphone to record.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setDuration(0);
    }
  };

  return (
    <div>
      <button onClick={isRecording ? stopRecording : startRecording}>
        {isRecording ? 'Stop Recording' : 'Start Recording'}
      </button>
      {isRecording && <p>Recording: {duration}s / 120s</p>}
    </div>
  );
}
```

---

## 10. Deployment Guide

### 10.1 Replit Deployment

**Step 1:** Create Replit project
```bash
# Select Node.js template
```

**Step 2:** Configure `.replit` file
```toml
run = "npm start"

[env]
SESSION_SECRET = "your-secret-here"
NODE_ENV = "production"
```

**Step 3:** Install dependencies
```bash
npm install express express-session connect-sqlite3 better-sqlite3 bcrypt multer helmet express-rate-limit express-validator uuid
```

**Step 4:** Project structure
```
/project-root
  /public           # Frontend files
  /uploads/audio    # Audio storage
  /data             # SQLite database
  /routes           # API routes
  /middleware       # Auth, validation
  /utils            # Safety, response generator
  server.js         # Main entry point
  package.json
```

**Step 5:** Run
```bash
npm start
```

Replit auto-generates HTTPS URL: `https://your-project.username.repl.co`

---

### 10.2 Environment Variables

**Required:**
```bash
SESSION_SECRET=<random-64-char-string>
NODE_ENV=production
DATABASE_PATH=/data/app.db
UPLOAD_PATH=/uploads/audio
```

**Optional (Phase 2):**
```bash
CRISIS_HOTLINE=1800-551-800  # Region-specific
EMAIL_SERVICE_API_KEY=...
```

---

### 10.3 Database Initialization

**Run on first deploy:**
```javascript
const Database = require('better-sqlite3');
const db = new Database('./data/app.db');

// Create tables (see schema in Section 3)
db.exec(`
  CREATE TABLE IF NOT EXISTS parents (...);
  CREATE TABLE IF NOT EXISTS children (...);
  CREATE TABLE IF NOT EXISTS journal_entries (...);
  CREATE TABLE IF NOT EXISTS weekly_summaries (...);
  CREATE TABLE IF NOT EXISTS safety_events (...);
`);

console.log('Database initialized.');
```

---

## 11. Testing Strategy

### 11.1 Backend Unit Tests

**Test Coverage:**
- Password hashing/validation
- Kid Code generation uniqueness
- Safety keyword detection
- Response template selection
- Session middleware

**Example (Jest):**
```javascript
const { detectSafetyConcerns } = require('./utils/safety');

test('should flag self-harm keywords', () => {
  const result = detectSafetyConcerns('I want to hurt myself');
  expect(result.flagged).toBe(true);
  expect(result.category).toBe('self_harm');
  expect(result.severity).toBe(3);
});

test('should not flag normal content', () => {
  const result = detectSafetyConcerns('Today was a great practice!');
  expect(result.flagged).toBe(false);
});
```

---

### 11.2 API Integration Tests

**Test Coverage:**
- Parent signup/login/logout
- Child creation (Kid Code returned once)
- Kid login (case-insensitive)
- Check-in submission (audio + metadata)
- Parent dashboard data retrieval
- Safety alert flow

**Example (Supertest):**
```javascript
const request = require('supertest');
const app = require('./server');

describe('POST /api/parent/signup', () => {
  test('should create parent account with consent', async () => {
    const res = await request(app)
      .post('/api/parent/signup')
      .send({
        email: 'test@example.com',
        password: 'SecurePass123!',
        consent: true
      });

    expect(res.statusCode).toBe(201);
    expect(res.body.success).toBe(true);
  });

  test('should reject signup without consent', async () => {
    const res = await request(app)
      .post('/api/parent/signup')
      .send({
        email: 'test2@example.com',
        password: 'SecurePass123!',
        consent: false
      });

    expect(res.statusCode).toBe(400);
  });
});
```

---

### 11.3 Frontend Tests

**Test Coverage:**
- Voice recorder (mock MediaRecorder API)
- Form validation
- Audio playback
- Safety alert display

---

### 11.4 Manual QA Checklist

**Critical Paths:**
- [ ] Parent can sign up with consent
- [ ] Parent can create child and receive Kid Code once
- [ ] Kid can log in (case-insensitive)
- [ ] Kid can record 30-120 second audio
- [ ] Check-in submission works on Chrome mobile
- [ ] Safety keywords trigger correct response
- [ ] Parent sees safety alerts
- [ ] Weekly summary generates correctly
- [ ] Export downloads valid JSON
- [ ] Delete removes all data

**Browser Testing:**
- [ ] Chrome (desktop + mobile)
- [ ] Safari (iOS)
- [ ] Edge (desktop)
- [ ] Firefox (warn if MediaRecorder unsupported)

---

## Appendix A: Sample Database Queries

### Get last 7 days of entries for a child
```sql
SELECT * FROM journal_entries
WHERE child_id = ?
  AND created_at >= datetime('now', '-7 days')
ORDER BY created_at DESC;
```

### Generate weekly summary data
```sql
SELECT
  COUNT(*) as checkin_count,
  mood,
  focus,
  GROUP_CONCAT(proud, '|||') as proud_moments,
  GROUP_CONCAT(try_next, '|||') as try_next_items
FROM journal_entries
WHERE child_id = ?
  AND created_at >= datetime('now', '-7 days')
GROUP BY mood, focus;
```

### Get unread safety alerts
```sql
SELECT * FROM safety_events
WHERE child_id = ?
  AND reviewed_by_parent = 0
ORDER BY created_at DESC;
```

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-08 | Engineering Team | Initial technical specification |

---

**Next Steps:**
1. Review technical spec with development team
2. Set up Replit project with boilerplate
3. Implement database schema and seed data
4. Build API endpoints (auth → child mgmt → check-in)
5. Build frontend components (kid UI → parent dashboard)
6. Test end-to-end flows
7. Deploy to production and pilot with 5-10 families
