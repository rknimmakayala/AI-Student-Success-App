# Phase 1 MVP: Product Requirements Document

## Executive Summary

**Product Name:** AI Student Success App - Phase 1 MVP
**Target Users:** Cricket academy students (ages 9-14) and their parents
**Core Value Proposition:** A safe, parent-controlled voice journaling app that helps young athletes reflect on their cricket practice, build emotional awareness, and maintain consistent check-ins with supportive AI-generated responses.

**Key Differentiators:**
- Child-safe by design (no public sharing, parental consent required)
- Voice-first interface optimized for 9-14 year olds
- Rule-based responses (no unpredictable LLM behavior in Phase 1)
- Parent transparency and control over all child data
- "Not a coach replacement" boundary messaging

---

## Product Vision & Goals

### Vision
Create a trusted digital companion that helps young cricket players develop self-reflection habits, emotional awareness, and resilience through daily voice check-ins, while giving parents meaningful insight into their child's athletic journey.

### Phase 1 Goals
1. **Safety First:** Zero tolerance for unsafe content or data exposure
2. **Parent Trust:** Full transparency and control over child data
3. **Habit Formation:** Enable daily voice check-ins with supportive feedback
4. **Simplicity:** Ship a functional MVP in weeks, not months
5. **Validation:** Prove the concept with 5-10 pilot families

### Success Metrics (Phase 1)
- 70%+ of kids complete 4+ check-ins in their first week
- 90%+ parent satisfaction with safety controls
- Zero safety incidents requiring external escalation
- 80%+ parents find weekly summaries valuable

---

## User Personas

### Primary Persona: "Priya" (Age 11, Cricket Player)
- **Background:** Plays cricket 3-4 times/week at local academy
- **Tech Comfort:** Uses iPad for games, familiar with voice recording
- **Needs:**
  - Quick way to capture thoughts after practice
  - Encouragement and acknowledgment of effort
  - Private space (not shared with teammates or coach)
- **Pain Points:**
  - Too tired to type long entries
  - Wants to talk but parents are busy
  - Feels pressure about performance

### Secondary Persona: "Rajesh" (Parent of 11-year-old)
- **Background:** Invested in child's cricket development, not tech-savvy
- **Needs:**
  - Know how child is doing emotionally with cricket
  - Safety guarantees (no strangers, no public posts)
  - Simple weekly summaries (not overwhelming data)
- **Pain Points:**
  - Doesn't always know what questions to ask
  - Worried about screen time and online safety
  - Wants insight without invading privacy

---

## Functional Requirements

### FR-1: Parent Account Management

#### FR-1.1: Parent Registration
- Parent can create account with email and password/PIN
- **Required fields:** email (unique), password (min 8 chars), explicit consent checkbox
- System validates email format and password strength
- System sends verification email (optional in Phase 1, recommended for Phase 2)
- **Acceptance Criteria:**
  - Account creation fails if consent is not checked
  - Duplicate email addresses are rejected
  - Session is created upon successful signup

#### FR-1.2: Parent Login/Logout
- Parent can log in with email + password
- System creates secure session cookie
- Parent can log out (session destroyed)
- **Acceptance Criteria:**
  - Failed login shows generic error (no email enumeration)
  - Sessions expire after 7 days of inactivity
  - Rate limiting: max 5 failed attempts per 15 minutes

#### FR-1.3: Parent Consent Management
- Consent must be explicitly granted during signup
- Parent cannot create child profiles without consent
- Consent timestamp is stored (`consent_at` field)
- **Acceptance Criteria:**
  - UI clearly states what data is collected and how it's used
  - Child creation is blocked until consent exists

---

### FR-2: Child Profile Management

#### FR-2.1: Create Child Profile
- Parent can create multiple child profiles
- **Required fields:** child name, age (9-14), tone preference (calm/cheerful/coachy)
- System generates unique 6-character Kid Code (alphanumeric, case-insensitive)
- Kid Code is hashed before storage
- **Acceptance Criteria:**
  - Kid Code shown only once to parent (with copy button)
  - Age validation: must be 9-14
  - Parent can create up to 5 child profiles per account

#### FR-2.2: View Child Profiles
- Parent dashboard shows list of all child profiles
- Each profile card displays: name, age, Kid Code status (active/inactive)
- Parent can select a child to view details and check-ins
- **Acceptance Criteria:**
  - List is sorted by creation date (newest first)
  - Visual indicator for children with unread safety alerts

#### FR-2.3: Edit Child Profile
- Parent can update: child name, age, tone preference
- Parent cannot change Kid Code (must delete/recreate for new code)
- **Acceptance Criteria:**
  - Changes save immediately
  - Age still constrained to 9-14

#### FR-2.4: Delete Child Profile
- Parent can delete child profile
- Deletion removes: profile, all journal entries, audio files, summaries
- System shows confirmation dialog with warning
- **Acceptance Criteria:**
  - Deletion is permanent and irreversible
  - All associated data is removed from database and file system
  - Parent receives confirmation message

---

### FR-3: Child Authentication

#### FR-3.1: Kid Code Login
- Child enters 6-character Kid Code (case-insensitive)
- System validates code against hashed values
- System creates kid session (separate from parent session)
- **Acceptance Criteria:**
  - Generic error message for invalid code (no enumeration)
  - Rate limiting: max 10 failed attempts per 30 minutes per IP
  - Session expires after 24 hours or on logout

#### FR-3.2: Kid Logout
- Child can log out manually
- Logout destroys kid session
- **Acceptance Criteria:**
  - Child is redirected to login screen
  - Session cookie is cleared

---

### FR-4: Voice Check-In Submission

#### FR-4.1: Voice Recording
- Child can start/stop audio recording via simple UI
- Recording uses browser MediaRecorder API
- Recording duration: 30-120 seconds
- Visual feedback during recording (timer, waveform optional)
- **Acceptance Criteria:**
  - Browser requests microphone permission on first use
  - Recording stops automatically at 120 seconds
  - Child can re-record before submission
  - Audio format: WebM or MP3 (browser default)

#### FR-4.2: Optional Transcript
- Browser attempts speech-to-text using Web Speech API (best-effort)
- If STT fails, child can type short summary manually
- Transcript is optional (not required for submission)
- **Acceptance Criteria:**
  - STT works in Chrome/Edge (skip Safari/Firefox if not supported)
  - Child can edit auto-generated transcript
  - Submission proceeds even if transcript is empty

#### FR-4.3: Structured Check-In Fields
- **Mood:** dropdown (happy, okay, tired, frustrated, nervous, other)
- **Focus:** dropdown (batting, bowling, fielding, fitness, mindset, other)
- **Proud moment:** text input (optional, max 200 chars)
- **Try next:** text input (optional, max 200 chars)
- **Acceptance Criteria:**
  - Mood and Focus are required fields
  - Character limits enforced client-side and server-side

#### FR-4.4: Submit Check-In
- Child clicks "Submit" to upload recording + metadata
- System sends multipart form data to backend
- Loading indicator shown during upload
- **Acceptance Criteria:**
  - Submission disabled if audio is missing
  - Client shows error if upload fails (with retry option)
  - Success redirects to response screen

---

### FR-5: AI Response Generation (Rule-Based)

#### FR-5.1: Normal Response
- System generates supportive response using rule-based templates
- Response includes:
  - Opener based on tone preference (calm/cheerful/coachy)
  - Mood acknowledgment
  - Effort-based praise (not outcome-based)
  - One gentle reflection question
  - Boundary reminder: "Remember to check with your coach for technique tips!"
- **Acceptance Criteria:**
  - Response is deterministic (same inputs = same output)
  - No LLM calls in Phase 1
  - Response length: 50-150 words
  - Age-appropriate language (9-14 year olds)

#### FR-5.2: Safety-Flagged Response
- If safety rules are triggered, override normal response
- Safety response template:
  - "I can hear that something difficult is happening."
  - "Please talk to a parent or trusted adult right now."
  - "You can also call [crisis hotline] anytime."
- **Acceptance Criteria:**
  - Safety response is calm and non-alarming
  - Safety flag is stored in database
  - Parent is alerted on dashboard

---

### FR-6: Safety System (Phase 1 Rules)

#### FR-6.1: Keyword Detection
- System scans transcript for high-risk keywords:
  - **Self-harm:** "hurt myself", "want to die", "kill myself", etc.
  - **Abuse:** "hit me", "touched me", "scared to go home", etc.
  - **Violence:** "bring a weapon", "hurt someone", etc.
  - **Explicit content:** sexual language (age-inappropriate)
- Pattern matching is case-insensitive
- **Acceptance Criteria:**
  - False positives are acceptable (conservative approach)
  - Flagged entries show safety badge in parent view

#### FR-6.2: Safety Event Logging
- Each flag creates SafetyEvent record with:
  - child_id, timestamp, category, severity (0-3), matched_keywords
- Parent sees alert in dashboard with category (not full transcript)
- **Acceptance Criteria:**
  - Parent can view flagged entry details
  - Safety events are never auto-deleted

#### FR-6.3: Parent Safety Alerts
- Dashboard shows badge/notification for new safety flags
- Parent can mark alerts as "reviewed"
- **Acceptance Criteria:**
  - Alerts persist until parent acknowledges
  - Visual hierarchy: critical > high > medium > low

---

### FR-7: Parent Dashboard

#### FR-7.1: View Check-Ins (Last 7 Days)
- Parent selects child to view recent check-ins
- Display shows: date, time, mood, focus, audio player
- Audio can be played inline
- **Acceptance Criteria:**
  - Entries sorted by date (newest first)
  - Only last 7 days shown by default
  - "View all" option loads older entries

#### FR-7.2: Transcript Visibility Toggle
- Parent can show/hide transcripts for each entry
- Toggle respects child privacy (encourage trust)
- **Acceptance Criteria:**
  - Default state: transcripts hidden
  - Toggle is per-entry (not global)

#### FR-7.3: Weekly Summary Generation
- Parent clicks "Generate Weekly Summary" for selected child
- System aggregates last 7 days:
  - Number of check-ins
  - Most common mood
  - Most common focus area
  - Top 3 "proud moments"
  - Top 3 "try next" items
- Summary presented in parent-friendly language
- **Acceptance Criteria:**
  - Summary generation takes <2 seconds
  - Summary is saved to database (WeeklySummaries table)
  - Parent can regenerate summary if new entries added

#### FR-7.4: Data Export
- Parent can export all data for a child as JSON
- Export includes: profile, check-ins, transcripts, summaries
- Download triggers immediately
- **Acceptance Criteria:**
  - JSON is valid and human-readable
  - Export includes all historical data (not just last 7 days)
  - File name: `{child_name}_export_{date}.json`

#### FR-7.5: Data Deletion
- Parent can delete all child data (see FR-2.4)
- Confirmation dialog warns about permanence
- **Acceptance Criteria:**
  - Same as FR-2.4

---

## Non-Functional Requirements

### NFR-1: Security
- All passwords hashed using bcrypt (min cost factor 10)
- Kid Codes hashed using bcrypt
- Sessions use httpOnly, secure cookies
- Rate limiting on all auth endpoints
- No sensitive data in client-side logs

### NFR-2: Privacy
- No third-party analytics in Phase 1
- No public sharing features
- No social features (no friend lists, no leaderboards)
- Audio files stored with non-enumerable filenames (UUIDs)

### NFR-3: Performance
- Page load time: <2 seconds on 3G connection
- Audio upload: supports files up to 5MB
- Database queries: <200ms for dashboard views

### NFR-4: Compatibility
- **Desktop:** Chrome, Edge, Firefox (latest 2 versions)
- **Mobile:** Chrome on Android, Safari on iOS
- **Audio:** MediaRecorder supported in Chrome/Edge (warn on unsupported browsers)

### NFR-5: Accessibility
- Minimum font size: 16px (readable for 9-14 year olds)
- High contrast text (WCAG AA)
- Keyboard navigation for all interactive elements
- Screen reader friendly (basic ARIA labels)

---

## Out of Scope (Phase 1)

### Explicitly Excluded
- ❌ LLM-generated responses (rule-based only)
- ❌ Coach dashboards or coach access
- ❌ Multi-academy support or academy admin roles
- ❌ Social features (sharing, commenting, likes)
- ❌ Gamification (points, badges, streaks)
- ❌ Email/SMS notifications
- ❌ Mobile apps (web-only for Phase 1)
- ❌ Advanced analytics or ML insights
- ❌ Video check-ins (audio only)
- ❌ Automatic crisis intervention (parent alert only)

### Deferred to Phase 2
- Coach view (read-only, aggregate insights)
- Advanced safety (sentiment analysis, LLM-based detection)
- Email summaries to parents
- Streak tracking and habit nudges
- Multi-language support

---

## User Stories & Acceptance Criteria

### Parent User Stories

#### Story P-1: Parent Registration
**As a parent,**
**I want to** create an account with explicit consent,
**So that** I can safely manage my child's check-ins.

**Acceptance Criteria:**
- Given I visit the signup page
- When I enter email, password, and check the consent box
- Then my account is created and I'm logged in
- And I cannot create an account without checking consent
- And duplicate emails are rejected with clear error

---

#### Story P-2: Create Child Profile
**As a parent,**
**I want to** create a profile for my child and receive a Kid Code,
**So that** my child can log in safely without email/password.

**Acceptance Criteria:**
- Given I'm logged in as a parent
- When I create a child profile with name, age, and tone
- Then a unique Kid Code is generated and shown once
- And I can copy the Kid Code to share with my child
- And the Kid Code is not visible after I navigate away

---

#### Story P-3: View Child Check-Ins
**As a parent,**
**I want to** view my child's recent check-ins and listen to their recordings,
**So that** I can understand their emotional state and cricket journey.

**Acceptance Criteria:**
- Given my child has submitted check-ins
- When I select my child in the dashboard
- Then I see the last 7 days of entries with date, mood, focus
- And I can play audio recordings inline
- And I can toggle transcript visibility per entry

---

#### Story P-4: Generate Weekly Summary
**As a parent,**
**I want to** generate a weekly summary of my child's check-ins,
**So that** I can quickly understand patterns and trends.

**Acceptance Criteria:**
- Given my child has 1+ check-ins in the last 7 days
- When I click "Generate Weekly Summary"
- Then I see aggregated data: check-in count, common mood/focus, top proud/try-next
- And the summary is in simple, parent-friendly language
- And I can regenerate if new check-ins are added

---

#### Story P-5: Receive Safety Alerts
**As a parent,**
**I want to** be immediately alerted if my child's check-in is flagged for safety,
**So that** I can provide timely support.

**Acceptance Criteria:**
- Given my child's check-in contains flagged keywords
- When I log into my dashboard
- Then I see a prominent safety alert badge
- And I can view the category and severity
- And I can mark the alert as reviewed

---

#### Story P-6: Export Child Data
**As a parent,**
**I want to** export all of my child's data,
**So that** I have a backup or can share with professionals if needed.

**Acceptance Criteria:**
- Given I'm viewing my child's profile
- When I click "Export Data"
- Then a JSON file downloads with all profile, check-in, and summary data
- And the file is named with child name and date

---

#### Story P-7: Delete Child Data
**As a parent,**
**I want to** permanently delete my child's profile and all data,
**So that** I have full control over data retention.

**Acceptance Criteria:**
- Given I'm viewing my child's profile
- When I click "Delete" and confirm
- Then all profile, check-in, audio, and summary data is deleted
- And I receive confirmation of deletion
- And the child no longer appears in my dashboard

---

### Child User Stories

#### Story C-1: Kid Code Login
**As a child,**
**I want to** log in using my Kid Code,
**So that** I can access my check-in journal easily.

**Acceptance Criteria:**
- Given I have a valid Kid Code from my parent
- When I enter the code on the login page
- Then I'm logged in and see the check-in page
- And the code is case-insensitive
- And I see a generic error if the code is invalid

---

#### Story C-2: Record Voice Check-In
**As a child,**
**I want to** record my thoughts about cricket practice using my voice,
**So that** I can reflect quickly without typing.

**Acceptance Criteria:**
- Given I'm on the check-in page
- When I click "Start Recording"
- Then my microphone activates and a timer starts
- And I can speak for 30-120 seconds
- And I can stop early or re-record before submitting

---

#### Story C-3: Add Structured Reflection
**As a child,**
**I want to** select my mood and focus area,
**So that** I can give context to my voice recording.

**Acceptance Criteria:**
- Given I've recorded audio
- When I fill out mood, focus, proud, and try-next fields
- Then I can submit the complete check-in
- And mood/focus are required but proud/try-next are optional

---

#### Story C-4: Receive Supportive Response
**As a child,**
**I want to** receive an encouraging response after submitting,
**So that** I feel heard and motivated.

**Acceptance Criteria:**
- Given I submitted a check-in
- When the upload completes
- Then I see a supportive message that matches my tone preference
- And the message acknowledges my mood
- And the message includes one gentle question
- And the message reminds me to talk to my coach for technique help

---

#### Story C-5: Safety Escalation Response
**As a child,**
**I want to** be gently directed to help if I mention something concerning,
**So that** I get appropriate support.

**Acceptance Criteria:**
- Given my check-in contains safety-flagged keywords
- When I submit
- Then I see a calm message encouraging me to talk to a trusted adult
- And crisis hotline information is provided
- And my parent is alerted on their dashboard

---

## Technical Requirements

### Tech Stack (Recommended)
- **Frontend:** HTML/CSS/JavaScript or React (single-page app)
- **Backend:** Node.js + Express (or Python FastAPI)
- **Database:** SQLite (Phase 1) or PostgreSQL (if scaling expected)
- **File Storage:** Local file system (`/uploads` directory)
- **Authentication:** Session cookies (express-session or similar)
- **Audio Capture:** MediaRecorder API (browser)
- **Speech-to-Text:** Web Speech API (best-effort, optional)

### Data Model (Minimum Viable Schema)
See separate technical specification document for full schema.

**Core Tables:**
1. `Parents` (parent_id, email, password_hash, consent_at, created_at)
2. `Children` (child_id, parent_id, name, age, tone, kid_code_hash, created_at)
3. `JournalEntries` (entry_id, child_id, audio_path, transcript, mood, focus, proud, try_next, safety_flag, created_at)
4. `WeeklySummaries` (summary_id, child_id, week_start_date, summary_json, created_at)
5. `SafetyEvents` (event_id, child_id, category, severity, note, created_at)

---

## Deployment & Infrastructure (Phase 1)

### Hosting Options
- **Replit:** Fastest for MVP (built-in hosting, simple deployment)
- **Heroku/Railway:** If you need more control
- **VPS (DigitalOcean/Linode):** If you expect >100 users in Phase 1

### MVP Infrastructure
- Single server (monolith OK for Phase 1)
- SQLite database (on-disk)
- Local file storage for audio
- HTTPS required (Let's Encrypt or platform default)

### Backup Strategy
- Daily database backups (automated via cron or platform)
- Audio files backed up to separate storage (optional in Phase 1)

---

## Go-Live Checklist

Before calling Phase 1 "fully functional," verify:

- [ ] Parent signup/login works
- [ ] Consent gating prevents child creation without consent
- [ ] Child profile creation generates Kid Code (shown once)
- [ ] Kid Code login works (case-insensitive)
- [ ] Voice recording starts/stops correctly (Chrome desktop + mobile)
- [ ] Check-in submission uploads audio + metadata
- [ ] Rule-based response generates correctly for all tone preferences
- [ ] Safety flags trigger for test keywords
- [ ] Safety-flagged entries show override response to child
- [ ] Parent dashboard shows last 7 days of check-ins
- [ ] Audio playback works in parent dashboard
- [ ] Weekly summary generates with correct aggregations
- [ ] Data export downloads valid JSON
- [ ] Data deletion removes all child records and audio files
- [ ] Rate limiting works on login endpoints
- [ ] No public sharing features exist
- [ ] HTTPS enabled
- [ ] Basic error handling (network failures, invalid inputs)

---

## Risks & Mitigations

### Risk 1: Child Safety Incident
**Likelihood:** Medium | **Impact:** Critical
**Mitigation:**
- Conservative keyword detection (accept false positives)
- Parent alerts for all flagged content
- Clear boundary messaging: "This is not a crisis service"
- Crisis hotline information in safety responses

### Risk 2: Browser Compatibility (Audio Recording)
**Likelihood:** Medium | **Impact:** High
**Mitigation:**
- Test on Chrome, Edge, Safari (iOS), Chrome (Android)
- Show browser compatibility warning on unsupported browsers
- Provide fallback: manual text entry if recording fails

### Risk 3: Parent Doesn't Review Safety Alerts
**Likelihood:** High | **Impact:** High
**Mitigation:**
- Prominent visual alerts on dashboard
- Safety alerts never auto-dismiss
- Phase 2: Add email/SMS notifications

### Risk 4: Kid Code Shared Publicly
**Likelihood:** Low | **Impact:** Medium
**Mitigation:**
- Educate parents: "Keep Kid Code private"
- Allow parent to regenerate Kid Code (Phase 2 feature)
- Rate limiting on login attempts

---

## Phase 2 Preview (Not in Scope for MVP)

- Coach read-only dashboards (aggregate insights, no individual access)
- Email summaries to parents (weekly automated emails)
- Streak tracking and gentle habit nudges
- LLM-generated responses (with additional safety layers)
- Advanced safety: sentiment analysis + context-aware flagging
- Mobile app (React Native or PWA)
- Multi-academy support with academy admin roles

---

## Appendix: Response Templates (Phase 1)

### Calm Tone Example
**Mood: Tired | Focus: Batting**

> "Thanks for checking in. It sounds like today was a big effort in batting practice. Even when you're tired, showing up matters. One thing to think about: what felt good about your practice today, even if it was hard? Remember, if you need help with technique, your coach is the best person to ask!"

### Cheerful Tone Example
**Mood: Happy | Focus: Bowling**

> "Awesome! It's great to hear you had a happy practice working on your bowling! Your effort is really shining through. Here's a question: what's one thing you want to try next time to keep building on today? And don't forget—your coach is there for any technique questions!"

### Coachy Tone Example
**Mood: Frustrated | Focus: Fielding**

> "I hear you—fielding can be frustrating sometimes. What matters is that you're putting in the work. Think about this: what's one small thing you did well today, even if it didn't feel perfect? Keep talking to your coach about the details—they're your best resource for improvement!"

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-08 | Product Team | Initial Phase 1 PRD |

---

**Next Steps:**
1. Review and approve PRD with stakeholders
2. Create technical specification document
3. Design UI/UX mockups for kid and parent flows
4. Define pilot academy selection criteria
5. Build Phase 1 MVP (estimated 2-3 weeks for experienced dev team)

