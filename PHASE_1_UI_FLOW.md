# Phase 1 MVP: Screen-by-Screen UI Flow

## Document Overview

**Version:** 1.0
**Last Updated:** 2026-01-08
**Purpose:** Define the complete user interface flows for both parent and child users, including screen layouts, interactions, and navigation paths.

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [Parent User Flows](#parent-user-flows)
3. [Child User Flows](#child-user-flows)
4. [Shared Components](#shared-components)
5. [Error States & Edge Cases](#error-states--edge-cases)
6. [Mobile Responsive Considerations](#mobile-responsive-considerations)

---

## 1. Design Principles

### Child UI (Ages 9-14)
- **Large touch targets:** Minimum 44x44px for buttons
- **Simple language:** Age-appropriate, encouraging, no jargon
- **Minimal steps:** Reduce cognitive load (3-step max flows)
- **Visual feedback:** Clear recording states, loading indicators
- **Friendly colors:** Warm, inviting (avoid harsh reds/grays)
- **No distractions:** No ads, no social features, no gamification (Phase 1)

### Parent UI
- **Information clarity:** Easy-to-scan dashboards
- **Safety first:** Prominent safety alerts
- **Privacy controls:** Clear data export/delete options
- **Trust signals:** Explicit consent language, transparent data use
- **Efficiency:** Quick access to recent check-ins and summaries

---

## 2. Parent User Flows

### Flow 2.1: Parent Signup

**Screen: Parent Signup**

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│              AI Student Success App                 │
│          Help your child reflect on cricket         │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │ Email                                         │ │
│  │ ┌───────────────────────────────────────────┐ │ │
│  │ │ parent@example.com                        │ │ │
│  │ └───────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │ Password                                      │ │
│  │ ┌───────────────────────────────────────────┐ │ │
│  │ │ ••••••••                                  │ │ │
│  │ └───────────────────────────────────────────┘ │ │
│  │ Must be at least 8 characters                 │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │ ☐ I consent to my child using this app       │ │
│  │                                               │ │
│  │   By checking this box, I agree that:         │ │
│  │   • Voice recordings and text will be stored  │ │
│  │   • I can export or delete data anytime       │ │
│  │   • This is not a substitute for professional │ │
│  │     coaching or mental health support         │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│         ┌─────────────────────────────┐            │
│         │      Create Account         │            │
│         └─────────────────────────────┘            │
│                                                     │
│         Already have an account? Log in            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Validation:**
- Email format check (real-time)
- Password strength indicator (min 8 chars, 1 uppercase, 1 number)
- Consent checkbox must be checked

**Success State:**
- Redirect to "Create Your First Child Profile" screen
- Session cookie set

**Error States:**
- Email already exists: "This email is already registered. Try logging in instead."
- Weak password: Show strength meter (red → yellow → green)
- Consent not checked: Disable "Create Account" button

---

### Flow 2.2: Parent Login

**Screen: Parent Login**

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│              AI Student Success App                 │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │ Email                                         │ │
│  │ ┌───────────────────────────────────────────┐ │ │
│  │ │ parent@example.com                        │ │ │
│  │ └───────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │ Password                                      │ │
│  │ ┌───────────────────────────────────────────┐ │ │
│  │ │ ••••••••                                  │ │ │
│  │ └───────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│         ┌─────────────────────────────┐            │
│         │         Log In              │            │
│         └─────────────────────────────┘            │
│                                                     │
│         Don't have an account? Sign up             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Success State:**
- Redirect to Child List screen

**Error State:**
- "Invalid email or password. Please try again." (generic, no enumeration)
- After 5 failed attempts: "Too many attempts. Please try again in 15 minutes."

---

### Flow 2.3: Create Child Profile

**Screen: Create Child Profile**

```
┌─────────────────────────────────────────────────────┐
│  ← Back to Dashboard                                │
│                                                     │
│           Create Your Child's Profile               │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │ Child's Name                                  │ │
│  │ ┌───────────────────────────────────────────┐ │ │
│  │ │ Priya                                     │ │ │
│  │ └───────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │ Age                                           │ │
│  │ ┌───────────────────────────────────────────┐ │ │
│  │ │ 11                         ▼              │ │ │
│  │ └───────────────────────────────────────────┘ │ │
│  │ (Dropdown: 9, 10, 11, 12, 13, 14)             │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │ Tone Preference                               │ │
│  │                                               │ │
│  │  ○ Calm – Gentle and supportive              │ │
│  │  ● Cheerful – Upbeat and energetic           │ │
│  │  ○ Coachy – Direct and motivating            │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│         ┌─────────────────────────────┐            │
│         │    Create Profile           │            │
│         └─────────────────────────────┘            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Success State:**
- Modal overlay with Kid Code

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│               ✓ Profile Created!                   │
│                                                     │
│  Here is Priya's Kid Code. Save it now—you won't   │
│  see it again!                                      │
│                                                     │
│         ┌─────────────────────────────┐            │
│         │         AB3X7K              │  [Copy]    │
│         └─────────────────────────────┘            │
│                                                     │
│  Share this code with Priya so she can log in.     │
│                                                     │
│         ┌─────────────────────────────┐            │
│         │      Got it!                │            │
│         └─────────────────────────────┘            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**After "Got it!":**
- Redirect to Child List screen
- New child appears in list

---

### Flow 2.4: Child List (Dashboard Home)

**Screen: Child List**

```
┌─────────────────────────────────────────────────────┐
│  My Children                            [+ Add Child] [Logout]
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  Priya                                 🟡 1  │   │
│  │  Age 11 • Cheerful tone                     │   │
│  │  Last check-in: 2 hours ago                 │   │
│  │                                  [View →]   │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  Rohan                                      │   │
│  │  Age 13 • Calm tone                         │   │
│  │  Last check-in: 3 days ago                  │   │
│  │                                  [View →]   │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Legend:**
- 🟡 = Unread safety alert
- Last check-in timestamp shows recency

**Interactions:**
- Click "View →" → Navigate to Child Dashboard
- Click "+ Add Child" → Navigate to Create Child Profile
- Click "Logout" → Log out parent

---

### Flow 2.5: Child Dashboard (Selected Child)

**Screen: Child Dashboard**

```
┌─────────────────────────────────────────────────────┐
│  ← Back to Children                         [Logout]│
│                                                     │
│  Priya's Check-Ins          [⚙️ Settings]          │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ 🟡 Safety Alert                             │   │
│  │ A check-in from Jan 7 was flagged for      │   │
│  │ review. [View Details]                      │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ 📊 Weekly Summary                           │   │
│  │ [Generate Summary for Last 7 Days]          │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Last 7 Days                                        │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ Jan 7, 4:30 PM            😊 Happy          │   │
│  │ Focus: Bowling                              │   │
│  │ Proud: Got 3 wickets today!                 │   │
│  │                                             │   │
│  │ [▶️ Play Audio]    [Show Transcript]       │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ Jan 6, 3:45 PM            😴 Tired          │   │
│  │ Focus: Fitness                              │   │
│  │ Try next: Get more sleep                    │   │
│  │                                             │   │
│  │ [▶️ Play Audio]    [Show Transcript]       │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  [Load More]                                        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Safety Alert Interaction:**
- Click "View Details" → Navigate to Safety Alert Detail screen

**Weekly Summary Interaction:**
- Click "Generate Summary" → Show loading spinner → Display summary modal

**Entry Interactions:**
- Click "▶️ Play Audio" → Play audio inline (show pause button when playing)
- Click "Show Transcript" → Expand transcript below audio player

**Settings (Gear Icon):**
- Edit child profile
- Export data
- Delete child

---

### Flow 2.6: Weekly Summary

**Screen: Weekly Summary Modal**

```
┌─────────────────────────────────────────────────────┐
│                                                  ✕  │
│            Priya's Weekly Summary                   │
│            Week of Jan 6 - Jan 12                   │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ 📅 Check-ins: 5                             │   │
│  │ 😊 Most common mood: Happy                  │   │
│  │ 🏏 Most common focus: Bowling               │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Top Proud Moments:                                 │
│  • Got 3 wickets today!                             │
│  • Ran 2km without stopping                         │
│  • Helped teammate with grip                        │
│                                                     │
│  What Priya Wants to Try Next:                      │
│  • Work on slower balls                             │
│  • Practice cover drives                            │
│  • Build leg strength                               │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ Summary:                                    │   │
│  │ Priya had a great week with 5 check-ins!   │   │
│  │ She was mostly happy and focused on bowling.│   │
│  │ She's proud of her wicket-taking and       │   │
│  │ fitness progress. Next week, she wants to   │   │
│  │ work on slower balls and batting technique. │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│         ┌─────────────────────────────┐            │
│         │    Close                    │            │
│         └─────────────────────────────┘            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

### Flow 2.7: Safety Alert Detail

**Screen: Safety Alert Detail**

```
┌─────────────────────────────────────────────────────┐
│  ← Back to Dashboard                                │
│                                                     │
│             🟡 Safety Alert                         │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ Entry from: Jan 7, 4:30 PM                  │   │
│  │ Category: Self-harm language detected       │   │
│  │ Severity: High                              │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  What happened:                                     │
│  Priya's check-in contained language that may       │
│  indicate distress or self-harm thoughts. We        │
│  recommend reviewing the audio and transcript.      │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ [▶️ Play Audio]                             │   │
│  │                                             │   │
│  │ Transcript:                                 │   │
│  │ "Today was tough. I keep making mistakes   │   │
│  │ and sometimes I just want to hurt myself    │   │
│  │ when I mess up..."                          │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Recommended Actions:                               │
│  • Talk to Priya about what she's feeling          │
│  • Consider speaking with her coach or a            │
│    counselor if concerns persist                    │
│  • If you're worried, call CHILDLINE: 1098         │
│                                                     │
│         ┌─────────────────────────────┐            │
│         │  Mark as Reviewed           │            │
│         └─────────────────────────────┘            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**After "Mark as Reviewed":**
- Alert badge removed from Child List
- Alert still visible in history but not marked urgent

---

### Flow 2.8: Settings (Export/Delete)

**Screen: Child Settings**

```
┌─────────────────────────────────────────────────────┐
│  ← Back to Dashboard                                │
│                                                     │
│             Priya's Settings                        │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ Edit Profile                                │   │
│  │ Change name, age, or tone preference        │   │
│  │                                  [Edit →]   │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ Export Data                                 │   │
│  │ Download all check-ins, summaries, and     │   │
│  │ audio files as a JSON file                  │   │
│  │                              [Export →]     │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ ⚠️ Delete Child Profile                     │   │
│  │ Permanently delete Priya's profile and all  │   │
│  │ data. This cannot be undone.                │   │
│  │                              [Delete →]     │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Delete Confirmation Modal:**

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│            ⚠️ Delete Priya's Profile?              │
│                                                     │
│  This will permanently delete:                      │
│  • Priya's profile                                  │
│  • All check-in recordings and transcripts          │
│  • All weekly summaries                             │
│  • All safety alerts                                │
│                                                     │
│  This action cannot be undone.                      │
│                                                     │
│  Type "DELETE" to confirm:                          │
│  ┌───────────────────────────────────────────────┐ │
│  │                                               │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│    ┌──────────────┐        ┌──────────────┐       │
│    │   Cancel     │        │   Delete     │       │
│    └──────────────┘        └──────────────┘       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**After Deletion:**
- Redirect to Child List
- Show success message: "Priya's profile has been deleted."

---

## 3. Child User Flows

### Flow 3.1: Kid Login

**Screen: Kid Login**

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│              AI Student Success App                 │
│                                                     │
│                   🏏                                │
│                                                     │
│          Enter your Kid Code to check in           │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │ Kid Code                                      │ │
│  │ ┌───────────────────────────────────────────┐ │ │
│  │ │ AB3X7K                                    │ │ │
│  │ └───────────────────────────────────────────┘ │ │
│  │ (6 letters/numbers - ask your parent)         │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│         ┌─────────────────────────────┐            │
│         │       Let's Go!             │            │
│         └─────────────────────────────┘            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**UX Notes:**
- Input auto-capitalizes
- Simple, friendly language
- Large "Let's Go!" button (primary color, easy to tap)

**Success State:**
- Redirect to Check-In screen
- Show greeting: "Hi Priya! Ready to check in?"

**Error State:**
- "Hmm, that code didn't work. Try again or ask your parent."
- After 10 failed attempts: "Too many tries. Ask your parent for help."

---

### Flow 3.2: Voice Check-In

**Screen: Check-In (Step 1 - Record)**

```
┌─────────────────────────────────────────────────────┐
│                                        [Logout]     │
│                                                     │
│              Hi Priya! Ready to check in?           │
│                                                     │
│  Tell me about your cricket practice today.         │
│  You can talk for up to 2 minutes.                  │
│                                                     │
│                                                     │
│              ┌─────────────────┐                    │
│              │                 │                    │
│              │       🎤        │                    │
│              │                 │                    │
│              │  Start Recording│                    │
│              │                 │                    │
│              └─────────────────┘                    │
│                                                     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**After clicking "Start Recording":**

```
┌─────────────────────────────────────────────────────┐
│                                        [Logout]     │
│                                                     │
│              Recording... 00:45 / 02:00             │
│                                                     │
│              ┌─────────────────┐                    │
│              │                 │                    │
│              │  ⏺️ Recording   │                    │
│              │   [Waveform]    │                    │
│              │                 │                    │
│              └─────────────────┘                    │
│                                                     │
│              ┌─────────────────┐                    │
│              │  Stop Recording │                    │
│              └─────────────────┘                    │
│                                                     │
│         Speak clearly and take your time!           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**After clicking "Stop Recording":**

```
┌─────────────────────────────────────────────────────┐
│                                        [Logout]     │
│                                                     │
│              Great! Here's what you said:           │
│                                                     │
│  [▶️ Play Back]                                     │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │ Transcript (optional):                        │ │
│  │ ┌───────────────────────────────────────────┐ │ │
│  │ │ "Today was awesome! I got 3 wickets and   │ │ │
│  │ │  my coach said my action was looking      │ │ │
│  │ │  better..."                               │ │ │
│  │ └───────────────────────────────────────────┘ │ │
│  │ You can edit this if it's wrong               │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│    ┌──────────────┐        ┌──────────────┐       │
│    │  Re-Record   │        │   Next →     │       │
│    └──────────────┘        └──────────────┘       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**UX Notes:**
- Transcript is auto-generated (best-effort via Web Speech API)
- If STT fails, show empty text box: "You can type what you said here (optional)"
- "Re-Record" clears audio and returns to Step 1
- "Next →" advances to Step 2

---

### Flow 3.3: Check-In (Step 2 - Structured Fields)

**Screen: Check-In Details**

```
┌─────────────────────────────────────────────────────┐
│  ← Back                                 [Logout]    │
│                                                     │
│              Almost done! A few quick questions:    │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │ How are you feeling? *                        │ │
│  │ ┌───────────────────────────────────────────┐ │ │
│  │ │ 😊 Happy                     ▼            │ │ │
│  │ └───────────────────────────────────────────┘ │ │
│  │ Options: Happy, Okay, Tired, Frustrated,      │ │
│  │          Nervous, Other                       │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │ What did you focus on? *                      │ │
│  │ ┌───────────────────────────────────────────┐ │ │
│  │ │ 🏏 Bowling                   ▼            │ │ │
│  │ └───────────────────────────────────────────┘ │ │
│  │ Options: Batting, Bowling, Fielding, Fitness, │ │
│  │          Mindset, Other                       │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │ What are you proud of today?                  │ │
│  │ ┌───────────────────────────────────────────┐ │ │
│  │ │ Got 3 wickets!                            │ │ │
│  │ └───────────────────────────────────────────┘ │ │
│  │ (Optional - up to 200 characters)             │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │ What will you try next practice?              │ │
│  │ ┌───────────────────────────────────────────┐ │ │
│  │ │ Work on slower balls                      │ │ │
│  │ └───────────────────────────────────────────┘ │ │
│  │ (Optional - up to 200 characters)             │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│         ┌─────────────────────────────┐            │
│         │      Submit Check-In        │            │
│         └─────────────────────────────┘            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Validation:**
- Mood and Focus are required (marked with *)
- Proud and Try Next are optional
- Character limits enforced (200 chars each)

**After "Submit Check-In":**
- Show loading spinner: "Processing your check-in..."
- Advance to Response screen

---

### Flow 3.4: Response Display

**Screen: Response (Normal)**

```
┌─────────────────────────────────────────────────────┐
│                                        [Logout]     │
│                                                     │
│              ✓ Check-in submitted!                  │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │                                               │ │
│  │  Awesome! It's great to hear you had a happy  │ │
│  │  practice working on your bowling! Your       │ │
│  │  effort is really shining through.            │ │
│  │                                               │ │
│  │  Here's a question: what's one thing you want │ │
│  │  to try next time to keep building on today?  │ │
│  │                                               │ │
│  │  And don't forget—your coach is there for any │ │
│  │  technique questions!                         │ │
│  │                                               │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│         ┌─────────────────────────────┐            │
│         │     Done for Today          │            │
│         └─────────────────────────────┘            │
│                                                     │
│         ┌─────────────────────────────┐            │
│         │   Do Another Check-In       │            │
│         └─────────────────────────────┘            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Interactions:**
- "Done for Today" → Logout
- "Do Another Check-In" → Return to Step 1 (Record)

---

**Screen: Response (Safety Flagged)**

```
┌─────────────────────────────────────────────────────┐
│                                        [Logout]     │
│                                                     │
│              ✓ Check-in submitted                   │
│                                                     │
│  ┌───────────────────────────────────────────────┐ │
│  │                                               │ │
│  │  I can hear that something difficult is       │ │
│  │  happening.                                   │ │
│  │                                               │ │
│  │  Please talk to a parent or trusted adult     │ │
│  │  right now.                                   │ │
│  │                                               │ │
│  │  You can also call the Kids Helpline at       │ │
│  │  1800-551-800 anytime.                        │ │
│  │                                               │ │
│  │  You're not alone, and people care about you. │ │
│  │                                               │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│         ┌─────────────────────────────┐            │
│         │        Okay                 │            │
│         └─────────────────────────────┘            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**UX Notes:**
- Calm, non-alarming tone
- Clear action steps
- Crisis hotline number prominent
- Parent is alerted on their dashboard (child doesn't see this notification)

---

## 4. Shared Components

### Component 4.1: Loading Spinner

**Used in:**
- Login (parent & kid)
- Check-in submission
- Weekly summary generation
- Data export

**Design:**
```
    ┌──────────────┐
    │   ⏳ Loading  │
    │   Please wait│
    └──────────────┘
```

---

### Component 4.2: Error Message

**Used in:**
- Form validation errors
- Network errors
- Rate limiting errors

**Design:**
```
┌─────────────────────────────────────────────────────┐
│  ⚠️ Oops! Something went wrong.                     │
│  [Specific error message here]                      │
│  [Try Again Button]                                 │
└─────────────────────────────────────────────────────┘
```

---

### Component 4.3: Audio Player

**Used in:**
- Parent dashboard (entry playback)
- Kid check-in (playback before submit)

**Design:**
```
┌───────────────────────────────────────┐
│ [▶️] ━━━━●━━━━━━ 1:23 / 2:05        │
└───────────────────────────────────────┘
```

---

## 5. Error States & Edge Cases

### 5.1 Parent Errors

| Scenario | Error Message | Action |
|----------|---------------|--------|
| Duplicate email on signup | "This email is already registered. Try logging in instead." | Show "Log In" link |
| Weak password | "Password must be at least 8 characters with 1 uppercase and 1 number." | Inline validation |
| Consent not checked | Disable "Create Account" button | Show tooltip on hover |
| Failed login (5 attempts) | "Too many login attempts. Please try again in 15 minutes." | Disable form for 15 min |
| Network error on submission | "Network error. Please check your connection and try again." | Show retry button |
| Child age out of range | "Age must be between 9 and 14." | Inline validation |

---

### 5.2 Kid Errors

| Scenario | Error Message | Action |
|----------|---------------|--------|
| Invalid Kid Code | "Hmm, that code didn't work. Try again or ask your parent." | Clear input field |
| Failed kid login (10 attempts) | "Too many tries. Ask your parent for help." | Disable form for 30 min |
| Microphone denied | "We need your microphone to record. Please allow access in your browser settings." | Show help link |
| Recording too short (<30s) | "Recording is too short. Try recording for at least 30 seconds." | Allow re-record |
| Audio upload failed | "Upload failed. Check your internet and try again." | Show retry button |
| Missing required fields | Highlight fields in red with message: "Please answer this question." | Inline validation |

---

### 5.3 Edge Cases

**Parent:**
- No children created yet → Show "Create Your First Child" CTA
- Child has no check-ins → Show "No check-ins yet" placeholder
- Weekly summary with 0 check-ins → "No check-ins this week. Summary unavailable."
- Export data (large file) → Show progress bar

**Kid:**
- Web Speech API unavailable → Show manual text input fallback
- Browser doesn't support MediaRecorder → Show error: "Your browser doesn't support recording. Try Chrome or Edge."
- Session expired mid-check-in → Redirect to login with message: "Your session expired. Please log in again."

---

## 6. Mobile Responsive Considerations

### 6.1 Parent UI (Mobile)

- Stack child cards vertically
- Full-width buttons
- Collapsible safety alerts (tap to expand)
- Sticky header with "Back" and "Logout" always visible

### 6.2 Kid UI (Mobile)

- **Critical:** Large record button (min 100px diameter on mobile)
- Single-column layout for all forms
- Auto-zoom disabled on input fields (prevent iOS zoom)
- Use native dropdowns for better mobile UX

---

## Appendix: UI Design Tokens (Phase 1)

### Colors
```css
/* Primary Colors */
--primary-blue: #2563EB;
--primary-green: #10B981;
--primary-red: #EF4444;

/* Neutrals */
--neutral-white: #FFFFFF;
--neutral-light-gray: #F3F4F6;
--neutral-gray: #6B7280;
--neutral-dark: #1F2937;

/* Safety Alert */
--alert-yellow: #FBBF24;
--alert-red: #DC2626;

/* Kid UI (Warm & Friendly) */
--kid-primary: #F59E0B;  /* Orange */
--kid-background: #FEF3C7;  /* Light yellow */
```

### Typography
```css
/* Parent UI */
--font-family: 'Inter', 'Segoe UI', sans-serif;
--font-size-base: 16px;
--font-size-large: 20px;
--font-size-small: 14px;

/* Kid UI (Larger for readability) */
--kid-font-family: 'Poppins', 'Comic Sans MS', sans-serif;
--kid-font-size-base: 18px;
--kid-font-size-large: 24px;
```

### Spacing
```css
--spacing-xs: 8px;
--spacing-sm: 16px;
--spacing-md: 24px;
--spacing-lg: 32px;
--spacing-xl: 48px;
```

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-08 | Design Team | Initial UI flow documentation |

---

**Next Steps:**
1. Review UI flows with stakeholders
2. Create high-fidelity mockups in Figma
3. Build frontend components (React or Vanilla JS)
4. User testing with 2-3 parent-child pairs
5. Iterate based on feedback
