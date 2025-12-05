"""
Enrollment FAQ Chatbot with Human-in-the-Loop
Multi-Agent Workflow for Competency-Based University

This workflow creates a complete chatbot system for handling prospective student
inquiries about enrollment at a large online university following a competency model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import AgentOrchestrator, TaskStatus, WorkflowPhase
from agent_base import create_agent
from agent_definitions import AgentRole
import json


def print_section(title, emoji=""):
    """Print formatted section header"""
    print("\n" + "=" * 100)
    print(f"{emoji} {title}")
    print("=" * 100)


def run_enrollment_chatbot_workflow():
    """
    Complete workflow for building enrollment FAQ chatbot with HITL

    Project Context:
    - Target: Prospective students considering online competency-based programs
    - University: Large online institution with competency-based education model
    - Goal: Answer FAQs automatically, escalate complex queries to human advisors
    - Key Feature: Human-in-the-Loop for seamless handoff
    """

    print_section("ENROLLMENT FAQ CHATBOT WITH HUMAN-IN-THE-LOOP", "🎓")

    print("""
Project Overview:
─────────────────
Build an intelligent FAQ chatbot for prospective students at a large online university
that follows a competency-based education model.

Key Requirements:
• Answer common enrollment questions automatically (FAQs)
• Understand competency-based education model context
• Escalate complex queries to human enrollment advisors
• Seamless handoff with context preservation
• Track conversation history and outcomes

Competency-Based Model Context:
• Students progress by demonstrating mastery (not seat time)
• Flexible, self-paced learning
• Prior learning assessment (PLA) for transfer credits
• Subscription-based tuition model

Let's build this with our 7-agent team...
    """)

    # Initialize orchestrator
    orchestrator = AgentOrchestrator("Enrollment FAQ Chatbot - Competency-Based University")

    # Set initial context
    orchestrator.context.constraints = [
        "Must handle 10,000+ prospective student inquiries per month",
        "95% of simple FAQs handled without human intervention",
        "Complex queries escalated within 30 seconds",
        "FERPA compliance for student data",
        "Integration with CRM (Salesforce/HubSpot)",
        "24/7 availability",
        "Support for multiple languages (English, Spanish priority)"
    ]

    orchestrator.context.metadata = {
        "university_type": "Large online university",
        "education_model": "Competency-based (self-paced, mastery-based)",
        "target_audience": "Prospective students (adults, working professionals)",
        "primary_use_case": "Enrollment/admissions FAQs",
        "key_differentiator": "Human-in-the-Loop for complex queries"
    }

    # Create agents
    agents = {role: create_agent(role, orchestrator) for role in AgentRole}
    print(f"\n✓ Initialized orchestrator and {len(agents)} specialized agents")

    # =========================================================================
    # PHASE 1: DISCOVERY - Product Manager
    # =========================================================================
    print_section("PHASE 1: DISCOVERY", "📋")
    orchestrator.advance_phase(WorkflowPhase.DISCOVERY)

    pm_task = orchestrator.create_task(
        title="Analyze Enrollment Chatbot Requirements",
        description="""
Analyze requirements for an enrollment FAQ chatbot with Human-in-the-Loop capability.

Context:
- Large online university with 50,000+ students
- Competency-based education model (CBE)
- Target: Prospective students in enrollment/inquiry phase
- Current: Call center with 20 advisors handles all inquiries
- Pain: Advisors spend 60% time answering repetitive FAQs
- Goal: Automate FAQs, free advisors for complex consultations

Competency-Based Model Specifics:
- No traditional semesters or credit hours
- Students pay subscription fee (e.g., $3,000 per 6-month term)
- Progress by demonstrating competency in assessments
- Prior learning assessment (PLA) can reduce time-to-degree
- Flexible start dates (monthly cohorts)

Please analyze:
1. User personas (prospective students, enrollment advisors)
2. Common FAQ categories and questions
3. Criteria for escalation to human advisors
4. HITL handoff workflow requirements
5. Success metrics
6. Integration needs (CRM, telephony, chat platforms)
        """,
        assigned_to=AgentRole.PRODUCT_MANAGER
    )

    print(f"\n📝 Task Created: {pm_task.title}")

    # Execute PM task with domain-specific analysis
    pm_agent = agents[AgentRole.PRODUCT_MANAGER]
    context = orchestrator.get_context_for_agent(AgentRole.PRODUCT_MANAGER)

    pm_output = {
        "agent": "product_manager",
        "task_id": pm_task.id,
        "deliverable": "requirements_document",

        "user_personas": [
            {
                "name": "Maria - Career Changer",
                "age": 35,
                "background": "Retail manager, 2 years community college",
                "situation": "Wants to finish bachelor's degree while working full-time",
                "goals": [
                    "Understand if her previous college credits will transfer",
                    "Learn how competency-based model works",
                    "Determine if she can afford tuition while working",
                    "Find out how quickly she can complete degree"
                ],
                "pain_points": [
                    "Confused about 'competency-based' vs traditional college",
                    "Worried about cost and time commitment",
                    "Needs flexible schedule around work",
                    "Wants to know if prior work experience counts"
                ],
                "typical_questions": [
                    "How does competency-based education work?",
                    "Can I transfer my community college credits?",
                    "How much does it cost per term?",
                    "How long will it take me to graduate?",
                    "Can I get credit for my work experience?"
                ]
            },
            {
                "name": "James - Military Veteran",
                "age": 28,
                "background": "6 years military service, some college credits",
                "situation": "Using GI Bill benefits, wants accelerated degree",
                "goals": [
                    "Maximize GI Bill benefits",
                    "Get credit for military training",
                    "Complete degree as fast as possible",
                    "Understand admission requirements"
                ],
                "pain_points": [
                    "Traditional colleges too slow (semester-based)",
                    "Needs to know if military training transfers",
                    "Concerned about using up GI Bill benefits",
                    "Wants clear ROI on education investment"
                ],
                "typical_questions": [
                    "Do you accept GI Bill?",
                    "Can I get credit for military training (JST)?",
                    "How fast can I finish?",
                    "What's the cost if using VA benefits?",
                    "Can I start anytime or only certain dates?"
                ]
            },
            {
                "name": "Lisa - Enrollment Advisor",
                "role": "Enrollment advisor with 3 years experience",
                "responsibilities": [
                    "Guide prospective students through enrollment",
                    "Explain competency-based model",
                    "Assess prior learning for transfer credits",
                    "Help students understand costs and financial aid"
                ],
                "pain_points": [
                    "60% of time answering same basic questions",
                    "Can't give personalized attention to all inquiries",
                    "Repetitive work leads to burnout",
                    "Miss follow-up opportunities due to volume"
                ],
                "goals": [
                    "Spend time on consultative, high-value conversations",
                    "Focus on students needing guidance, not just info",
                    "Improve conversion rates through better engagement",
                    "Reduce response time for all inquiries"
                ]
            }
        ],

        "faq_categories": {
            "program_information": {
                "complexity": "Simple",
                "bot_handle": True,
                "questions": [
                    "What programs do you offer?",
                    "Is your university accredited?",
                    "What is competency-based education?",
                    "How is this different from traditional college?",
                    "What degrees can I earn?"
                ]
            },
            "admission_requirements": {
                "complexity": "Simple to Medium",
                "bot_handle": "Partial (escalate for complex cases)",
                "questions": [
                    "What are the admission requirements?",
                    "Do I need a high school diploma or GED?",
                    "Is there an application fee?",
                    "When can I start?",
                    "Do you require SAT/ACT scores?"
                ]
            },
            "transfer_credits": {
                "complexity": "Medium to Complex",
                "bot_handle": "Initial info only, escalate for evaluation",
                "questions": [
                    "Will my credits transfer?",
                    "How do I get a transcript evaluation?",
                    "Can I get credit for work experience?",
                    "What is prior learning assessment (PLA)?",
                    "How many credits do I need to graduate?"
                ]
            },
            "tuition_and_costs": {
                "complexity": "Simple",
                "bot_handle": True,
                "questions": [
                    "How much does it cost?",
                    "What is the tuition per term?",
                    "Are there additional fees?",
                    "Do you offer payment plans?",
                    "Is financial aid available?"
                ]
            },
            "financial_aid": {
                "complexity": "Complex",
                "bot_handle": "Basic info, escalate for personalized guidance",
                "questions": [
                    "Do you accept FAFSA?",
                    "Can I use my GI Bill?",
                    "What scholarships are available?",
                    "How do I apply for financial aid?",
                    "Will I qualify for aid?"
                ]
            },
            "program_format": {
                "complexity": "Simple",
                "bot_handle": True,
                "questions": [
                    "Is this program online?",
                    "How does the pacing work?",
                    "Are there live classes or self-paced?",
                    "How long does it take to complete?",
                    "Can I work full-time while enrolled?"
                ]
            },
            "competency_model": {
                "complexity": "Medium",
                "bot_handle": "Explanation, escalate if confused",
                "questions": [
                    "What does 'competency-based' mean?",
                    "How do assessments work?",
                    "Are there grades?",
                    "How is this different from credit hours?",
                    "What if I already know the material?"
                ]
            }
        },

        "escalation_criteria": {
            "automatic_escalation": [
                "Student indicates they have specific disabilities/accommodations",
                "Complex financial situation (bankruptcy, defaulted loans)",
                "International student visa questions",
                "Student expresses dissatisfaction or frustration",
                "Question about re-admission after academic dismissal",
                "Questions about specific medical/nursing program requirements",
                "Student requests to speak to human advisor",
                "Bot confidence score < 70% on response"
            ],
            "suggested_escalation": [
                "Student asks 3+ follow-up questions on same topic",
                "Prior learning assessment evaluation needed",
                "Student has unique work experience for credit",
                "Complex transfer credit situation (multiple institutions)",
                "Student asking about program-specific career outcomes",
                "Questions about military credit evaluation (JST, ACE)"
            ],
            "escalation_timing": {
                "immediate": "Student explicitly requests human",
                "within_30_seconds": "Automatic escalation triggers",
                "next_available": "Suggested escalation, can continue with bot"
            }
        },

        "core_requirements": {
            "functional": [
                {
                    "id": "REQ-001",
                    "category": "FAQ Handling",
                    "requirement": "Answer common enrollment questions automatically",
                    "description": "Bot handles 70+ common FAQs across 7 categories with 90%+ accuracy",
                    "priority": "P0"
                },
                {
                    "id": "REQ-002",
                    "category": "HITL",
                    "requirement": "Intelligent escalation to human advisors",
                    "description": "System detects complex queries and seamlessly transfers to available human advisor with full context",
                    "priority": "P0"
                },
                {
                    "id": "REQ-003",
                    "category": "Context Preservation",
                    "requirement": "Maintain conversation history during handoff",
                    "description": "When escalating, human advisor sees full chat history, student profile, and bot's confidence scores",
                    "priority": "P0"
                },
                {
                    "id": "REQ-004",
                    "category": "CBE Expertise",
                    "requirement": "Deep knowledge of competency-based education model",
                    "description": "Bot can explain CBE vs traditional model, pacing, assessments, subscription model with examples",
                    "priority": "P0"
                },
                {
                    "id": "REQ-005",
                    "category": "Lead Capture",
                    "requirement": "Capture and qualify leads for CRM",
                    "description": "Collect contact info, program interest, timeline, create lead in CRM with qualification score",
                    "priority": "P1"
                },
                {
                    "id": "REQ-006",
                    "category": "Multi-channel",
                    "requirement": "Support multiple communication channels",
                    "description": "Website chat, SMS, WhatsApp, Facebook Messenger with unified experience",
                    "priority": "P1"
                },
                {
                    "id": "REQ-007",
                    "category": "Analytics",
                    "requirement": "Track bot performance and conversation analytics",
                    "description": "Dashboard showing resolution rate, escalation rate, common questions, student satisfaction",
                    "priority": "P1"
                },
                {
                    "id": "REQ-008",
                    "category": "Advisor Tools",
                    "requirement": "Agent dashboard for managing conversations",
                    "description": "Advisors see queue, student info, suggested responses, can take over or monitor bot",
                    "priority": "P0"
                }
            ],
            "non_functional": [
                {
                    "id": "NFR-001",
                    "category": "Performance",
                    "requirement": "Response time < 2 seconds for 95% of bot responses",
                    "rationale": "Students expect instant answers"
                },
                {
                    "id": "NFR-002",
                    "category": "Availability",
                    "requirement": "99.9% uptime for bot, human handoff always available",
                    "rationale": "Students inquire 24/7, can't afford downtime"
                },
                {
                    "id": "NFR-003",
                    "category": "Scalability",
                    "requirement": "Handle 10,000+ concurrent conversations",
                    "rationale": "Peak enrollment periods, marketing campaigns"
                },
                {
                    "id": "NFR-004",
                    "category": "Accuracy",
                    "requirement": "90%+ accuracy on FAQ responses (verified by advisors)",
                    "rationale": "Wrong information damages trust and enrollment"
                },
                {
                    "id": "NFR-005",
                    "category": "Compliance",
                    "requirement": "FERPA compliant, no PII logged unnecessarily",
                    "rationale": "Student data privacy laws"
                }
            ]
        },

        "success_metrics": {
            "bot_effectiveness": {
                "resolution_rate": "Target: 70%+ of inquiries resolved without human",
                "accuracy_rate": "Target: 90%+ correct responses (advisor validation)",
                "response_time": "Target: <2 seconds average",
                "satisfaction": "Target: 4.0+ out of 5 rating"
            },
            "advisor_impact": {
                "time_savings": "Target: 60% reduction in time on FAQs",
                "focus_improvement": "Target: 80%+ of advisor time on consultative conversations",
                "capacity_increase": "Target: Each advisor handles 3x more students",
                "satisfaction": "Target: Advisor satisfaction score 4.5+"
            },
            "business_outcomes": {
                "lead_capture": "Target: 90%+ contact info capture rate",
                "response_time": "Target: <1 minute average first response",
                "conversion_rate": "Target: 15% improvement in inquiry-to-application",
                "cost_savings": "Target: $500K annually in advisor capacity"
            }
        },

        "hitl_workflow": {
            "stages": [
                {
                    "stage": "1. Bot First Contact",
                    "description": "Bot greets student, asks how it can help, handles initial questions",
                    "duration": "0-5 minutes typically"
                },
                {
                    "stage": "2. Escalation Trigger",
                    "description": "Bot detects complex query or student requests human",
                    "actions": [
                        "Save conversation context",
                        "Check advisor availability",
                        "Queue student if no advisors available",
                        "Notify student of wait time"
                    ]
                },
                {
                    "stage": "3. Human Handoff",
                    "description": "Advisor receives notification with full context",
                    "advisor_sees": [
                        "Complete chat history",
                        "Student profile (if returning)",
                        "Topics discussed",
                        "Bot confidence scores",
                        "Recommended talking points"
                    ]
                },
                {
                    "stage": "4. Human Conversation",
                    "description": "Advisor takes over, bot monitors in background",
                    "features": [
                        "Bot suggests answers (advisor can use or ignore)",
                        "Bot flags when CRM update needed",
                        "Bot tracks conversation for analytics"
                    ]
                },
                {
                    "stage": "5. Resolution & Follow-up",
                    "description": "Advisor resolves query, schedules follow-up if needed",
                    "actions": [
                        "Update CRM with notes",
                        "Schedule callback if requested",
                        "Send summary email to student",
                        "Bot learns from conversation (feedback loop)"
                    ]
                }
            ],
            "advisor_dashboard_features": [
                "Real-time queue of waiting students",
                "Student profiles with prior interactions",
                "Suggested responses from bot",
                "One-click actions (schedule call, send email, create application)",
                "Performance metrics (conversations handled, average time, satisfaction)",
                "Knowledge base search",
                "CRM integration (view/update records)"
            ]
        },

        "sample_faqs_and_answers": {
            "what_is_cbe": {
                "question": "What is competency-based education?",
                "bot_response": "Great question! Competency-based education (CBE) is a flexible, self-paced approach where you progress by demonstrating mastery of skills, not by spending time in class.\n\nKey differences from traditional college:\n• **Self-paced**: Move as fast or slow as you need\n• **No semesters**: Start monthly, work on your own schedule\n• **Prior knowledge counts**: If you already know material, prove it and move on\n• **Subscription model**: Pay per 6-month term, take as many courses as you can complete\n• **Real-world skills**: Focus on what employers need\n\nExample: If you already know accounting from work, take the assessment. Pass it, and you're done with that course - no need to sit through lectures!\n\nWant to learn more about how assessments work?",
                "confidence": 0.95,
                "category": "competency_model"
            },
            "will_credits_transfer": {
                "question": "Will my previous college credits transfer?",
                "bot_response": "Good question! Credit transfer depends on several factors:\n\n**Generally transferable:**\n• Regionally accredited colleges/universities\n• Grades of C or better\n• Courses relevant to your program\n• Completed within last 10 years (for most programs)\n\n**We'll evaluate:**\n• Official transcripts from all institutions\n• Course descriptions and syllabi\n• Alignment with our competency outcomes\n\n**To get started:**\n1. Request transcripts from previous schools\n2. Submit them with your application\n3. We'll provide a transfer credit evaluation (usually within 2 weeks)\n\nMany students transfer 30-60 credits! Want me to connect you with an enrollment advisor who can give you a preliminary assessment?",
                "confidence": 0.85,
                "escalation_trigger": "User responds 'yes' to advisor question",
                "category": "transfer_credits"
            },
            "how_much_does_it_cost": {
                "question": "How much does tuition cost?",
                "bot_response": "Our tuition structure is designed to be affordable and predictable:\n\n**Subscription Model:**\n• $3,000 per 6-month term (undergraduate)\n• $4,000 per 6-month term (graduate)\n• Take as many courses as you can complete in that term\n• No per-credit fees\n\n**Additional costs:**\n• $50 application fee (often waived during promotions)\n• ~$200-300 for course materials per term\n• Some programs have proctoring fees ($25-50 per assessment)\n\n**Financial aid available:**\n• Federal student aid (FAFSA)\n• Military benefits (GI Bill, TA)\n• Employer tuition assistance\n• Payment plans\n\n**Example:** If you complete 4 courses in 6 months = $3,000 total. That's $750 per course!\n\nWould you like information about financial aid or payment options?",
                "confidence": 0.95,
                "category": "tuition_and_costs"
            },
            "can_i_work_full_time": {
                "question": "Can I work full-time and attend this program?",
                "bot_response": "Absolutely! Most of our students work full-time. Here's how it works:\n\n**Flexibility built for working adults:**\n• 100% online - study anytime, anywhere\n• No scheduled class meetings (unless you want mentor sessions)\n• Self-paced - work ahead on weekends or slow weeks\n• Mobile-friendly platform for learning on-the-go\n\n**Typical time commitment:**\n• 15-20 hours per week for full-time progress\n• Can do more in busy work periods, catch up in slow periods\n• Average student completes bachelor's in 2-3 years (part-time)\n\n**Real student example:**\nMaria, a retail manager, studied early mornings (5-7am) and evenings (9-11pm). She completed her bachelor's in 2.5 years while working 40+ hours per week.\n\n**Support available:**\n• Program mentors (academic coaches)\n• Flexible assessment scheduling\n• Career-relevant coursework you can apply at work immediately\n\nWant to talk to an enrollment advisor about creating a plan that fits your schedule?",
                "confidence": 0.90,
                "category": "program_format"
            }
        },

        "integration_requirements": [
            {
                "system": "CRM (Salesforce/HubSpot)",
                "purpose": "Lead management and student tracking",
                "data_flow": "Bidirectional - create leads, update records, retrieve student history",
                "fields": ["Contact info", "Program interest", "Lead score", "Communication history"]
            },
            {
                "system": "Chat platforms (Website, SMS, WhatsApp)",
                "purpose": "Multi-channel communication",
                "integration": "Webhook-based, unified inbox"
            },
            {
                "system": "Knowledge base",
                "purpose": "FAQ content and university information",
                "update_frequency": "Weekly, version controlled"
            },
            {
                "system": "Advisor dashboard/telephony",
                "purpose": "Agent workspace for HITL",
                "features": ["Queue management", "Context transfer", "Call controls"]
            },
            {
                "system": "Analytics platform",
                "purpose": "Reporting and insights",
                "metrics": ["Bot performance", "Escalation patterns", "Student satisfaction"]
            }
        ],

        "scope": {
            "in_scope": [
                "FAQ chatbot for enrollment inquiries",
                "Human-in-the-loop escalation workflow",
                "Advisor dashboard for managing conversations",
                "CRM integration for lead capture",
                "Multi-channel support (web, SMS, WhatsApp)",
                "Analytics and reporting",
                "Competency-based education explanation",
                "Basic transfer credit information"
            ],
            "out_of_scope": [
                "Actual transcript evaluation (done by registrar)",
                "Application processing (separate system)",
                "Financial aid calculations (done by FA office)",
                "Course enrollment (post-admission)",
                "Student portal features (for enrolled students)",
                "Live video chat (text-based only for MVP)"
            ],
            "future_considerations": [
                "Voice/phone integration",
                "Proactive outreach (bot initiates contact)",
                "Multilingual support beyond English/Spanish",
                "AI-powered lead scoring and routing",
                "Automated appointment scheduling"
            ]
        },

        "next_steps": [
            "System Architect: Design chatbot architecture with HITL workflow",
            "AI Engineer: Design conversation flows and escalation logic",
            "Backend Engineer: Implement bot service and CRM integration",
            "Frontend Engineer: Build advisor dashboard",
            "QA Engineer: Create test scenarios covering FAQ and HITL flows"
        ]
    }

    orchestrator.update_task_status(pm_task.id, TaskStatus.COMPLETED, pm_output)
    orchestrator.context.requirements = pm_output

    print("\n✅ Product Manager Analysis Complete")
    print(f"   Deliverable: {pm_output['deliverable']}")
    print(f"   User Personas: {len(pm_output['user_personas'])}")
    print(f"   FAQ Categories: {len(pm_output['faq_categories'])}")
    print(f"   Functional Requirements: {len(pm_output['core_requirements']['functional'])}")

    print("\n👥 Key User Personas:")
    for persona in pm_output['user_personas']:
        print(f"   • {persona['name']}")

    print("\n📚 FAQ Categories:")
    for category, details in pm_output['faq_categories'].items():
        print(f"   • {category}: {details['complexity']} complexity")

    print("\n🎯 Sample FAQs Available:")
    for faq_key in pm_output['sample_faqs_and_answers'].keys():
        print(f"   • {faq_key}")

    # Save PM output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pm_output_file = os.path.join(script_dir, "enrollment_chatbot_pm_requirements.json")
    with open(pm_output_file, 'w') as f:
        json.dump(pm_output, f, indent=2)
    print(f"\n💾 Detailed requirements saved to: {pm_output_file}")

    # Continue with more phases...
    print_section("WORKFLOW SUMMARY", "🎉")

    status = orchestrator.get_workflow_status()
    print(f"""
📊 Current Status:
   Project: {status['project_name']}
   Phase: {status['current_phase']}
   Tasks Completed: {status['task_status']['completed']}/{status['total_tasks']}

✅ Phase 1 Complete: Discovery & Requirements
   • 3 User Personas (Prospective students + Enrollment advisor)
   • 7 FAQ Categories mapped
   • 8 Functional Requirements defined
   • 5 Non-Functional Requirements defined
   • HITL workflow stages designed
   • Sample FAQs with bot responses created
   • Escalation criteria defined

🎯 Key Insights:
   • 70%+ of FAQs can be automated
   • 60% advisor time savings potential
   • Human-in-the-loop essential for complex queries
   • Competency-based model needs clear explanation
   • Multi-channel support required (web, SMS, WhatsApp)

💰 Expected Benefits:
   • $500K annual cost savings
   • 3x advisor capacity increase
   • 15% conversion rate improvement
   • <1 minute average response time
    """)

    # Save workflow state
    workflow_file = os.path.join(script_dir, "enrollment_chatbot_workflow.json")
    orchestrator.save_state(workflow_file)
    print(f"\n💾 Workflow state saved to: {workflow_file}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    run_enrollment_chatbot_workflow()
