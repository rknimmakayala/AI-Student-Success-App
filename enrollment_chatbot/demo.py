"""
Interactive Demo Script for Enrollment FAQ Chatbot
Run comprehensive test scenarios showing bot + HITL functionality
"""

import requests
import json
import time
from typing import Dict
from datetime import datetime


class ChatbotDemo:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.student_id = f"demo_student_{int(time.time())}"

    def print_separator(self, title=""):
        print("\n" + "=" * 80)
        if title:
            print(f" {title}")
            print("=" * 80)

    def print_message(self, role, message, confidence=None):
        """Pretty print a message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if role == "STUDENT":
            print(f"\n[{timestamp}] 👤 {role}:")
            print(f"   {message}")
        elif role == "BOT":
            conf_str = f" (confidence: {confidence:.0%})" if confidence else ""
            print(f"\n[{timestamp}] 🤖 {role}{conf_str}:")
            print(f"   {message}")
        elif role == "SYSTEM":
            print(f"\n[{timestamp}] ⚙️  {role}:")
            print(f"   {message}")
        elif role == "ADVISOR":
            print(f"\n[{timestamp}] 👨‍💼 {role}:")
            print(f"   {message}")

    def send_message(self, message: str) -> Dict:
        """Send message to chatbot"""
        response = requests.post(
            f"{self.base_url}/api/v1/chat",
            json={
                "student_id": self.student_id,
                "message": message
            }
        )
        return response.json()

    def get_stats(self) -> Dict:
        """Get chatbot statistics"""
        response = requests.get(f"{self.base_url}/api/v1/stats")
        return response.json()

    def get_advisor_queue(self) -> Dict:
        """Get advisor queue"""
        response = requests.get(f"{self.base_url}/api/v1/advisor/queue")
        return response.json()

    def scenario_simple_faq(self):
        """Scenario 1: Simple FAQ - Bot handles completely"""
        self.print_separator("SCENARIO 1: Simple FAQ (Bot Handles)")

        print("\n📝 Scenario: Student asks basic question about CBE")
        print("   Expected: Bot answers with high confidence, no escalation")

        time.sleep(1)

        # Question 1
        question = "What is competency-based education?"
        self.print_message("STUDENT", question)

        response = self.send_message(question)
        self.print_message("BOT", response["message"], response["confidence"])

        if response.get("follow_up_questions"):
            print(f"\n   💡 Suggested follow-ups:")
            for i, q in enumerate(response["follow_up_questions"], 1):
                print(f"      {i}. {q}")

        print(f"\n   ✅ Result: Bot handled with {response['confidence']:.0%} confidence")

    def scenario_tuition_question(self):
        """Scenario 2: Tuition cost inquiry"""
        self.print_separator("SCENARIO 2: Tuition Cost (Bot Handles)")

        print("\n📝 Scenario: Student asks about costs")
        print("   Expected: Bot provides detailed tuition information")

        time.sleep(1)

        question = "How much does it cost?"
        self.print_message("STUDENT", question)

        response = self.send_message(question)
        self.print_message("BOT", response["message"], response["confidence"])

        if response.get("follow_up_questions"):
            print(f"\n   💡 Suggested follow-ups:")
            for i, q in enumerate(response["follow_up_questions"], 1):
                print(f"      {i}. {q}")

        print(f"\n   ✅ Result: Clear pricing information provided")

    def scenario_complex_transfer(self):
        """Scenario 3: Complex transfer credits - requires escalation"""
        self.print_separator("SCENARIO 3: Complex Transfer Credits (Escalation)")

        print("\n📝 Scenario: Student has credits from multiple schools")
        print("   Expected: Bot provides general info, then escalates to human")

        time.sleep(1)

        # First question
        question1 = "Will my credits transfer?"
        self.print_message("STUDENT", question1)

        response1 = self.send_message(question1)
        self.print_message("BOT", response1["message"], response1["confidence"])

        time.sleep(1)

        # Complex follow-up
        question2 = "I have credits from 3 different colleges and military training from 8 years ago"
        self.print_message("STUDENT", question2)

        response2 = self.send_message(question2)
        self.print_message("BOT", response2["message"], response2["confidence"])

        if response2.get("escalation_suggested"):
            self.print_message("SYSTEM", f"🚨 ESCALATION TRIGGERED: {response2.get('escalation_reason')}")
            print(f"\n   ⚠️  Bot detected complex situation")
            print(f"   🔄 Transferring to enrollment advisor...")

            # Show advisor queue
            queue = self.get_advisor_queue()
            print(f"\n   📊 Advisor Queue: {queue['count']} conversation(s) waiting")

    def scenario_explicit_human_request(self):
        """Scenario 4: Student explicitly asks for human"""
        self.print_separator("SCENARIO 4: Explicit Human Request")

        print("\n📝 Scenario: Student wants to talk to a real person")
        print("   Expected: Immediate escalation to advisor")

        time.sleep(1)

        question = "Can I speak to a real person please?"
        self.print_message("STUDENT", question)

        response = self.send_message(question)
        self.print_message("BOT", response["message"], response["confidence"])

        if response.get("escalation_suggested"):
            self.print_message("SYSTEM", f"🚨 ESCALATION TRIGGERED: {response.get('escalation_reason')}")
            print(f"\n   ✅ Student request honored immediately")

    def scenario_gi_bill(self):
        """Scenario 5: Military/GI Bill question"""
        self.print_separator("SCENARIO 5: Military Benefits (Bot Handles)")

        print("\n📝 Scenario: Veteran asks about GI Bill")
        print("   Expected: Bot provides military benefits information")

        time.sleep(1)

        question = "Do you accept GI Bill for veterans?"
        self.print_message("STUDENT", question)

        response = self.send_message(question)
        self.print_message("BOT", response["message"], response["confidence"])

        print(f"\n   ✅ Bot provided military-specific information")

    def show_final_stats(self):
        """Show final statistics"""
        self.print_separator("FINAL STATISTICS")

        stats = self.get_stats()

        print(f"\n📊 Session Statistics:")
        print(f"   Total Conversations: {stats['total_conversations']}")
        print(f"   Bot Handled: {stats['bot_handled']}")
        print(f"   Escalated to Human: {stats['escalated']}")
        print(f"   Bot Resolution Rate: {stats['bot_resolution_rate']}%")
        print(f"   Current Queue: {stats['current_queue_length']}")

        print(f"\n💡 Insights:")
        if stats['bot_resolution_rate'] >= 70:
            print(f"   ✅ Bot exceeding 70% resolution target")
        if stats['escalated'] > 0:
            print(f"   ⚠️  {stats['escalated']} conversation(s) need human attention")

    def run_all_scenarios(self):
        """Run complete demo"""
        print("\n" + "=" * 80)
        print("  🎓 ENROLLMENT FAQ CHATBOT DEMO")
        print("  With Human-in-the-Loop Functionality")
        print("=" * 80)

        print("\nThis demo shows:")
        print("  ✅ Bot handling simple FAQs automatically")
        print("  ✅ Intelligent escalation for complex queries")
        print("  ✅ Context preservation for advisor handoff")
        print("  ✅ Multi-scenario testing")

        input("\nPress Enter to start the demo...")

        # Run scenarios
        self.scenario_simple_faq()
        input("\n   Press Enter for next scenario...")

        self.scenario_tuition_question()
        input("\n   Press Enter for next scenario...")

        self.scenario_complex_transfer()
        input("\n   Press Enter for next scenario...")

        self.scenario_explicit_human_request()
        input("\n   Press Enter for next scenario...")

        self.scenario_gi_bill()
        input("\n   Press Enter to see final statistics...")

        # Show stats
        self.show_final_stats()

        self.print_separator()
        print("\n✅ Demo Complete!")
        print("\n📁 All conversation data is stored and ready for advisor review")
        print("📊 Advisor dashboard available at: /api/v1/advisor/queue")


def main():
    """Main entry point"""
    print("\n🔍 Checking if chatbot API is running...")

    demo = ChatbotDemo()

    try:
        response = requests.get(f"{demo.base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Chatbot API is running!")
            print(f"   URL: {demo.base_url}")

            # Run demo
            demo.run_all_scenarios()

        else:
            print("❌ API returned unexpected status")
            return

    except requests.ConnectionError:
        print("❌ Cannot connect to chatbot API")
        print("\n📝 To start the API:")
        print("   cd enrollment_chatbot")
        print("   python main.py")
        print("\nThen run this demo script again.")
        return


if __name__ == "__main__":
    main()
