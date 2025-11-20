"""
Interactive CLI for Multi-Agent System

This provides a command-line interface to interact with the agent orchestrator
and execute workflows.
"""

import argparse
import json
import sys
from typing import Optional

from orchestrator import AgentOrchestrator, TaskStatus, WorkflowPhase
from agent_base import create_agent
from agent_definitions import AgentRole


class AgentCLI:
    """Command-line interface for multi-agent system"""

    def __init__(self):
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.agents = {}

    def init_project(self, project_name: str):
        """Initialize a new project with orchestrator and agents"""
        print(f"\n🚀 Initializing project: {project_name}")

        self.orchestrator = AgentOrchestrator(project_name)

        # Create all agents
        self.agents = {
            role: create_agent(role, self.orchestrator)
            for role in AgentRole
        }

        print(f"✓ Created orchestrator and {len(self.agents)} agents")
        print("\nAvailable agents:")
        for role in AgentRole:
            print(f"  - {role.value}")

        return self.orchestrator

    def create_feature_workflow(self, feature_description: str):
        """Generate a complete workflow for a new feature"""
        if not self.orchestrator:
            print("❌ Error: No project initialized. Use 'init' first.")
            return

        print(f"\n📋 Creating workflow for: {feature_description}")

        tasks = self.orchestrator.generate_workflow_for_feature(feature_description)

        print(f"\n✓ Generated {len(tasks)} tasks:")
        for task in tasks:
            deps = f" (depends on: {', '.join(task.dependencies)})" if task.dependencies else ""
            print(f"  {task.id}: {task.title} [{task.assigned_to.value}]{deps}")

        return tasks

    def list_tasks(self, agent_role: Optional[str] = None, status: Optional[str] = None):
        """List all tasks, optionally filtered by agent or status"""
        if not self.orchestrator:
            print("❌ Error: No project initialized.")
            return

        tasks = list(self.orchestrator.tasks.values())

        # Filter by agent role
        if agent_role:
            try:
                role = AgentRole(agent_role)
                tasks = [t for t in tasks if t.assigned_to == role]
            except ValueError:
                print(f"❌ Error: Invalid agent role: {agent_role}")
                return

        # Filter by status
        if status:
            try:
                task_status = TaskStatus(status)
                tasks = [t for t in tasks if t.status == task_status]
            except ValueError:
                print(f"❌ Error: Invalid status: {status}")
                return

        if not tasks:
            print("\n📭 No tasks found matching criteria")
            return

        print(f"\n📋 Found {len(tasks)} tasks:")
        for task in tasks:
            status_emoji = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.IN_PROGRESS: "🔄",
                TaskStatus.BLOCKED: "🚫",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌"
            }.get(task.status, "❓")

            print(f"\n  {status_emoji} [{task.id}] {task.title}")
            print(f"     Agent: {task.assigned_to.value}")
            print(f"     Status: {task.status.value}")
            if task.dependencies:
                print(f"     Dependencies: {', '.join(task.dependencies)}")

    def execute_task(self, task_id: str):
        """Execute a specific task"""
        if not self.orchestrator:
            print("❌ Error: No project initialized.")
            return

        if task_id not in self.orchestrator.tasks:
            print(f"❌ Error: Task {task_id} not found")
            return

        task = self.orchestrator.tasks[task_id]

        # Check if dependencies are met
        ready_tasks = self.orchestrator.get_ready_tasks()
        if task not in ready_tasks:
            print(f"❌ Error: Task {task_id} is not ready (dependencies not met or already completed)")
            return

        print(f"\n🔄 Executing task: {task.title}")
        print(f"   Assigned to: {task.assigned_to.value}")

        # Get the agent
        agent = self.agents[task.assigned_to]

        # Get context
        context = self.orchestrator.get_context_for_agent(task.assigned_to)

        # Execute task
        output = agent.execute_task(task, context)

        # Update orchestrator
        self.orchestrator.update_task_status(task.id, TaskStatus.COMPLETED, output)

        print(f"\n✅ Task completed!")
        print(f"   Deliverable: {output.get('deliverable', 'N/A')}")

        return output

    def execute_ready_tasks(self):
        """Execute all tasks that are ready to run"""
        if not self.orchestrator:
            print("❌ Error: No project initialized.")
            return

        ready_tasks = self.orchestrator.get_ready_tasks()

        if not ready_tasks:
            print("\n📭 No tasks ready to execute")
            return

        print(f"\n🔄 Found {len(ready_tasks)} ready tasks")

        for task in ready_tasks:
            print(f"\n{'='*60}")
            self.execute_task(task.id)

    def show_status(self):
        """Show overall workflow status"""
        if not self.orchestrator:
            print("❌ Error: No project initialized.")
            return

        status = self.orchestrator.get_workflow_status()

        print("\n" + "=" * 60)
        print("WORKFLOW STATUS")
        print("=" * 60)

        print(f"\n📊 Project: {status['project_name']}")
        print(f"📍 Current Phase: {status['current_phase']}")
        print(f"\n📋 Tasks:")
        print(f"   Total: {status['total_tasks']}")
        for task_status, count in status['task_status'].items():
            if count > 0:
                emoji = {
                    'pending': '⏳',
                    'in_progress': '🔄',
                    'blocked': '🚫',
                    'completed': '✅',
                    'failed': '❌'
                }.get(task_status, '❓')
                print(f"   {emoji} {task_status.capitalize()}: {count}")

        print(f"\n🚦 Ready to Execute: {status['ready_tasks']}")
        print(f"💬 Messages Exchanged: {status['total_messages']}")
        print(f"📝 Decisions Made: {status['total_decisions']}")

    def show_decisions(self):
        """Show architectural decisions"""
        if not self.orchestrator:
            print("❌ Error: No project initialized.")
            return

        decisions = self.orchestrator.context.decisions

        if not decisions:
            print("\n📭 No decisions recorded yet")
            return

        print("\n" + "=" * 60)
        print("ARCHITECTURAL DECISIONS")
        print("=" * 60)

        for decision in decisions:
            print(f"\n📝 [{decision['id']}] {decision['title']}")
            print(f"   Decided by: {decision['decided_by']}")
            print(f"   Description: {decision['description']}")
            print(f"   Rationale: {decision['rationale']}")
            print(f"   Timestamp: {decision['timestamp']}")

    def save_state(self, filepath: str):
        """Save orchestrator state to file"""
        if not self.orchestrator:
            print("❌ Error: No project initialized.")
            return

        self.orchestrator.save_state(filepath)
        print(f"\n✅ State saved to: {filepath}")

    def export_summary(self, filepath: str):
        """Export workflow summary"""
        if not self.orchestrator:
            print("❌ Error: No project initialized.")
            return

        state = self.orchestrator.export_state()

        # Create human-readable summary
        summary = {
            "project": state["project_name"],
            "phase": state["context"]["current_phase"],
            "summary": state["workflow_status"],
            "tasks": [
                {
                    "id": t["id"],
                    "title": t["title"],
                    "agent": t["assigned_to"],
                    "status": t["status"]
                }
                for t in state["tasks"].values()
            ],
            "decisions": state["context"]["decisions"]
        }

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Summary exported to: {filepath}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Product Development System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize a new project
  python cli.py init "AI Student Success App - New Feature"

  # Create a workflow for a feature
  python cli.py workflow "Real-time alert system for at-risk students"

  # List all tasks
  python cli.py list

  # List tasks for specific agent
  python cli.py list --agent product_manager

  # Execute all ready tasks
  python cli.py execute

  # Show workflow status
  python cli.py status

  # Show decisions
  python cli.py decisions

  # Save state
  python cli.py save workflow_state.json
        """
    )

    parser.add_argument('command', choices=['init', 'workflow', 'list', 'execute', 'status', 'decisions', 'save', 'export'])
    parser.add_argument('value', nargs='?', help='Value for the command (project name, feature description, file path)')
    parser.add_argument('--agent', help='Filter by agent role')
    parser.add_argument('--status', help='Filter by task status')

    args = parser.parse_args()

    cli = AgentCLI()

    if args.command == 'init':
        if not args.value:
            print("❌ Error: Project name required")
            sys.exit(1)
        cli.init_project(args.value)

    elif args.command == 'workflow':
        if not args.value:
            print("❌ Error: Feature description required")
            sys.exit(1)
        # Need to init first
        cli.init_project("Current Project")
        cli.create_feature_workflow(args.value)

    elif args.command == 'list':
        cli.init_project("Current Project")
        cli.list_tasks(args.agent, args.status)

    elif args.command == 'execute':
        cli.init_project("Current Project")
        cli.execute_ready_tasks()

    elif args.command == 'status':
        cli.init_project("Current Project")
        cli.show_status()

    elif args.command == 'decisions':
        cli.init_project("Current Project")
        cli.show_decisions()

    elif args.command == 'save':
        if not args.value:
            print("❌ Error: File path required")
            sys.exit(1)
        cli.init_project("Current Project")
        cli.save_state(args.value)

    elif args.command == 'export':
        if not args.value:
            print("❌ Error: File path required")
            sys.exit(1)
        cli.init_project("Current Project")
        cli.export_summary(args.value)


if __name__ == "__main__":
    main()
