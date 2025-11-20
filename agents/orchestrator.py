"""
Agent Orchestrator - Coordinates multi-agent collaboration

The orchestrator manages communication between agents, delegates tasks,
tracks state, and ensures workflow progression.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from agent_definitions import AgentRole, get_agent_definition, get_agent_system_prompt


class TaskStatus(Enum):
    """Status of a task in the workflow"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowPhase(Enum):
    """Phases of the product development workflow"""
    DISCOVERY = "discovery"
    ARCHITECTURE = "architecture"
    PLANNING = "planning"
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    REVIEW = "review"


@dataclass
class Task:
    """Represents a task assigned to an agent"""
    id: str
    title: str
    description: str
    assigned_to: AgentRole
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    output: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert task to dictionary"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "assigned_to": self.assigned_to.value,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "output": self.output,
            "metadata": self.metadata
        }


@dataclass
class AgentMessage:
    """Message from one agent to another or to the orchestrator"""
    from_agent: AgentRole
    to_agent: Optional[AgentRole]
    message_type: str  # "task_complete", "question", "update", "decision"
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert message to dictionary"""
        return {
            "from_agent": self.from_agent.value,
            "to_agent": self.to_agent.value if self.to_agent else None,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class WorkflowContext:
    """Shared context accessible to all agents"""
    project_name: str
    current_phase: WorkflowPhase
    requirements: Dict[str, Any] = field(default_factory=dict)
    architecture: Dict[str, Any] = field(default_factory=dict)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    success_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert context to dictionary"""
        return {
            "project_name": self.project_name,
            "current_phase": self.current_phase.value,
            "requirements": self.requirements,
            "architecture": self.architecture,
            "decisions": self.decisions,
            "constraints": self.constraints,
            "success_metrics": self.success_metrics,
            "metadata": self.metadata
        }


class AgentOrchestrator:
    """
    Orchestrates multi-agent collaboration for product development

    The orchestrator:
    1. Manages workflow phases
    2. Delegates tasks to appropriate agents
    3. Tracks task dependencies and completion
    4. Facilitates inter-agent communication
    5. Maintains shared context and state
    """

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.context = WorkflowContext(
            project_name=project_name,
            current_phase=WorkflowPhase.DISCOVERY
        )
        self.tasks: Dict[str, Task] = {}
        self.messages: List[AgentMessage] = []
        self.task_counter = 0

    def create_task(
        self,
        title: str,
        description: str,
        assigned_to: AgentRole,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Create a new task and add it to the workflow"""
        self.task_counter += 1
        task_id = f"task_{self.task_counter:04d}"

        task = Task(
            id=task_id,
            title=title,
            description=description,
            assigned_to=assigned_to,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )

        self.tasks[task_id] = task
        return task

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        output: Optional[Dict[str, Any]] = None
    ):
        """Update the status of a task"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        task.status = status
        task.updated_at = datetime.now()

        if output:
            task.output = output

    def get_ready_tasks(self, agent_role: Optional[AgentRole] = None) -> List[Task]:
        """Get tasks that are ready to be worked on (dependencies met)"""
        ready_tasks = []

        for task in self.tasks.values():
            # Skip if not pending
            if task.status != TaskStatus.PENDING:
                continue

            # Skip if not for this agent (when agent_role is specified)
            if agent_role and task.assigned_to != agent_role:
                continue

            # Check if all dependencies are completed
            dependencies_met = all(
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
                if dep_id in self.tasks
            )

            if dependencies_met:
                ready_tasks.append(task)

        return ready_tasks

    def send_message(
        self,
        from_agent: AgentRole,
        message_type: str,
        content: Dict[str, Any],
        to_agent: Optional[AgentRole] = None
    ):
        """Send a message from one agent to another or to orchestrator"""
        message = AgentMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            content=content
        )
        self.messages.append(message)
        return message

    def get_messages_for_agent(
        self,
        agent_role: AgentRole,
        message_type: Optional[str] = None
    ) -> List[AgentMessage]:
        """Get messages intended for a specific agent"""
        messages = [
            msg for msg in self.messages
            if msg.to_agent == agent_role
        ]

        if message_type:
            messages = [msg for msg in messages if msg.message_type == message_type]

        return messages

    def advance_phase(self, next_phase: WorkflowPhase):
        """Move to the next phase of the workflow"""
        self.context.current_phase = next_phase

    def add_decision(
        self,
        title: str,
        description: str,
        rationale: str,
        decided_by: AgentRole,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record an architectural or product decision (ADR)"""
        decision = {
            "id": f"decision_{len(self.context.decisions) + 1:04d}",
            "title": title,
            "description": description,
            "rationale": rationale,
            "decided_by": decided_by.value,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.context.decisions.append(decision)
        return decision

    def get_context_for_agent(self, agent_role: AgentRole) -> Dict[str, Any]:
        """Get relevant context for a specific agent"""
        agent_def = get_agent_definition(agent_role)

        # Base context available to all agents
        context = {
            "project_name": self.context.project_name,
            "current_phase": self.context.current_phase.value,
            "agent_role": agent_role.value,
            "agent_expertise": agent_def["expertise"],
            "constraints": self.context.constraints,
        }

        # Add role-specific context
        if agent_role == AgentRole.PRODUCT_MANAGER:
            context.update({
                "requirements": self.context.requirements,
                "success_metrics": self.context.success_metrics
            })

        elif agent_role == AgentRole.SYSTEM_ARCHITECT:
            context.update({
                "requirements": self.context.requirements,
                "architecture": self.context.architecture,
                "decisions": self.context.decisions
            })

        elif agent_role in [AgentRole.FRONTEND_ENGINEER, AgentRole.BACKEND_ENGINEER, AgentRole.AI_ENGINEER]:
            context.update({
                "architecture": self.context.architecture,
                "relevant_decisions": [
                    d for d in self.context.decisions
                    if agent_role.value in d.get("metadata", {}).get("affects", [])
                ]
            })

        elif agent_role == AgentRole.QA_ENGINEER:
            context.update({
                "requirements": self.context.requirements,
                "acceptance_criteria": self.context.requirements.get("acceptance_criteria", [])
            })

        elif agent_role == AgentRole.DEVOPS_ENGINEER:
            context.update({
                "architecture": self.context.architecture,
                "deployment_requirements": self.context.metadata.get("deployment", {})
            })

        return context

    def generate_workflow_for_feature(
        self,
        feature_description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Task]:
        """
        Generate a standard workflow of tasks for a new feature

        This creates the typical task flow:
        Discovery -> Architecture -> Planning -> Development -> Testing -> Deployment
        """
        tasks = []

        # 1. Discovery Phase - PM analyzes requirements
        task1 = self.create_task(
            title="Analyze Feature Requirements",
            description=f"Analyze requirements for: {feature_description}",
            assigned_to=AgentRole.PRODUCT_MANAGER,
            metadata={"phase": WorkflowPhase.DISCOVERY.value, **(metadata or {})}
        )
        tasks.append(task1)

        # 2. Architecture Phase - Architect designs solution
        task2 = self.create_task(
            title="Design Architecture",
            description=f"Design system architecture for: {feature_description}",
            assigned_to=AgentRole.SYSTEM_ARCHITECT,
            dependencies=[task1.id],
            metadata={"phase": WorkflowPhase.ARCHITECTURE.value}
        )
        tasks.append(task2)

        # 3. Planning Phase - QA creates test plan
        task3 = self.create_task(
            title="Create Test Plan",
            description=f"Design test strategy for: {feature_description}",
            assigned_to=AgentRole.QA_ENGINEER,
            dependencies=[task1.id, task2.id],
            metadata={"phase": WorkflowPhase.PLANNING.value}
        )
        tasks.append(task3)

        # 4. Development Phase - Engineers implement
        task4_frontend = self.create_task(
            title="Implement Frontend",
            description=f"Build UI components for: {feature_description}",
            assigned_to=AgentRole.FRONTEND_ENGINEER,
            dependencies=[task2.id],
            metadata={"phase": WorkflowPhase.DEVELOPMENT.value}
        )
        tasks.append(task4_frontend)

        task4_backend = self.create_task(
            title="Implement Backend",
            description=f"Build API and logic for: {feature_description}",
            assigned_to=AgentRole.BACKEND_ENGINEER,
            dependencies=[task2.id],
            metadata={"phase": WorkflowPhase.DEVELOPMENT.value}
        )
        tasks.append(task4_backend)

        # 5. AI Implementation (if needed)
        task4_ai = self.create_task(
            title="Implement AI Components",
            description=f"Design and implement AI reasoning for: {feature_description}",
            assigned_to=AgentRole.AI_ENGINEER,
            dependencies=[task2.id],
            metadata={"phase": WorkflowPhase.DEVELOPMENT.value}
        )
        tasks.append(task4_ai)

        # 6. Testing Phase - QA validates
        task5 = self.create_task(
            title="Execute Tests",
            description=f"Test implementation of: {feature_description}",
            assigned_to=AgentRole.QA_ENGINEER,
            dependencies=[task4_frontend.id, task4_backend.id, task4_ai.id],
            metadata={"phase": WorkflowPhase.TESTING.value}
        )
        tasks.append(task5)

        # 7. Deployment Phase - DevOps deploys
        task6 = self.create_task(
            title="Deploy Feature",
            description=f"Deploy and monitor: {feature_description}",
            assigned_to=AgentRole.DEVOPS_ENGINEER,
            dependencies=[task5.id],
            metadata={"phase": WorkflowPhase.DEPLOYMENT.value}
        )
        tasks.append(task6)

        return tasks

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current status of the workflow"""
        status_counts = {status: 0 for status in TaskStatus}
        for task in self.tasks.values():
            status_counts[task.status] += 1

        return {
            "project_name": self.project_name,
            "current_phase": self.context.current_phase.value,
            "total_tasks": len(self.tasks),
            "task_status": {status.value: count for status, count in status_counts.items()},
            "ready_tasks": len(self.get_ready_tasks()),
            "total_messages": len(self.messages),
            "total_decisions": len(self.context.decisions)
        }

    def export_state(self) -> Dict[str, Any]:
        """Export complete orchestrator state"""
        return {
            "project_name": self.project_name,
            "context": self.context.to_dict(),
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "messages": [msg.to_dict() for msg in self.messages],
            "workflow_status": self.get_workflow_status()
        }

    def save_state(self, filepath: str):
        """Save orchestrator state to file"""
        with open(filepath, 'w') as f:
            json.dump(self.export_state(), f, indent=2)

    def generate_agent_prompt(self, agent_role: AgentRole, task: Task) -> str:
        """
        Generate a complete prompt for an agent to work on a task

        This combines:
        - Agent system prompt
        - Current context
        - Task description
        - Relevant dependencies
        """
        system_prompt = get_agent_system_prompt(agent_role)
        context = self.get_context_for_agent(agent_role)

        # Get outputs from dependency tasks
        dependency_outputs = []
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
                if dep_task.output:
                    dependency_outputs.append({
                        "task": dep_task.title,
                        "output": dep_task.output
                    })

        prompt = f"""{system_prompt}

## Current Context

{json.dumps(context, indent=2)}

## Your Task

**Task ID**: {task.id}
**Title**: {task.title}
**Description**: {task.description}

## Dependency Outputs

{json.dumps(dependency_outputs, indent=2) if dependency_outputs else "No dependencies"}

## Instructions

Please complete this task according to your role responsibilities. Provide your output in a structured format that can be used by downstream agents.

Your output should include:
1. Summary of what you accomplished
2. Key decisions or recommendations
3. Any artifacts (code, diagrams, specifications, etc.)
4. Questions or blockers for other agents
5. Next steps or handoff instructions
"""

        return prompt


# Example usage and testing
if __name__ == "__main__":
    # Create orchestrator for a new feature
    orchestrator = AgentOrchestrator("AI Student Success App - Alert System")

    # Generate workflow for a new feature
    feature_description = "Real-time alert system for at-risk students"
    tasks = orchestrator.generate_workflow_for_feature(feature_description)

    print("=" * 80)
    print(f"Generated Workflow for: {feature_description}")
    print("=" * 80)

    for task in tasks:
        print(f"\n[{task.id}] {task.title}")
        print(f"  Assigned to: {task.assigned_to.value}")
        print(f"  Status: {task.status.value}")
        print(f"  Dependencies: {task.dependencies if task.dependencies else 'None'}")

    print("\n" + "=" * 80)
    print("Workflow Status")
    print("=" * 80)
    status = orchestrator.get_workflow_status()
    print(json.dumps(status, indent=2))

    # Show ready tasks
    print("\n" + "=" * 80)
    print("Ready Tasks (can start now)")
    print("=" * 80)
    ready_tasks = orchestrator.get_ready_tasks()
    for task in ready_tasks:
        print(f"  - {task.title} ({task.assigned_to.value})")
