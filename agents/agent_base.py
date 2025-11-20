"""
Base Agent class and implementations for the multi-agent system

Each agent has memory, can execute tasks, and communicate with other agents.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import json

from agent_definitions import AgentRole, get_agent_definition, get_agent_system_prompt
from orchestrator import Task, AgentMessage, WorkflowContext


@dataclass
class AgentMemory:
    """Memory system for an agent"""
    short_term: List[Dict[str, Any]] = field(default_factory=list)  # Recent interactions
    long_term: Dict[str, Any] = field(default_factory=dict)  # Persistent knowledge
    context_window: int = 10  # Number of recent items to keep in short-term

    def add_to_short_term(self, item: Dict[str, Any]):
        """Add item to short-term memory, maintaining window size"""
        self.short_term.append(item)
        if len(self.short_term) > self.context_window:
            # Move oldest to long-term if important
            oldest = self.short_term.pop(0)
            if oldest.get("important", False):
                key = oldest.get("key", f"memory_{datetime.now().isoformat()}")
                self.long_term[key] = oldest

    def add_to_long_term(self, key: str, value: Any):
        """Add item to long-term memory"""
        self.long_term[key] = value

    def recall(self, key: str) -> Optional[Any]:
        """Recall from long-term memory"""
        return self.long_term.get(key)

    def get_recent_context(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent short-term memory"""
        if limit:
            return self.short_term[-limit:]
        return self.short_term


class BaseAgent(ABC):
    """
    Base class for all agents in the system

    Each agent:
    - Has a specific role and expertise
    - Maintains memory (short-term and long-term)
    - Can execute tasks
    - Can communicate with other agents via orchestrator
    """

    def __init__(self, role: AgentRole, orchestrator=None):
        self.role = role
        self.orchestrator = orchestrator
        self.memory = AgentMemory()
        self.definition = get_agent_definition(role)
        self.system_prompt = get_agent_system_prompt(role)
        self.task_history: List[str] = []

    @abstractmethod
    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent

        Args:
            task: The task to execute
            context: Current workflow context

        Returns:
            Dict containing task output
        """
        pass

    def communicate(
        self,
        message_type: str,
        content: Dict[str, Any],
        to_agent: Optional[AgentRole] = None
    ):
        """Send a message to another agent or the orchestrator"""
        if self.orchestrator:
            return self.orchestrator.send_message(
                from_agent=self.role,
                message_type=message_type,
                content=content,
                to_agent=to_agent
            )

    def remember(self, key: str, value: Any, important: bool = False):
        """Store information in memory"""
        if important:
            self.memory.add_to_long_term(key, value)
        else:
            self.memory.add_to_short_term({
                "key": key,
                "value": value,
                "timestamp": datetime.now().isoformat(),
                "important": important
            })

    def recall(self, key: str) -> Optional[Any]:
        """Retrieve information from memory"""
        return self.memory.recall(key)

    def get_context(self) -> Dict[str, Any]:
        """Get current context for this agent"""
        if self.orchestrator:
            return self.orchestrator.get_context_for_agent(self.role)
        return {}

    def log_task_completion(self, task: Task):
        """Log that a task was completed"""
        self.task_history.append(task.id)
        self.remember(
            f"task_{task.id}",
            {
                "title": task.title,
                "completed_at": datetime.now().isoformat(),
                "output_summary": str(task.output)[:200] if task.output else None
            },
            important=True
        )


class ProductManagerAgent(BaseAgent):
    """Product Manager / Analyst Agent"""

    def __init__(self, orchestrator=None):
        super().__init__(AgentRole.PRODUCT_MANAGER, orchestrator)

    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute PM tasks: requirements, acceptance criteria, prioritization"""

        # This is a template - in real implementation, this would call an LLM
        # or use other AI capabilities to analyze and generate requirements

        output = {
            "agent": self.role.value,
            "task_id": task.id,
            "task_title": task.title,
        }

        if "analyze" in task.title.lower() or "requirements" in task.title.lower():
            output.update(self._analyze_requirements(task, context))

        elif "acceptance" in task.title.lower() or "criteria" in task.title.lower():
            output.update(self._define_acceptance_criteria(task, context))

        elif "prioritize" in task.title.lower():
            output.update(self._prioritize_backlog(task, context))

        else:
            output["note"] = "Generic PM task execution"

        self.log_task_completion(task)
        return output

    def _analyze_requirements(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and document requirements"""
        return {
            "deliverable": "requirements_document",
            "user_problem": "Extracted from task description",
            "requirements": [
                "Requirement 1: [Details from analysis]",
                "Requirement 2: [Details from analysis]"
            ],
            "constraints": context.get("constraints", []),
            "success_metrics": {
                "metric_1": "Description",
                "metric_2": "Description"
            },
            "next_steps": [
                "System Architect: Design architecture",
                "QA Engineer: Create test plan"
            ]
        }

    def _define_acceptance_criteria(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Define acceptance criteria"""
        return {
            "deliverable": "acceptance_criteria",
            "criteria": [
                "Given [context], when [action], then [outcome]",
                "Given [context], when [action], then [outcome]"
            ]
        }

    def _prioritize_backlog(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prioritize features"""
        return {
            "deliverable": "prioritized_backlog",
            "priorities": [
                {"feature": "Feature 1", "priority": "P0", "rationale": "..."},
                {"feature": "Feature 2", "priority": "P1", "rationale": "..."}
            ]
        }


class SystemArchitectAgent(BaseAgent):
    """System Architect Agent"""

    def __init__(self, orchestrator=None):
        super().__init__(AgentRole.SYSTEM_ARCHITECT, orchestrator)

    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute architecture tasks: design, API specs, risk assessment"""

        output = {
            "agent": self.role.value,
            "task_id": task.id,
            "task_title": task.title,
        }

        if "design" in task.title.lower() or "architecture" in task.title.lower():
            output.update(self._design_architecture(task, context))

        elif "api" in task.title.lower():
            output.update(self._define_api_contracts(task, context))

        elif "risk" in task.title.lower():
            output.update(self._assess_risks(task, context))

        else:
            output["note"] = "Generic architecture task execution"

        self.log_task_completion(task)
        return output

    def _design_architecture(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Design system architecture"""
        return {
            "deliverable": "architecture_design",
            "components": [
                {
                    "name": "Component A",
                    "description": "Purpose and responsibilities",
                    "technology": "Recommended tech stack"
                },
                {
                    "name": "Component B",
                    "description": "Purpose and responsibilities",
                    "technology": "Recommended tech stack"
                }
            ],
            "data_flow": "Description of how data flows through the system",
            "integration_points": [
                "Integration 1: Details",
                "Integration 2: Details"
            ],
            "adr": {
                "title": "Architectural Decision",
                "decision": "What was decided",
                "rationale": "Why this decision was made"
            },
            "next_steps": [
                "Frontend Engineer: Implement UI components",
                "Backend Engineer: Implement APIs",
                "DevOps: Set up infrastructure"
            ]
        }

    def _define_api_contracts(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Define API specifications"""
        return {
            "deliverable": "api_specification",
            "endpoints": [
                {
                    "path": "/api/endpoint",
                    "method": "GET",
                    "description": "What this endpoint does",
                    "request": {"schema": "..."},
                    "response": {"schema": "..."}
                }
            ]
        }

    def _assess_risks(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess technical risks"""
        return {
            "deliverable": "risk_assessment",
            "risks": [
                {
                    "risk": "Description",
                    "severity": "High/Medium/Low",
                    "mitigation": "How to mitigate"
                }
            ]
        }


class FrontendEngineerAgent(BaseAgent):
    """Frontend Engineer Agent"""

    def __init__(self, orchestrator=None):
        super().__init__(AgentRole.FRONTEND_ENGINEER, orchestrator)

    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute frontend tasks: UI components, flows, optimization"""

        output = {
            "agent": self.role.value,
            "task_id": task.id,
            "task_title": task.title,
            "deliverable": "frontend_implementation",
            "components": [],
            "code_files": [],
            "next_steps": ["QA Engineer: Test UI components"]
        }

        self.log_task_completion(task)
        return output


class BackendEngineerAgent(BaseAgent):
    """Backend Engineer Agent"""

    def __init__(self, orchestrator=None):
        super().__init__(AgentRole.BACKEND_ENGINEER, orchestrator)

    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute backend tasks: APIs, data models, business logic"""

        output = {
            "agent": self.role.value,
            "task_id": task.id,
            "task_title": task.title,
            "deliverable": "backend_implementation",
            "endpoints": [],
            "models": [],
            "code_files": [],
            "next_steps": ["QA Engineer: Test API endpoints"]
        }

        self.log_task_completion(task)
        return output


class AIEngineerAgent(BaseAgent):
    """AI Engineer Agent"""

    def __init__(self, orchestrator=None):
        super().__init__(AgentRole.AI_ENGINEER, orchestrator)

    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI tasks: reasoning chains, prompts, evaluation"""

        output = {
            "agent": self.role.value,
            "task_id": task.id,
            "task_title": task.title,
            "deliverable": "ai_implementation",
            "reasoning_chains": [],
            "prompts": [],
            "evaluation_metrics": [],
            "next_steps": ["QA Engineer: Test AI outputs"]
        }

        self.log_task_completion(task)
        return output


class QAEngineerAgent(BaseAgent):
    """QA Engineer Agent"""

    def __init__(self, orchestrator=None):
        super().__init__(AgentRole.QA_ENGINEER, orchestrator)

    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute QA tasks: test plans, test implementation, validation"""

        output = {
            "agent": self.role.value,
            "task_id": task.id,
            "task_title": task.title,
        }

        if "test plan" in task.title.lower() or "strategy" in task.title.lower():
            output.update(self._create_test_plan(task, context))

        elif "execute" in task.title.lower() or "test" in task.title.lower():
            output.update(self._execute_tests(task, context))

        else:
            output["note"] = "Generic QA task execution"

        self.log_task_completion(task)
        return output

    def _create_test_plan(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive test plan"""
        return {
            "deliverable": "test_plan",
            "test_strategy": "Overall testing approach",
            "test_cases": [
                {
                    "id": "TC001",
                    "description": "Test case description",
                    "steps": ["Step 1", "Step 2"],
                    "expected": "Expected outcome"
                }
            ],
            "edge_cases": [
                "Edge case 1",
                "Edge case 2"
            ],
            "automation_plan": "How tests will be automated"
        }

    def _execute_tests(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tests and report results"""
        return {
            "deliverable": "test_results",
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "failures": [],
            "next_steps": ["DevOps: Deploy if all tests pass"]
        }


class DevOpsEngineerAgent(BaseAgent):
    """DevOps Engineer Agent"""

    def __init__(self, orchestrator=None):
        super().__init__(AgentRole.DEVOPS_ENGINEER, orchestrator)

    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DevOps tasks: CI/CD, deployment, monitoring"""

        output = {
            "agent": self.role.value,
            "task_id": task.id,
            "task_title": task.title,
            "deliverable": "deployment",
            "pipeline": "CI/CD pipeline configuration",
            "environments": [],
            "monitoring": "Monitoring setup",
            "status": "deployed"
        }

        self.log_task_completion(task)
        return output


# Agent factory
def create_agent(role: AgentRole, orchestrator=None) -> BaseAgent:
    """Factory function to create agents"""
    agent_classes = {
        AgentRole.PRODUCT_MANAGER: ProductManagerAgent,
        AgentRole.SYSTEM_ARCHITECT: SystemArchitectAgent,
        AgentRole.FRONTEND_ENGINEER: FrontendEngineerAgent,
        AgentRole.BACKEND_ENGINEER: BackendEngineerAgent,
        AgentRole.AI_ENGINEER: AIEngineerAgent,
        AgentRole.QA_ENGINEER: QAEngineerAgent,
        AgentRole.DEVOPS_ENGINEER: DevOpsEngineerAgent
    }

    agent_class = agent_classes.get(role)
    if not agent_class:
        raise ValueError(f"Unknown agent role: {role}")

    return agent_class(orchestrator)
