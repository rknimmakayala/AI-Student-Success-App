"""
Multi-Agent Product Development System

A framework for coordinating specialized AI agents to collaboratively
build software features through defined workflows.
"""

from .agent_definitions import AgentRole, get_agent_definition
from .orchestrator import AgentOrchestrator, Task, TaskStatus, WorkflowPhase
from .agent_base import BaseAgent, create_agent

__version__ = "1.0.0"

__all__ = [
    'AgentRole',
    'AgentOrchestrator',
    'BaseAgent',
    'Task',
    'TaskStatus',
    'WorkflowPhase',
    'create_agent',
    'get_agent_definition'
]
