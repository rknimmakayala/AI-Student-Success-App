"""
Example Workflow: Real-time Alert System for At-Risk Students

This demonstrates how the multi-agent system collaborates to build a new feature
from requirements through deployment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import AgentOrchestrator, TaskStatus, WorkflowPhase
from agent_base import create_agent
from agent_definitions import AgentRole
import json


def run_alert_system_workflow():
    """
    Complete workflow for building a real-time alert system

    This example shows:
    1. How agents are coordinated
    2. How tasks flow from discovery to deployment
    3. How agents communicate
    4. How context is shared and evolved
    """

    print("=" * 100)
    print("MULTI-AGENT WORKFLOW EXAMPLE")
    print("Feature: Real-time Alert System for At-Risk Students")
    print("=" * 100)

    # Initialize orchestrator
    orchestrator = AgentOrchestrator("AI Student Success App - Alert System")

    # Create all agents
    agents = {
        role: create_agent(role, orchestrator)
        for role in AgentRole
    }

    print("\n✓ Initialized orchestrator and 7 specialized agents")

    # =========================================================================
    # PHASE 1: DISCOVERY
    # =========================================================================
    print("\n" + "=" * 100)
    print("PHASE 1: DISCOVERY - Product Manager analyzes requirements")
    print("=" * 100)

    orchestrator.advance_phase(WorkflowPhase.DISCOVERY)

    # PM analyzes feature requirements
    pm_task = orchestrator.create_task(
        title="Analyze Alert System Requirements",
        description="""Analyze requirements for a real-time alert system that:
        - Monitors student engagement metrics in real-time
        - Triggers alerts when students show at-risk patterns
        - Notifies advisors and counselors
        - Provides actionable intervention recommendations
        """,
        assigned_to=AgentRole.PRODUCT_MANAGER
    )

    print(f"\n📋 Created Task: {pm_task.title}")
    print(f"   Assigned to: {pm_task.assigned_to.value}")

    # Execute PM task
    pm_agent = agents[AgentRole.PRODUCT_MANAGER]
    context = orchestrator.get_context_for_agent(AgentRole.PRODUCT_MANAGER)
    pm_output = pm_agent.execute_task(pm_task, context)

    # Update orchestrator with PM's output
    orchestrator.update_task_status(pm_task.id, TaskStatus.COMPLETED, pm_output)
    orchestrator.context.requirements = pm_output

    print(f"\n✓ PM Task Completed")
    print(f"   Deliverable: {pm_output.get('deliverable', 'N/A')}")
    print(f"   Requirements Count: {len(pm_output.get('requirements', []))}")

    # =========================================================================
    # PHASE 2: ARCHITECTURE
    # =========================================================================
    print("\n" + "=" * 100)
    print("PHASE 2: ARCHITECTURE - System Architect designs solution")
    print("=" * 100)

    orchestrator.advance_phase(WorkflowPhase.ARCHITECTURE)

    # Architect designs system
    arch_task = orchestrator.create_task(
        title="Design Alert System Architecture",
        description="Design architecture for real-time alert system with streaming data processing",
        assigned_to=AgentRole.SYSTEM_ARCHITECT,
        dependencies=[pm_task.id]
    )

    print(f"\n🏗️  Created Task: {arch_task.title}")
    print(f"   Assigned to: {arch_task.assigned_to.value}")
    print(f"   Dependencies: {arch_task.dependencies}")

    # Execute architecture task
    arch_agent = agents[AgentRole.SYSTEM_ARCHITECT]
    context = orchestrator.get_context_for_agent(AgentRole.SYSTEM_ARCHITECT)
    arch_output = arch_agent.execute_task(arch_task, context)

    orchestrator.update_task_status(arch_task.id, TaskStatus.COMPLETED, arch_output)
    orchestrator.context.architecture = arch_output

    # Record architectural decision
    orchestrator.add_decision(
        title="Use Event-Driven Architecture for Alerts",
        description="Implement alert system using event streaming for real-time processing",
        rationale="Event-driven architecture provides scalability and real-time capabilities needed for alert system",
        decided_by=AgentRole.SYSTEM_ARCHITECT,
        metadata={
            "affects": ["backend_engineer", "ai_engineer", "devops_engineer"]
        }
    )

    print(f"\n✓ Architecture Task Completed")
    print(f"   Deliverable: {arch_output.get('deliverable', 'N/A')}")
    print(f"   Components: {len(arch_output.get('components', []))}")
    print(f"   ADR Recorded: {orchestrator.context.decisions[-1]['title']}")

    # =========================================================================
    # PHASE 3: PLANNING
    # =========================================================================
    print("\n" + "=" * 100)
    print("PHASE 3: PLANNING - QA Engineer creates test plan")
    print("=" * 100)

    orchestrator.advance_phase(WorkflowPhase.PLANNING)

    # QA creates test plan
    qa_plan_task = orchestrator.create_task(
        title="Create Alert System Test Plan",
        description="Design comprehensive test strategy for alert system",
        assigned_to=AgentRole.QA_ENGINEER,
        dependencies=[pm_task.id, arch_task.id]
    )

    print(f"\n🧪 Created Task: {qa_plan_task.title}")
    print(f"   Assigned to: {qa_plan_task.assigned_to.value}")

    # Execute QA planning task
    qa_agent = agents[AgentRole.QA_ENGINEER]
    context = orchestrator.get_context_for_agent(AgentRole.QA_ENGINEER)
    qa_plan_output = qa_agent.execute_task(qa_plan_task, context)

    orchestrator.update_task_status(qa_plan_task.id, TaskStatus.COMPLETED, qa_plan_output)

    print(f"\n✓ QA Planning Task Completed")
    print(f"   Deliverable: {qa_plan_output.get('deliverable', 'N/A')}")
    print(f"   Test Cases: {len(qa_plan_output.get('test_cases', []))}")
    print(f"   Edge Cases: {len(qa_plan_output.get('edge_cases', []))}")

    # =========================================================================
    # PHASE 4: DEVELOPMENT
    # =========================================================================
    print("\n" + "=" * 100)
    print("PHASE 4: DEVELOPMENT - Engineers implement the feature")
    print("=" * 100)

    orchestrator.advance_phase(WorkflowPhase.DEVELOPMENT)

    # Create development tasks
    frontend_task = orchestrator.create_task(
        title="Implement Alert Dashboard UI",
        description="Build UI for viewing and managing alerts",
        assigned_to=AgentRole.FRONTEND_ENGINEER,
        dependencies=[arch_task.id]
    )

    backend_task = orchestrator.create_task(
        title="Implement Alert API and Processing",
        description="Build API endpoints and alert processing logic",
        assigned_to=AgentRole.BACKEND_ENGINEER,
        dependencies=[arch_task.id]
    )

    ai_task = orchestrator.create_task(
        title="Implement Risk Detection AI",
        description="Design AI model to detect at-risk student patterns",
        assigned_to=AgentRole.AI_ENGINEER,
        dependencies=[arch_task.id]
    )

    print(f"\n💻 Created Development Tasks:")
    print(f"   - {frontend_task.title} ({frontend_task.assigned_to.value})")
    print(f"   - {backend_task.title} ({backend_task.assigned_to.value})")
    print(f"   - {ai_task.title} ({ai_task.assigned_to.value})")

    # Execute development tasks (in parallel in real scenario)
    for task, agent_role in [
        (frontend_task, AgentRole.FRONTEND_ENGINEER),
        (backend_task, AgentRole.BACKEND_ENGINEER),
        (ai_task, AgentRole.AI_ENGINEER)
    ]:
        agent = agents[agent_role]
        context = orchestrator.get_context_for_agent(agent_role)
        output = agent.execute_task(task, context)
        orchestrator.update_task_status(task.id, TaskStatus.COMPLETED, output)
        print(f"   ✓ {task.title} completed")

    # =========================================================================
    # PHASE 5: TESTING
    # =========================================================================
    print("\n" + "=" * 100)
    print("PHASE 5: TESTING - QA Engineer validates implementation")
    print("=" * 100)

    orchestrator.advance_phase(WorkflowPhase.TESTING)

    # QA executes tests
    qa_test_task = orchestrator.create_task(
        title="Execute Alert System Tests",
        description="Run comprehensive tests on alert system implementation",
        assigned_to=AgentRole.QA_ENGINEER,
        dependencies=[frontend_task.id, backend_task.id, ai_task.id]
    )

    print(f"\n🧪 Created Task: {qa_test_task.title}")

    # Execute testing task
    context = orchestrator.get_context_for_agent(AgentRole.QA_ENGINEER)
    qa_test_output = qa_agent.execute_task(qa_test_task, context)

    orchestrator.update_task_status(qa_test_task.id, TaskStatus.COMPLETED, qa_test_output)

    print(f"\n✓ Testing Task Completed")
    print(f"   Tests Run: {qa_test_output.get('tests_run', 0)}")
    print(f"   Tests Passed: {qa_test_output.get('tests_passed', 0)}")
    print(f"   Tests Failed: {qa_test_output.get('tests_failed', 0)}")

    # =========================================================================
    # PHASE 6: DEPLOYMENT
    # =========================================================================
    print("\n" + "=" * 100)
    print("PHASE 6: DEPLOYMENT - DevOps deploys and monitors")
    print("=" * 100)

    orchestrator.advance_phase(WorkflowPhase.DEPLOYMENT)

    # DevOps deploys
    deploy_task = orchestrator.create_task(
        title="Deploy Alert System",
        description="Deploy alert system to production with monitoring",
        assigned_to=AgentRole.DEVOPS_ENGINEER,
        dependencies=[qa_test_task.id]
    )

    print(f"\n🚀 Created Task: {deploy_task.title}")

    # Execute deployment task
    devops_agent = agents[AgentRole.DEVOPS_ENGINEER]
    context = orchestrator.get_context_for_agent(AgentRole.DEVOPS_ENGINEER)
    deploy_output = devops_agent.execute_task(deploy_task, context)

    orchestrator.update_task_status(deploy_task.id, TaskStatus.COMPLETED, deploy_output)

    print(f"\n✓ Deployment Task Completed")
    print(f"   Status: {deploy_output.get('status', 'N/A')}")

    # =========================================================================
    # PHASE 7: REVIEW
    # =========================================================================
    print("\n" + "=" * 100)
    print("PHASE 7: REVIEW - Workflow completed!")
    print("=" * 100)

    orchestrator.advance_phase(WorkflowPhase.REVIEW)

    # Get final workflow status
    status = orchestrator.get_workflow_status()

    print(f"\n📊 Final Workflow Status:")
    print(f"   Project: {status['project_name']}")
    print(f"   Phase: {status['current_phase']}")
    print(f"   Total Tasks: {status['total_tasks']}")
    print(f"   Completed: {status['task_status']['completed']}")
    print(f"   Decisions Made: {status['total_decisions']}")
    print(f"   Messages Exchanged: {status['total_messages']}")

    # =========================================================================
    # SAVE STATE
    # =========================================================================
    print("\n" + "=" * 100)
    print("SAVING WORKFLOW STATE")
    print("=" * 100)

    # Use absolute path based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "alert_system_workflow_state.json")
    orchestrator.save_state(output_file)
    print(f"\n✓ Workflow state saved to: {output_file}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 100)
    print("WORKFLOW SUMMARY")
    print("=" * 100)

    print("""
The multi-agent system successfully collaborated to build the Alert System feature:

1. Product Manager analyzed requirements and defined acceptance criteria
2. System Architect designed event-driven architecture and API contracts
3. QA Engineer created comprehensive test plan with edge cases
4. Frontend Engineer built alert dashboard UI
5. Backend Engineer implemented APIs and processing logic
6. AI Engineer designed risk detection model
7. QA Engineer validated implementation with automated tests
8. DevOps Engineer deployed system with monitoring

Key Benefits of Multi-Agent Approach:
✓ Specialized expertise applied at each phase
✓ Parallel development where possible
✓ Built-in quality checks and validations
✓ Complete documentation and decision history
✓ Coordinated handoffs between phases
✓ Scalable and maintainable workflow
    """)

    print("=" * 100)


if __name__ == "__main__":
    run_alert_system_workflow()
