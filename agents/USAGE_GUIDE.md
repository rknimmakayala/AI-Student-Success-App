# Multi-Agent Product Development System - Usage Guide

## Overview

This multi-agent system simulates a cross-functional product development team where specialized AI agents collaborate to build, enhance, and maintain software features.

## Quick Start

### 1. Run the Example Workflow

See the complete system in action:

```bash
cd agents
python examples/workflow_example.py
```

This demonstrates building a "Real-time Alert System for At-Risk Students" feature from requirements through deployment.

### 2. Use the Interactive CLI

```bash
# Create a workflow for a new feature
python cli.py workflow "Add student performance analytics dashboard"

# Execute all ready tasks
python cli.py execute

# Check workflow status
python cli.py status

# View architectural decisions
python cli.py decisions

# Save workflow state
python cli.py save my_workflow.json
```

## System Components

### 1. Agent Definitions (`agent_definitions.py`)

Defines the 7 specialized agent roles:

- **Product Manager** - Requirements, constraints, scope
- **System Architect** - Architecture, APIs, risk assessment
- **Frontend Engineer** - UI components, user flows
- **Backend Engineer** - APIs, data models, business logic
- **AI Engineer** - Reasoning chains, prompts, evaluation
- **QA Engineer** - Tests, edge cases, validation
- **DevOps Engineer** - CI/CD, deployment, monitoring

Each agent has:
- Specific expertise areas
- Defined capabilities
- System prompts for AI behavior
- Interaction patterns with other agents

### 2. Orchestrator (`orchestrator.py`)

Coordinates all agents and manages workflow:

- **Task Management** - Creates, tracks, and delegates tasks
- **Dependency Resolution** - Ensures tasks execute in correct order
- **Context Sharing** - Maintains shared state across agents
- **Message Routing** - Facilitates inter-agent communication
- **Decision Recording** - Tracks architectural decisions (ADRs)
- **Phase Management** - Moves workflow through stages

### 3. Agent Base Classes (`agent_base.py`)

Provides base functionality for all agents:

- **Memory System** - Short-term and long-term memory
- **Task Execution** - Execute assigned tasks
- **Communication** - Send/receive messages
- **Context Access** - Get relevant workflow context

### 4. CLI Tool (`cli.py`)

Command-line interface for interacting with the system.

## Detailed Usage

### Creating a Custom Workflow

```python
from orchestrator import AgentOrchestrator, WorkflowPhase
from agent_base import create_agent
from agent_definitions import AgentRole

# Initialize
orchestrator = AgentOrchestrator("My Project")

# Create agents
agents = {
    role: create_agent(role, orchestrator)
    for role in AgentRole
}

# Create tasks manually
task1 = orchestrator.create_task(
    title="Define Analytics Requirements",
    description="Analyze requirements for student analytics dashboard",
    assigned_to=AgentRole.PRODUCT_MANAGER
)

task2 = orchestrator.create_task(
    title="Design Analytics Architecture",
    description="Design data pipeline and visualization architecture",
    assigned_to=AgentRole.SYSTEM_ARCHITECT,
    dependencies=[task1.id]
)

# Execute tasks
pm_agent = agents[AgentRole.PRODUCT_MANAGER]
context = orchestrator.get_context_for_agent(AgentRole.PRODUCT_MANAGER)
output = pm_agent.execute_task(task1, context)

# Update status
orchestrator.update_task_status(
    task1.id,
    TaskStatus.COMPLETED,
    output
)
```

### Using Pre-built Workflows

The orchestrator can generate standard workflows:

```python
# Generate complete workflow for a feature
tasks = orchestrator.generate_workflow_for_feature(
    "Student performance analytics dashboard"
)

# This creates tasks for all phases:
# 1. PM analyzes requirements
# 2. Architect designs solution
# 3. QA creates test plan
# 4. Engineers implement (frontend, backend, AI)
# 5. QA validates
# 6. DevOps deploys

# Execute ready tasks
ready_tasks = orchestrator.get_ready_tasks()
for task in ready_tasks:
    agent = agents[task.assigned_to]
    context = orchestrator.get_context_for_agent(task.assigned_to)
    output = agent.execute_task(task, context)
    orchestrator.update_task_status(task.id, TaskStatus.COMPLETED, output)
```

### Recording Decisions

Track important architectural and product decisions:

```python
orchestrator.add_decision(
    title="Use React for Frontend",
    description="Implement dashboard using React framework",
    rationale="React provides component reusability and strong ecosystem for data visualization",
    decided_by=AgentRole.SYSTEM_ARCHITECT,
    metadata={
        "affects": ["frontend_engineer"],
        "alternatives_considered": ["Vue.js", "Angular"]
    }
)
```

### Managing Context

The orchestrator maintains shared context accessible to all agents:

```python
# Update requirements (typically done by PM)
orchestrator.context.requirements = {
    "user_stories": [...],
    "acceptance_criteria": [...],
    "constraints": [...]
}

# Update architecture (typically done by Architect)
orchestrator.context.architecture = {
    "components": [...],
    "apis": [...],
    "data_flow": "..."
}

# Each agent gets relevant context
context = orchestrator.get_context_for_agent(AgentRole.FRONTEND_ENGINEER)
# Context includes architecture but not internal PM details
```

### Inter-Agent Communication

Agents can send messages to each other:

```python
# Agent sends question to another agent
pm_agent.communicate(
    message_type="question",
    content={
        "question": "What's the estimated effort for this feature?",
        "context": {...}
    },
    to_agent=AgentRole.BACKEND_ENGINEER
)

# Retrieve messages for an agent
messages = orchestrator.get_messages_for_agent(
    AgentRole.BACKEND_ENGINEER,
    message_type="question"
)
```

### Saving and Loading State

```python
# Save complete workflow state
orchestrator.save_state("workflow_state.json")

# Export state for inspection
state = orchestrator.export_state()
print(json.dumps(state, indent=2))

# Get workflow status
status = orchestrator.get_workflow_status()
```

## Workflow Phases

The system follows a standard product development workflow:

1. **DISCOVERY** - Understand requirements and user needs
   - PM analyzes and documents requirements
   - Constraints and success metrics defined

2. **ARCHITECTURE** - Design technical solution
   - Architect designs system architecture
   - API contracts and data flows defined
   - Technical decisions recorded

3. **PLANNING** - Plan implementation and testing
   - QA creates test strategy
   - Implementation approach finalized

4. **DEVELOPMENT** - Build the feature
   - Frontend, Backend, and AI engineers implement
   - Code reviews and collaboration

5. **TESTING** - Validate implementation
   - QA executes test plan
   - Bugs identified and fixed

6. **DEPLOYMENT** - Release to production
   - DevOps deploys feature
   - Monitoring and observability set up

7. **REVIEW** - Assess outcomes and learnings
   - Team reviews what worked well
   - Identify improvements for next iteration

## Extending the System

### Adding Custom Agent Capabilities

```python
from agent_base import BaseAgent
from agent_definitions import AgentRole

class CustomAIEngineer(BaseAgent):
    def __init__(self, orchestrator=None):
        super().__init__(AgentRole.AI_ENGINEER, orchestrator)

    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        # Custom implementation
        if "prompt optimization" in task.title.lower():
            return self._optimize_prompts(task, context)

        # Fall back to base implementation
        return super().execute_task(task, context)

    def _optimize_prompts(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        # Your custom logic here
        return {
            "agent": self.role.value,
            "deliverable": "optimized_prompts",
            "prompts": [...]
        }
```

### Adding New Agent Roles

1. Add role to `AgentRole` enum in `agent_definitions.py`
2. Add agent definition to `AGENT_DEFINITIONS` dict
3. Create agent class extending `BaseAgent`
4. Add to agent factory in `agent_base.py`

### Customizing Workflows

```python
def custom_workflow(orchestrator, feature_description):
    """Create a custom workflow"""

    # Your custom task creation logic
    tasks = []

    # Example: Add research phase before discovery
    research_task = orchestrator.create_task(
        title="Market Research",
        description=f"Research market for: {feature_description}",
        assigned_to=AgentRole.PRODUCT_MANAGER
    )
    tasks.append(research_task)

    # Continue with standard workflow
    # ...

    return tasks
```

## Integration with LLMs

The current implementation provides the framework. To integrate with actual LLMs:

### Option 1: OpenAI API

```python
import openai

class LLMProductManagerAgent(ProductManagerAgent):
    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        # Generate prompt for LLM
        prompt = self.orchestrator.generate_agent_prompt(self.role, task)

        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        # Parse and return output
        output = self._parse_llm_response(response.choices[0].message.content)
        self.log_task_completion(task)
        return output
```

### Option 2: Anthropic Claude API

```python
import anthropic

class ClaudeBackendEngineerAgent(BackendEngineerAgent):
    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        prompt = self.orchestrator.generate_agent_prompt(self.role, task)

        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )

        output = self._parse_claude_response(message.content[0].text)
        self.log_task_completion(task)
        return output
```

### Option 3: LangChain Integration

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

class LangChainAgent(BaseAgent):
    def __init__(self, role: AgentRole, orchestrator=None):
        super().__init__(role, orchestrator)
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)

    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{task_prompt}")
        ])

        chain = LLMChain(llm=self.llm, prompt=prompt_template)

        task_prompt = self.orchestrator.generate_agent_prompt(self.role, task)
        result = chain.run(task_prompt=task_prompt)

        # Parse output
        output = self._parse_result(result)
        self.log_task_completion(task)
        return output
```

## Best Practices

1. **Clear Task Descriptions** - Be specific about what each task should accomplish

2. **Proper Dependencies** - Set task dependencies to ensure correct execution order

3. **Context Management** - Keep shared context updated as work progresses

4. **Decision Recording** - Document important decisions with rationale

5. **Memory Usage** - Use agent memory for important information that needs to persist

6. **Status Updates** - Update task status promptly as work completes

7. **Error Handling** - Mark tasks as blocked/failed when issues arise

8. **Phase Transitions** - Advance phases as workflow progresses

## Troubleshooting

### Tasks Not Becoming Ready

- Check task dependencies are completed
- Verify task status is PENDING
- Use `orchestrator.get_ready_tasks()` to see what's ready

### Agent Context Issues

- Ensure orchestrator is passed to agents
- Verify context is updated after task completion
- Check agent-specific context filtering

### Memory Issues

- Short-term memory has limited window (default 10 items)
- Use long-term memory for important information
- Call `agent.remember()` with `important=True`

## Examples

See `/agents/examples/` for complete working examples:

- `workflow_example.py` - Full feature development workflow
- `alert_system_workflow_state.json` - Saved workflow state example

## Support

For issues, questions, or contributions, refer to the main project documentation.
