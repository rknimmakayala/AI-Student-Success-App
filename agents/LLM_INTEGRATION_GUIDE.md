# LLM Integration Guide

This guide shows how to integrate the multi-agent system with various LLM providers to create actual AI-powered agents.

## Overview

The base system provides the framework for multi-agent coordination. To make agents truly intelligent, you'll integrate LLM APIs.

## Architecture

```
┌─────────────────────────────────────────┐
│         Agent Base Class                │
│  (Defines interface and behavior)       │
└─────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│      LLM-Powered Agent                  │
│  • Uses agent system prompt             │
│  • Generates prompts from tasks         │
│  • Calls LLM API                        │
│  • Parses and structures responses      │
└─────────────────────────────────────────┘
```

## Option 1: OpenAI Integration

### Installation

```bash
pip install openai>=1.0.0
```

### Implementation

```python
import openai
import os
import json
from typing import Dict, Any
from agent_base import ProductManagerAgent
from orchestrator import Task

class OpenAIProductManager(ProductManagerAgent):
    """Product Manager powered by OpenAI GPT-4"""

    def __init__(self, orchestrator=None, model="gpt-4-turbo-preview"):
        super().__init__(orchestrator)
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using OpenAI"""

        # Generate comprehensive prompt
        prompt = self._generate_llm_prompt(task, context)

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            response_format={"type": "json_object"}  # Request structured JSON
        )

        # Parse response
        result = json.loads(response.choices[0].message.content)

        # Structure output
        output = {
            "agent": self.role.value,
            "task_id": task.id,
            "task_title": task.title,
            **result
        }

        self.log_task_completion(task)
        return output

    def _generate_llm_prompt(self, task: Task, context: Dict[str, Any]) -> str:
        """Generate detailed prompt for LLM"""

        # Get dependency outputs
        dep_outputs = []
        if self.orchestrator and task.dependencies:
            for dep_id in task.dependencies:
                if dep_id in self.orchestrator.tasks:
                    dep_task = self.orchestrator.tasks[dep_id]
                    if dep_task.output:
                        dep_outputs.append({
                            "task": dep_task.title,
                            "output": dep_task.output
                        })

        prompt = f"""
You are executing a task as a Product Manager.

## Task
**Title**: {task.title}
**Description**: {task.description}

## Context
{json.dumps(context, indent=2)}

## Previous Task Outputs
{json.dumps(dep_outputs, indent=2) if dep_outputs else "No dependencies"}

## Instructions
Please complete this task according to your Product Manager role.

Provide your response as a JSON object with the following structure:
{{
    "deliverable": "name of deliverable (e.g., 'requirements_document')",
    "summary": "brief summary of your analysis",
    "requirements": ["list", "of", "requirements"],
    "acceptance_criteria": ["list", "of", "criteria"],
    "success_metrics": {{"metric_name": "description"}},
    "constraints": ["any", "constraints"],
    "next_steps": ["recommended", "next", "steps"]
}}
"""
        return prompt


# Example usage
if __name__ == "__main__":
    from orchestrator import AgentOrchestrator
    from agent_definitions import AgentRole

    # Create orchestrator
    orchestrator = AgentOrchestrator("My Project")

    # Create OpenAI-powered PM
    pm = OpenAIProductManager(orchestrator)

    # Create task
    task = orchestrator.create_task(
        title="Analyze Feature Requirements",
        description="Analyze requirements for student analytics dashboard",
        assigned_to=AgentRole.PRODUCT_MANAGER
    )

    # Execute
    context = orchestrator.get_context_for_agent(AgentRole.PRODUCT_MANAGER)
    result = pm.execute_task(task, context)

    print(json.dumps(result, indent=2))
```

### Cost Optimization

```python
class CostOptimizedAgent(BaseAgent):
    """Use different models based on task complexity"""

    def __init__(self, orchestrator=None):
        super().__init__(orchestrator)
        self.client = openai.OpenAI()

    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        # Use cheaper model for simple tasks
        if self._is_simple_task(task):
            model = "gpt-3.5-turbo"
        else:
            model = "gpt-4-turbo-preview"

        # Rest of implementation...

    def _is_simple_task(self, task: Task) -> bool:
        """Determine if task is simple"""
        simple_keywords = ["list", "summarize", "format"]
        return any(kw in task.title.lower() for kw in simple_keywords)
```

## Option 2: Anthropic Claude Integration

### Installation

```bash
pip install anthropic>=0.18.0
```

### Implementation

```python
import anthropic
import os
import json
from typing import Dict, Any
from agent_base import SystemArchitectAgent
from orchestrator import Task

class ClaudeSystemArchitect(SystemArchitectAgent):
    """System Architect powered by Anthropic Claude"""

    def __init__(self, orchestrator=None, model="claude-3-5-sonnet-20241022"):
        super().__init__(orchestrator)
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model

    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using Claude"""

        prompt = self._generate_llm_prompt(task, context)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        # Parse Claude's response
        result = self._parse_claude_response(message.content[0].text)

        output = {
            "agent": self.role.value,
            "task_id": task.id,
            "task_title": task.title,
            **result
        }

        self.log_task_completion(task)
        return output

    def _parse_claude_response(self, text: str) -> Dict[str, Any]:
        """Parse Claude's response into structured format"""

        # Try to extract JSON if present
        try:
            # Look for JSON code blocks
            if "```json" in text:
                json_start = text.index("```json") + 7
                json_end = text.index("```", json_start)
                json_str = text[json_start:json_end].strip()
                return json.loads(json_str)
            else:
                # Try to parse entire response as JSON
                return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            # Fall back to extracting key information
            return {
                "deliverable": "architecture_design",
                "raw_response": text,
                "note": "Manual parsing required"
            }

    def _generate_llm_prompt(self, task: Task, context: Dict[str, Any]) -> str:
        """Generate prompt for Claude"""

        prompt = f"""
You are a System Architect working on: {task.title}

{task.description}

## Current Context
{json.dumps(context, indent=2)}

Please provide your architectural design in the following JSON format:

```json
{{
    "deliverable": "architecture_design",
    "components": [
        {{
            "name": "Component Name",
            "description": "What it does",
            "technology": "Recommended tech"
        }}
    ],
    "data_flow": "Description of data flow",
    "api_contracts": [
        {{
            "endpoint": "/api/path",
            "method": "GET",
            "description": "What it does"
        }}
    ],
    "adr": {{
        "title": "Key Decision",
        "decision": "What was decided",
        "rationale": "Why"
    }},
    "next_steps": ["step1", "step2"]
}}
```
"""
        return prompt
```

## Option 3: LangChain Integration

### Installation

```bash
pip install langchain>=0.1.0 langchain-openai>=0.0.5 langchain-anthropic>=0.0.1
```

### Implementation

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Define structured output
class RequirementsOutput(BaseModel):
    deliverable: str = Field(description="Type of deliverable")
    summary: str = Field(description="Summary of analysis")
    requirements: List[str] = Field(description="List of requirements")
    acceptance_criteria: List[str] = Field(description="Acceptance criteria")
    success_metrics: Dict[str, str] = Field(description="Success metrics")
    next_steps: List[str] = Field(description="Recommended next steps")


class LangChainProductManager(ProductManagerAgent):
    """Product Manager using LangChain"""

    def __init__(self, orchestrator=None, provider="openai"):
        super().__init__(orchestrator)

        # Choose LLM provider
        if provider == "openai":
            self.llm = ChatOpenAI(
                model="gpt-4-turbo-preview",
                temperature=0.7
            )
        elif provider == "anthropic":
            self.llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                temperature=0.7
            )

        # Set up output parser
        self.output_parser = PydanticOutputParser(
            pydantic_object=RequirementsOutput
        )

    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using LangChain"""

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{task_description}\n\n{format_instructions}")
        ])

        # Create chain
        chain = prompt | self.llm | self.output_parser

        # Execute
        result = chain.invoke({
            "task_description": self._format_task(task, context),
            "format_instructions": self.output_parser.get_format_instructions()
        })

        # Convert Pydantic model to dict
        output = {
            "agent": self.role.value,
            "task_id": task.id,
            **result.dict()
        }

        self.log_task_completion(task)
        return output

    def _format_task(self, task: Task, context: Dict[str, Any]) -> str:
        return f"""
Task: {task.title}
Description: {task.description}

Context:
{json.dumps(context, indent=2)}
"""
```

### Advanced LangChain: Agent with Tools

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub

class LangChainToolAgent(BaseAgent):
    """Agent with access to tools via LangChain"""

    def __init__(self, role: AgentRole, orchestrator=None):
        super().__init__(role, orchestrator)

        # Define tools the agent can use
        tools = [
            Tool(
                name="SearchCodebase",
                func=self._search_codebase,
                description="Search the codebase for specific patterns or files"
            ),
            Tool(
                name="AnalyzeMetrics",
                func=self._analyze_metrics,
                description="Analyze system metrics and performance data"
            ),
            Tool(
                name="GetRequirements",
                func=self._get_requirements,
                description="Retrieve project requirements and constraints"
            )
        ]

        # Create agent
        llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
        prompt = hub.pull("hwchase17/openai-functions-agent")

        agent = create_openai_functions_agent(llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True
        )

    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using LangChain agent with tools"""

        prompt = f"""
Task: {task.title}
Description: {task.description}

Context: {json.dumps(context, indent=2)}

Complete this task using the available tools as needed.
"""

        result = self.agent_executor.invoke({"input": prompt})

        output = {
            "agent": self.role.value,
            "task_id": task.id,
            "result": result["output"]
        }

        self.log_task_completion(task)
        return output

    def _search_codebase(self, query: str) -> str:
        """Tool: Search codebase"""
        # Implementation here
        return f"Search results for: {query}"

    def _analyze_metrics(self, metric_name: str) -> str:
        """Tool: Analyze metrics"""
        # Implementation here
        return f"Analysis of {metric_name}"

    def _get_requirements(self, feature: str) -> str:
        """Tool: Get requirements"""
        if self.orchestrator:
            return json.dumps(self.orchestrator.context.requirements)
        return "{}"
```

## Option 4: Local LLMs (Ollama, LM Studio)

### Installation

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama2
ollama pull codellama
```

### Implementation

```python
import requests
import json
from typing import Dict, Any

class OllamaAgent(BaseAgent):
    """Agent using local Ollama LLM"""

    def __init__(self, role: AgentRole, orchestrator=None, model="llama2"):
        super().__init__(role, orchestrator)
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"

    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using local Ollama"""

        prompt = self._generate_prompt(task, context)

        response = requests.post(
            self.api_url,
            json={
                "model": self.model,
                "prompt": f"{self.system_prompt}\n\n{prompt}",
                "stream": False
            }
        )

        result = response.json()
        output_text = result.get("response", "")

        # Parse output
        output = self._parse_output(output_text)
        output.update({
            "agent": self.role.value,
            "task_id": task.id
        })

        self.log_task_completion(task)
        return output

    def _parse_output(self, text: str) -> Dict[str, Any]:
        """Parse LLM output"""
        # Implementation depends on output format
        return {"raw_response": text}
```

## Best Practices

### 1. Error Handling

```python
class RobustAgent(BaseAgent):
    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self._call_llm(task, context)
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    # Final attempt failed
                    return {
                        "agent": self.role.value,
                        "task_id": task.id,
                        "error": str(e),
                        "status": "failed"
                    }
                # Wait before retry
                time.sleep(2 ** attempt)
```

### 2. Response Validation

```python
from pydantic import BaseModel, ValidationError

def validate_llm_output(output: Dict[str, Any], schema: BaseModel) -> Dict[str, Any]:
    """Validate LLM output against schema"""
    try:
        validated = schema(**output)
        return validated.dict()
    except ValidationError as e:
        # Handle validation errors
        return {
            "validation_error": str(e),
            "raw_output": output
        }
```

### 3. Cost Tracking

```python
class CostTrackingAgent(BaseAgent):
    def __init__(self, role: AgentRole, orchestrator=None):
        super().__init__(role, orchestrator)
        self.total_tokens = 0
        self.total_cost = 0.0

    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client.chat.completions.create(...)

        # Track usage
        tokens = response.usage.total_tokens
        cost = self._calculate_cost(tokens)

        self.total_tokens += tokens
        self.total_cost += cost

        # Log to agent memory
        self.remember("cost_tracking", {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost
        }, important=True)

        # Continue with normal processing...
```

### 4. Caching

```python
import hashlib
from functools import lru_cache

class CachedAgent(BaseAgent):
    @lru_cache(maxsize=100)
    def _call_llm_cached(self, prompt_hash: str, prompt: str) -> str:
        """Cache LLM responses"""
        response = self.client.chat.completions.create(...)
        return response.choices[0].message.content

    def execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._generate_prompt(task, context)
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

        # Use cached version if available
        result = self._call_llm_cached(prompt_hash, prompt)
        # Process result...
```

## Environment Variables

Create a `.env` file:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# LangChain
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
```

Load in Python:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Testing LLM Integration

```python
import pytest
from unittest.mock import Mock, patch

def test_openai_agent():
    """Test OpenAI agent"""

    with patch('openai.OpenAI') as mock_openai:
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "deliverable": "test",
            "requirements": ["req1", "req2"]
        })

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        # Test agent
        agent = OpenAIProductManager()
        task = Mock()
        task.id = "test_1"
        task.title = "Test Task"
        task.description = "Test"
        task.dependencies = []

        result = agent.execute_task(task, {})

        assert result["deliverable"] == "test"
        assert len(result["requirements"]) == 2
```

## Production Considerations

1. **Rate Limiting** - Implement rate limiting for API calls
2. **Monitoring** - Track API usage, costs, and errors
3. **Fallbacks** - Have fallback strategies when APIs fail
4. **Security** - Protect API keys, sanitize inputs
5. **Logging** - Log all LLM interactions for debugging
6. **Version Control** - Track prompt versions and performance

## Next Steps

1. Choose your LLM provider(s)
2. Implement agent classes using examples above
3. Test with sample tasks
4. Deploy and monitor
5. Iterate on prompts based on performance
