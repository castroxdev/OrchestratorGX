# OrchestratorGX

A supervisor-based multi-agent LLM architecture with specialized agents and tool execution.

## Overview

OrchestratorGX is a learning and portfolio project focused on building a **supervisor-driven multi-agent system** with LLMs.

The main idea is to create an architecture where a **Supervisor Agent** receives the user's request, delegates the task to a specialized agent, allows that agent to use tools, and then composes the final response back to the user.

This project is being built to better understand:
- multi-agent LLM architectures
- supervisor-based orchestration
- tool execution workflows
- structured software planning with AI
- clean separation of responsibilities in code

## Architecture

The system follows this high-level flow:

User → Supervisor Agent → Specialized Agent → Tools → Supervisor Agent → User

### Main Components

- **User**
  - Sends a request to the system.

- **Supervisor Agent**
  - Interprets the user request.
  - Chooses the most appropriate specialized agent.
  - Receives the agent output.
  - Generates the final response.

- **Specialized Agents**
  - Focus on specific domains.
  - Can use one or more tools to complete their task.

- **Tools**
  - Perform concrete and limited actions.
  - Return structured outputs to support the agents.

## Initial Agents

### Planner Agent
Responsible for product planning tasks such as:
- MVP definition
- core feature extraction
- user flow definition

Tools:
- `generate_mvp_plan`
- `extract_core_features`
- `define_user_flows`

### API Agent
Responsible for backend/API design tasks such as:
- endpoint generation
- request/response model suggestions

Tools:
- `generate_api_endpoints`
- `suggest_request_response_models`

### Database Agent
Responsible for data modeling tasks such as:
- SQL schema generation
- entity suggestions
- relationship mapping

Tools:
- `generate_sql_schema`
- `suggest_entities`
- `map_relationships`

## Goals

The first goal of this project is not to build a huge production-ready system, but to build a **clear and understandable architecture** that can later grow into a more advanced orchestration platform.

Main goals:
- understand the role of a supervisor agent
- avoid relying only on fixed keywords
- let the LLM decide which agent should handle each task
- keep tool execution separate from final response generation
- build a clean and explainable codebase

## Project Status

This project is currently in its early development stage.

Planned milestones:
- [ ] Define the initial project structure
- [ ] Implement the Supervisor Agent
- [ ] Implement the Planner Agent
- [ ] Implement the API Agent
- [ ] Implement the Database Agent
- [ ] Implement tool execution flow
- [ ] Add structured logging
- [ ] Improve documentation and examples

## Planned Project Structure

orchestratorgx/
├── app/
│   ├── main.py
│   ├── core/
│   │   ├── supervisor.py
│   │   ├── router.py
│   │   └── llm_client.py
│   ├── agents/
│   │   ├── planner_agent.py
│   │   ├── api_agent.py
│   │   └── database_agent.py
│   ├── tools/
│   │   ├── planner_tools.py
│   │   ├── api_tools.py
│   │   └── database_tools.py
│   └── schemas/
│       └── messages.py
├── tests/
├── README.md
├── requirements.txt
└── .gitignore

## Why this project matters

OrchestratorGX is not just about making an AI system work.

It is about understanding:
- how LLMs can coordinate tasks
- how agents can be separated by responsibility
- how tools can support structured outputs
- how to design an architecture that is easier to reason about, explain, and improve

##