from app.core.llm_client import LLMClient
from app.schemas.messages import AgentResult
from app.tools.database.generate_sql_schema import generate_sql_schema
from app.tools.database.map_relationships import map_relationships
from app.tools.database.suggest_entities import suggest_entities


class DatabaseAgent:
    name = "database"

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        self.tools = {
            "generate_sql_schema": generate_sql_schema,
            "suggest_entities": suggest_entities,
            "map_relationships": map_relationships,
        }

    def choose_tools(self, user_message: str) -> list[str]:
        prompt = (
            "You are the Database Agent of a software assistant system.\n"
            "Your task is to choose the best database options for the user's request.\n"
            "You may choose one tool, multiple tools, or none.\n"
            "Choose multiple tools when the request clearly includes multiple database tasks.\n"
            "For example:\n"
            "- entities + relationships -> suggest_entities, map_relationships\n"
            "- entities + relationships + sql schema -> suggest_entities, map_relationships, generate_sql_schema\n\n"
            "Available options:\n"
            "- generate_sql_schema: use this for requests asking for SQL schemas, table structures, database scripts, or CREATE TABLE definitions\n"
            "- suggest_entities: use this for requests asking for main entities, database objects, or what should exist in the data model\n"
            "- map_relationships: use this for requests asking for relationships between entities, foreign keys, associations, or how tables connect\n"
            "- none: use this if the request is database-related but none of the available tools is a strong fit, and the agent should answer directly\n\n"
            "Reply with:\n"
            "- one or more option names separated by commas\n"
            "- or none\n\n"
            "Reply with only valid option names and nothing else.\n\n"
            "Valid options:\n"
            "generate_sql_schema\n"
            "suggest_entities\n"
            "map_relationships\n"
            "none\n\n"
            f"User request: {user_message}"
        )

        raw_selection = self.llm_client.generate(prompt).strip().lower()

        if raw_selection == "none":
            return []

        valid_tools = {
            "generate_sql_schema",
            "suggest_entities",
            "map_relationships",
        }

        selected_tools: list[str] = []
        for item in raw_selection.split(","):
            tool_name = item.strip()
            if tool_name in valid_tools and tool_name not in selected_tools:
                selected_tools.append(tool_name)

        return selected_tools

    def respond_directly(self, user_message: str) -> str:
        prompt = (
            "You are the Database Agent of a software assistant system.\n"
            "Answer the user's request directly without using any tool.\n"
            "You specialize in database design tasks such as schemas, entities, relationships, normalization, "
            "table structure, and data modeling decisions.\n"
            "Be clear, practical, and well organized.\n"
            "Do not mention internal tools, routing, or system behavior.\n\n"
            f"User request: {user_message}"
        )

        return self.llm_client.generate(prompt)

    def combine_tool_results(self, results: list[tuple[str, str]]) -> str:
        sections: list[str] = []

        for tool_name, content in results:
            title = tool_name.replace("_", " ").title()
            sections.append(f"{title}:\n{content}")

        return "\n\n".join(sections)

    def handle(self, user_message: str) -> AgentResult:
        selected_tools = self.choose_tools(user_message)

        if not selected_tools:
            response = self.respond_directly(user_message)

            return AgentResult(
                agent_name=self.name,
                content=response,
                used_tools=[]
            )

        tool_results: list[tuple[str, str]] = []

        for tool_name in selected_tools:
            tool_function = self.tools[tool_name]
            tool_output = tool_function(self.llm_client, user_message)
            tool_results.append((tool_name, tool_output))

        combined_result = self.combine_tool_results(tool_results)

        return AgentResult(
            agent_name=self.name,
            content=combined_result,
            used_tools=selected_tools
        )