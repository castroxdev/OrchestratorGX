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

    def choose_tool(self, user_message: str) -> str:
        prompt = (
            "You are the Database Agent of a software assistant system.\n"
            "Choose the best database tool for the user's request.\n"
            "Available tools:\n"
            "- generate_sql_schema: for generating a basic SQL schema\n"
            "- suggest_entities: for identifying the main entities of the system\n"
            "- map_relationships: for describing the main relationships between entities\n\n"
            "Reply with only one tool name.\n\n"
            f"User request: {user_message}"
        )

        selected_tool = self.llm_client.generate(prompt).strip().lower()

        if selected_tool not in self.tools:
            return "generate_sql_schema"

        return selected_tool

    def handle(self, user_message: str) -> AgentResult:
        selected_tool = self.choose_tool(user_message)
        tool_function = self.tools[selected_tool]
        tool_result = tool_function(self.llm_client, user_message)

        return AgentResult(
            agent_name=self.name,
            content=tool_result,
            used_tool=selected_tool
        )