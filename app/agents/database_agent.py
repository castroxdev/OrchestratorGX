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
            "Your task is to choose the best option for the user's request.\n"
            "Available options:\n"
            "- generate_sql_schema: use this for requests asking for SQL schemas, table structures, database scripts, or CREATE TABLE definitions\n"
            "- suggest_entities: use this for requests asking for main entities, database objects, or what should exist in the data model\n"
            "- map_relationships: use this for requests asking for relationships between entities, foreign keys, associations, or how tables connect\n"
            "- none: use this if the request is database-related but none of the available tools is a strong fit, and the agent should answer directly\n\n"
            "Reply with only one option name and nothing else.\n"
            "Valid options:\n"
            "generate_sql_schema\n"
            "suggest_entities\n"
            "map_relationships\n"
            "none\n\n"
            f"User request: {user_message}"
        )

        selected_tool = self.llm_client.generate(prompt).strip().lower()

        valid_options = {
            "generate_sql_schema",
            "suggest_entities",
            "map_relationships",
            "none",
        }

        if selected_tool not in valid_options:
            return "none"

        return selected_tool

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

    def handle(self, user_message: str) -> AgentResult:
        selected_tool = self.choose_tool(user_message)

        if selected_tool == "none":
            response = self.respond_directly(user_message)

            return AgentResult(
                agent_name=self.name,
                content=response,
                used_tools=[]
            )

        tool_function = self.tools[selected_tool]
        tool_result = tool_function(self.llm_client, user_message)

        return AgentResult(
            agent_name=self.name,
            content=tool_result,
            used_tools=[selected_tool]
        )