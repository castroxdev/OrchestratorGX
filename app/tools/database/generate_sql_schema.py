from app.core.llm_client import LLMClient


def generate_sql_schema(llm_client: LLMClient, tool_request: str) -> str:
    # Restringe-se a exemplos básicos de schema para funcionar como passo de
    # síntese depois da identificação de entidades e relações.
    prompt = (
        "You are a database design tool inside a software assistant system.\n"
        "Generate a simple SQL schema based on the user's idea.\n"
        "Keep the answer short, practical, and clear.\n"
        "Return only basic CREATE TABLE examples.\n"
        "Do not add an introduction or conclusion.\n\n"
        f"User request: {tool_request}"
    )

    return llm_client.generate(prompt)
