from app.core.llm_client import LLMClient


def map_relationships(llm_client: LLMClient, tool_request: str) -> str:
    # Assume que as entidades já estão implícitas no pedido e concentra-se só
    # nas ligações entre elas para manter a resposta separada por responsabilidade.
    prompt = (
        "You are a database design tool inside a software assistant system.\n"
        "Map the main relationships between the entities of the user's idea.\n"
        "Keep the answer short, practical, and clear.\n"
        "Return only 3 to 5 bullet points.\n"
        "Each bullet point must describe one relationship.\n"
        "Do not add an introduction or conclusion.\n\n"
        f"User request: {tool_request}"
    )

    return llm_client.generate(prompt)
