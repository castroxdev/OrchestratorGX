from app.core.llm_client import LLMClient


def suggest_entities(llm_client: LLMClient, tool_request: str) -> str:
    # Isola a descoberta das entidades principais para que o agente possa
    # combinar esta base com relações ou schema quando fizer sentido.
    prompt = (
        "You are a database design tool inside a software assistant system.\n"
        "Suggest the main entities for the user's idea.\n"
        "Keep the answer short, practical, and clear.\n"
        "Return only 3 to 6 bullet points.\n"
        "Each bullet point must contain one entity name.\n"
        "Do not add an introduction or conclusion.\n\n"
        f"User request: {tool_request}"
    )

    return llm_client.generate(prompt)
