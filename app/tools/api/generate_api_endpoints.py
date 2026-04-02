from app.core.llm_client import LLMClient


def generate_api_endpoints(llm_client: LLMClient, tool_request: str) -> str:
    # Esta tool devolve só endpoints e propósito, sem detalhe adicional, para
    # o agente decidir se precisa de complementar com modelos.
    prompt = (
        "You are an API design tool inside a software assistant system.\n"
        "Generate basic REST API endpoints based on the user's idea.\n"
        "Keep the answer short, practical, and clear.\n"
        "Return only 4 to 8 bullet points.\n"
        "Each bullet point must contain one endpoint and its purpose.\n"
        "Do not add an introduction or conclusion.\n\n"
        f"User request: {tool_request}"
    )

    return llm_client.generate(prompt)
