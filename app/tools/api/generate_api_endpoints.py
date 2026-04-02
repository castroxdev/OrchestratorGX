from app.core.llm_client import LLMClient


def generate_api_endpoints(llm_client: LLMClient, tool_request: str) -> str:
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