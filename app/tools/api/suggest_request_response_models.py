from app.core.llm_client import LLMClient


def suggest_request_response_models(llm_client: LLMClient, tool_request: str) -> str:
    prompt = (
        "You are an API design tool inside a software assistant system.\n"
        "Suggest simple request and response models based on the user's idea.\n"
        "Keep the answer short, practical, and clear.\n"
        "Return 2 to 4 examples.\n"
        "Each example must include:\n"
        "- endpoint name\n"
        "- request fields\n"
        "- response fields\n"
        "Do not add an introduction or conclusion.\n\n"
        f"User request: {tool_request}"
    )

    return llm_client.generate(prompt)