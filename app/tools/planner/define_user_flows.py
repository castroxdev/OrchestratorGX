from app.core.llm_client import LLMClient


def define_user_flows(llm_client: LLMClient, user_message: str) -> str:
    prompt = (
        "You are a planning tool inside a software assistant system.\n"
        "Define the main user flows for the user's idea.\n"
        "Keep the answer short, practical, and focused on the essential first version.\n"
        "Return only 3 to 5 bullet points.\n"
        "Each bullet point must describe one simple user flow.\n"
        "Focus on core usage flows, not extra features.\n"
        "Do not add an introduction.\n"
        "Do not add a conclusion.\n\n"
        f"User request: {user_message}"
    )

    return llm_client.generate(prompt)