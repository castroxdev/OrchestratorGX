from app.core.llm_client import LLMClient


def extract_core_features(llm_client: LLMClient, user_message: str) -> str:
    prompt = (
        "You are a planning tool inside a software assistant system.\n"
        "Extract the core features of the user's idea.\n"
        "Return only 3 to 5 bullet points.\n"
        "Do not add an introduction.\n"
        "Do not add a conclusion.\n"
        "Do not write any extra explanation.\n\n"
        f"User request: {user_message}"
    )

    return llm_client.generate(prompt)