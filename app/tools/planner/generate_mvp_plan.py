from app.core.llm_client import LLMClient


def generate_mvp_plan(llm_client: LLMClient, tool_request: str) -> str:
    # Produz um plano inicial curto e estruturado para ser combinado depois
    # com outros resultados, se o agente tiver escolhido mais tools.
    prompt = (
        "You are a planning tool inside a software assistant system.\n"
        "Generate a short MVP plan based on the user's idea.\n"
        "Keep the answer practical, clear, and structured.\n\n"
        "Include:\n"
        "- project goal\n"
        "- core features\n"
        "- simple first version scope\n\n"
        f"User request: {tool_request}"
    )

    return llm_client.generate(prompt)
