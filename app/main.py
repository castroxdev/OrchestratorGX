from app.core.llm_client import LLMClient
from app.core.supervisor import SupervisorAgent


def main() -> None:
    llm_client = LLMClient(model="qwen2.5-coder:3b")
    supervisor = SupervisorAgent(llm_client)

    print("OrchestratorGX")
    print("Type 'exit' to quit.\n")

    while True:
        user_message = input("You: ").strip()

        if user_message.lower() == "exit":
            print("Goodbye.")
            break

        if not user_message:
            continue

        response = supervisor.handle(user_message)

        print(f"\nSelected agent: {response.selected_agent}")

        if response.used_tools:
            print(f"Used tools: {', '.join(response.used_tools)}")
        else:
            print("Used tools: none")

        print(f"\nAssistant: {response.final_response}\n")


if __name__ == "__main__":
    main()