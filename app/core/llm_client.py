import ollama


class LLMClient:
    def __init__(self, model: str = "qwen2.5:7b") -> None:
        self.model = model

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"].strip()