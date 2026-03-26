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

    def stream_generate(self, prompt: str):
        stream = ollama.chat(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True,
        )

        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content
