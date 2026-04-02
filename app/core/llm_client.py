import ollama


class LLMClient:
    # Wrapper mínimo sobre o cliente Ollama para manter um único ponto de
    # integração com o modelo em todo o projeto.
    def __init__(self, model: str = "qwen2.5:7b") -> None:
        self.model = model

    def generate(self, prompt: str) -> str:
        # Centraliza chamadas síncronas para que agentes e tools usem sempre
        # o mesmo formato de pedido ao modelo.
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"].strip()

    def stream_generate(self, prompt: str):
        # Expõe o stream bruto de texto em chunks para a camada web atualizar
        # a resposta sem duplicar lógica de geração.
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
