from typing import List, Dict, Any, Optional

from groq import Groq


class GroqLLM:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = Groq(api_key=api_key) if api_key else None

    def ensure_client(self):
        if not self.client:
            raise RuntimeError("GROQ API Key não configurada.")

    def _map_deprecated_model(self, model: str) -> str:
        mapping = {
            "llama3-8b-8192": "llama-3.1-8b-instant",
            "llama3-70b-8192": "llama-3.3-70b-versatile",
        }
        return mapping.get(model, model)

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama3-8b-8192",
        temperature: float = 0.2,
        max_output_tokens: int = 2000,
    ) -> str:
        self.ensure_client()
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e).lower()
            if "decommissioned" in msg or "model_decommissioned" in msg:
                upgraded = self._map_deprecated_model(model)
                if upgraded != model:
                    resp = self.client.chat.completions.create(
                        model=upgraded,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_output_tokens,
                    )
                    return resp.choices[0].message.content
            raise