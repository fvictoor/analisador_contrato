from typing import List, Dict, Any, Optional

from groq import Groq
import time

# Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None


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

    def _fallback_daily_limit_model(self, model: str) -> str:
        # Em caso de TPD, tentar automaticamente um modelo menor para reduzir custo de tokens
        # Mantém o mesmo modelo se já for o menor
        if model == "llama-3.3-70b-versatile":
            return "llama-3.1-8b-instant"
        return model

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama3-8b-8192",
        temperature: float = 0.2,
        max_output_tokens: int = 2000,
    ) -> str:
        self.ensure_client()
        # Retry/backoff básico para rate limit/TPM
        tries = 3
        model_to_use = model
        for attempt in range(tries):
            try:
                resp = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                )
                return resp.choices[0].message.content
            except Exception as e:
                msg = str(e).lower()
                # Se modelo foi desativado, troca para mapeado
                if "decommissioned" in msg or "model_decommissioned" in msg:
                    upgraded = self._map_deprecated_model(model_to_use)
                    if upgraded != model_to_use:
                        model_to_use = upgraded
                        # tenta novamente imediatamente com o modelo novo
                        continue
                # Backoff em caso de rate limit/TPM
                if ("rate_limit" in msg) and ("tokens per minute" in msg or "tpm" in msg):
                    if attempt < tries - 1:
                        # espera incremental: 6s, 12s
                        time.sleep(6 * (attempt + 1))
                        continue
                # Fallback em caso de limite diário (TPD): tentar modelo menor
                if ("rate_limit" in msg) and ("tokens per day" in msg or "tpd" in msg):
                    fallback = self._fallback_daily_limit_model(model_to_use)
                    if fallback != model_to_use:
                        model_to_use = fallback
                        # tenta novamente com o modelo menor
                        continue
                # Se não conseguimos tratar, repropaga
                raise


class GeminiLLM:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        if genai and api_key:
            genai.configure(api_key=api_key)
        self._configured = bool(genai and api_key)

    def ensure_client(self):
        if not self._configured:
            raise RuntimeError("Gemini API Key não configurada ou biblioteca ausente.")

    def _candidate_models(self, model: str) -> List[str]:
        # Gera alternativas conhecidas para evitar 404/método não suportado em v1beta
        aliases = {
            # Antigos → Novos sugeridos
            "gemini-1.5-flash": ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-1.5-flash-001"],
            "gemini-1.5-pro": ["gemini-2.5-pro", "gemini-1.5-pro-001"],
            # Novos → Alternativas próximas
            "gemini-2.5-flash-lite": ["gemini-2.5-flash", "gemini-1.5-flash"],
            "gemini-2.5-flash": ["gemini-2.5-flash-lite", "gemini-1.5-flash"],
        }
        candidates = [model]
        candidates += aliases.get(model, [])
        return candidates

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str = "gemini-1.5-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 2000,
    ) -> str:
        self.ensure_client()
        # Concatena mensagens com rótulos de papel para simular histórico
        prompt_parts = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        prompt = "\n\n".join(prompt_parts)

        last_error: Optional[Exception] = None
        for model_to_use in self._candidate_models(model):
            try:
                model_obj = genai.GenerativeModel(model_to_use)
                resp = model_obj.generate_content(
                    prompt,
                    generation_config={
                        "temperature": float(max(0.0, min(1.0, temperature))),
                        "max_output_tokens": int(max_output_tokens),
                    },
                )
                return getattr(resp, "text", "") or ""
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                # Em caso de modelo não encontrado/unsupported, tenta próximo candidato
                if (
                    "404" in msg
                    or "not found" in msg
                    or "listmodels" in msg
                    or "not supported for generatecontent" in msg
                ):
                    continue
                # Em caso de rate/quota, pequeno backoff e tenta novamente mesmo modelo
                if ("rate" in msg or "quota" in msg or "429" in msg):
                    time.sleep(2)
                    continue
                # Outros erros: interrompe
                break
        # Se nenhuma tentativa funcionou, repropaga última exceção
        if last_error:
            raise last_error
        return ""