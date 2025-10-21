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
        response_mime_type: Optional[str] = None,
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

    def _extract_text(self, resp) -> str:
        try:
            return getattr(resp, "text", "")
        except Exception:
            pass
        try:
            data = resp.to_dict() if hasattr(resp, "to_dict") else None
            if data:
                cands = data.get("candidates") or []
                for c in cands:
                    content = c.get("content") or {}
                    for p in content.get("parts") or []:
                        t = p.get("text")
                        if t:
                            return t
            cands = getattr(resp, "candidates", []) or []
            for c in cands:
                content = getattr(c, "content", None)
                parts = getattr(content, "parts", []) if content else []
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        return t
        except Exception:
            pass
        return ""

    def _candidate_models(self, model: str) -> List[str]:
        # Gera alternativas conhecidas para evitar 404/método não suportado em v1beta
        aliases = {
            # Família 2.5 flash
            "gemini-2.5-flash-lite": ["gemini-2.5-flash", "gemini-1.5-flash"],
            "gemini-2.5-flash": ["gemini-2.5-flash-lite", "gemini-1.5-flash"],
            # Família 2.5 pro
            "gemini-2.5-pro": ["gemini-1.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-1.5-flash"],
            # Família 1.5
            "gemini-1.5-flash": ["gemini-2.5-flash-lite", "gemini-2.5-flash"],
            "gemini-1.5-pro": ["gemini-2.5-pro", "gemini-1.5-flash"],
            # Família 2.0 e aliases "latest"
            "gemini-2.0-flash": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-1.5-flash"],
            "gemini-2.0-flash-exp": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
            "gemini-2.0-flash-lite": ["gemini-2.5-flash-lite", "gemini-2.5-flash"],
            "gemini-2.0-flash-lite-preview": ["gemini-2.0-flash-lite", "gemini-2.5-flash-lite"],
            "gemini-2.0-pro-exp": ["gemini-2.5-pro", "gemini-2.5-flash"],
            "gemini-2.0-flash-thinking-exp": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
            "gemini-2.0-flash-preview-image-generation": ["gemini-2.0-flash", "gemini-2.5-flash"],
            "gemini-2.0-flash-exp-image-generation": ["gemini-2.0-flash", "gemini-2.5-flash"],
            "gemini-2.5-flash-preview-tts": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
            "gemini-2.5-pro-preview-tts": ["gemini-2.5-pro", "gemini-2.5-flash"],
            "gemini-flash-latest": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite"],
            "gemini-flash-lite-latest": ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-2.5-flash"],
            "gemini-pro-latest": ["gemini-2.5-pro", "gemini-2.0-pro-exp", "gemini-2.5-flash"],
        }
        # Lista candidatos e tenta filtrar por modelos realmente disponíveis
        cands = [model] + aliases.get(model, [])
        try:
            models = list(genai.list_models())
            names_full = [getattr(m, "name", "") for m in models]
            names_simplified = {n.split("/")[-1] for n in names_full if n}
            supported_full = [
                getattr(m, "name", "")
                for m in models
                if (getattr(m, "supported_generation_methods", []) or [])
                and ("generateContent" in getattr(m, "supported_generation_methods", []) or "generate_content" in getattr(m, "supported_generation_methods", []))
            ]
            supported_simplified = {n.split("/")[-1] for n in supported_full if n}
            filtered = [
                c for c in cands
                if (c in names_simplified) or (c in supported_simplified)
            ]
            return filtered if filtered else cands
        except Exception:
            return cands

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.2,
        max_output_tokens: int = 2000,
        response_mime_type: Optional[str] = None,
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
                gen_cfg = {
                    "temperature": float(max(0.0, min(1.0, temperature))),
                    "max_output_tokens": int(max_output_tokens),
                }
                if response_mime_type:
                    gen_cfg["response_mime_type"] = response_mime_type
                resp = model_obj.generate_content(
                    prompt,
                    generation_config=gen_cfg,
                )
                text_out = self._extract_text(resp)
                if text_out and text_out.strip():
                    return text_out.strip()
                # Se vier vazio ou sem Parts, inspeciona finish_reason e tenta próximo candidato
                try:
                    c0 = getattr(resp, "candidates", [None])[0]
                    finish_reason = getattr(c0, "finish_reason", None)
                except Exception:
                    finish_reason = None
                last_error = RuntimeError(f"Empty content from model '{model_to_use}', finish_reason={finish_reason}")
                continue
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
                if "rate" in msg or "quota" in msg or "limit" in msg:
                    time.sleep(0.8)
                    try:
                        model_obj = genai.GenerativeModel(model_to_use)
                        gen_cfg = {
                            "temperature": float(max(0.0, min(1.0, temperature))),
                            "max_output_tokens": int(max_output_tokens),
                        }
                        if response_mime_type:
                            gen_cfg["response_mime_type"] = response_mime_type
                        resp = model_obj.generate_content(
                            prompt,
                            generation_config=gen_cfg,
                        )
                        text_out = self._extract_text(resp)
                        if text_out and text_out.strip():
                            return text_out.strip()
                        continue
                    except Exception:
                        continue
                continue
        if last_error:
            raise last_error
        return ""