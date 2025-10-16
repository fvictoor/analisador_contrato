import json
import re
from typing import Dict, Any, Tuple, Optional

from .prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_user_prompt
from .llm_client import GroqLLM


def _clean_output(raw: str) -> str:
    """Remove cercas de código e espaços extras."""
    if not raw:
        return ""
    cleaned = raw.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "")
    return cleaned


def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        # Tentativa de recuperar JSON envolto em markdown ou com texto extra
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
        return {}


def analyze_contract(
    contract_text: str,
    llm: GroqLLM,
    model: str = "llama3-8b-8192",
    temperature: float = 0.2,
    max_output_tokens: int = 2000,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": build_extraction_user_prompt(contract_text)},
    ]
    output = llm.complete(messages, model=model, temperature=temperature, max_output_tokens=max_output_tokens)
    data = _safe_json_loads(_clean_output(output))

    # Se resultado vazio ou sem as principais listas, tenta uma segunda chamada mais estrita
    if _is_empty_result(data):
        strict_messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT + " Responda SOMENTE com JSON válido, sem markdown e sem texto fora."},
            {"role": "user", "content": build_extraction_user_prompt(contract_text) + "\nRetorne apenas o JSON começando com '{' e terminando com '}'."},
        ]
        output2 = llm.complete(strict_messages, model=model, temperature=max(0.0, temperature - 0.1), max_output_tokens=max_output_tokens)
        data = _safe_json_loads(_clean_output(output2))

    return _normalize_values_multas(_ensure_schema(data))


def _parse_brl_amount(text: str) -> Optional[float]:
    """Extrai o primeiro valor monetário no padrão brasileiro (R$ 1.234,56) do texto.
    Retorna float ou None se não encontrado.
    """
    if not text:
        return None
    # Padrões comuns: "R$ 1.234,56", "R$1.234", "R$ 123,45"
    m = re.search(r"R\$\s*([\d\.]+(?:,[\d]{2})?)", text)
    if not m:
        return None
    raw = m.group(1)
    # Converter para float: remover pontos (milhares) e trocar vírgula por ponto
    normalized = raw.replace(".", "").replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return None


def _format_brl(value: float) -> str:
    """Formata float em moeda BRL com milhares e 2 casas decimais."""
    # Usa formatação US e ajusta para notação PT-BR
    s = f"{value:,.2f}"
    return "R$ " + s.replace(",", "@").replace(".", ",").replace("@", ".")


def _normalize_values_multas(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normaliza 'valores_multas': tenta preencher valor_monetario e moeda a partir do texto.
    Também aplica formatação BRL para melhorar exibição na UI.
    """
    try:
        items = data.get("valores_multas", []) or []
        normalized = []
        for it in items:
            it = dict(it)  # cópia defensiva
            valor = it.get("valor_monetario")
            moeda = it.get("moeda")
            texto = it.get("texto_origem", "")

            # Se não há valor e há texto com R$, tentar extrair
            if (valor is None or valor == "" or str(valor).lower() == "none") and "R$" in texto:
                parsed = _parse_brl_amount(texto)
                if parsed is not None:
                    it["valor_monetario"] = _format_brl(parsed)
                    it["moeda"] = moeda or "BRL"
                else:
                    # mantém None se não foi possível
                    pass
            else:
                # Se já veio número, formatar
                try:
                    if isinstance(valor, (int, float)):
                        it["valor_monetario"] = _format_brl(float(valor))
                        it["moeda"] = moeda or "BRL"
                except Exception:
                    pass

            normalized.append(it)
        data["valores_multas"] = normalized
        return data
    except Exception:
        return data


def _ensure_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """Garante que todas as chaves esperadas existam com tipos padrão."""
    base = dict(data or {})
    base.setdefault("datas_vencimento", [])
    base.setdefault("valores_multas", [])
    base.setdefault("partes", [])
    base.setdefault("clausulas_comprometedoras", [])
    base.setdefault("clausulas_padrao", [])
    base.setdefault("analise_risco", {})
    base.setdefault("resumo_juridico", "")
    return base


def _is_empty_result(data: Dict[str, Any]) -> bool:
    if not data:
        return True
    try:
        return not any([
            bool(data.get("datas_vencimento")),
            bool(data.get("valores_multas")),
            bool(data.get("partes")),
            bool(data.get("clausulas_comprometedoras")),
            bool(data.get("clausulas_padrao")),
            bool(data.get("resumo_juridico")),
        ])
    except Exception:
        return True