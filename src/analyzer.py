import json
import re
from typing import Dict, Any, Tuple, Optional, Callable

from .prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_user_prompt
from .llm_client import GroqLLM
from .rag import _chunk_text, retrieve_relevant_chunks


def _clean_output(raw: str) -> str:
    """Remove cercas de código e espaços extras."""
    if not raw:
        return ""
    cleaned = raw.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "")
    return cleaned


def _clean_summary_text(text: str) -> str:
    """Normaliza o texto do resumo para melhor leitura.
    - Remove quebras e espaços irregulares (inclui \n, \t, NBSP)
    - Corrige acentos quebrados (ex.: "a ˊ" -> "á", "a ˋ" -> "à")
    - Une letras separadas por espaços (ex.: "E M P R E E N D E D O R A" -> "EMPREENDEDORA")
    - Ajusta pontuação e moeda BRL e insere parágrafos legíveis
    """
    if not text:
        return ""
    s = text
    try:
        # Remover espaços especiais e zero-width
        s = s.replace("\u00A0", " ")
        s = re.sub(r"[\u200B\u200C\u200D]", "", s)
        # Corrigir acentos escritos como letra + marcador
        acute_map = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú", "A": "Á", "E": "É", "I": "Í", "O": "Ó", "U": "Ú"}
        def _acute_repl(m):
            return acute_map.get(m.group(1), m.group(1))
        s = re.sub(r"([AaEeIiOoUu])\s*[ˊ´]", _acute_repl, s)
        s = re.sub(r"([Aa])\s*ˋ", lambda m: "À" if m.group(1).isupper() else "à", s)
        # Remover marcadores remanescentes
        s = s.replace("ˋ", "").replace("ˊ", "")
        # Unir sequências de letras separadas por espaços: "p e l a" -> "pela"; "E M P..." -> "EMP..."
        alpha = r"A-Za-zÀ-ÖØ-öø-ÿ"
        s = re.sub(
            rf"(?:\b[{alpha}]\b(?:\s+\b[{alpha}]\b){{2,}})",
            lambda m: "".join(re.findall(rf"[{alpha}]", m.group(0))),
            s,
        )
        # Colapsar qualquer whitespace em um único espaço
        s = re.sub(r"\s+", " ", s)
        # Espaço entre minúscula e MAIÚSCULA grudadas
        s = re.sub(r"([a-záéíóúãõç])([A-ZÁÉÍÓÚÃÕÇ])", r"\1 \2", s)
        # Ajuste de pontuação (sem espaço antes; um espaço depois)
        s = re.sub(r"\s*([,.;:])\s*", r"\1 ", s)
        # Remover espaços dentro de números
        s = re.sub(r"(?<=\d)\s+(?=[\.,])", "", s)
        s = re.sub(r"(?<=[\.,])\s+(?=\d)", "", s)
        s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)
        # Normalizar moeda BRL
        s = re.sub(r"\bR\s*(?=\d)", "R$ ", s)
        # Paragrafar por padrões recorrentes para leitura
        leads = [
            "O contrato estabelece",
            "As partes",
            "A TERRENISTA",
            "A EMPREENDEDORA",
        ]
        for phrase in leads:
            s = re.sub(rf"(?<!^)\s+({phrase}\b)", r"\n\n\1", s)
        # Espaços finais
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip()
    except Exception:
        return text


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
    max_chunks: int = 12,
    progress_hook: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    # Fallback: se o contrato for muito grande, analisa em chunks para evitar limites de tokens/TPM
    text_len = len(contract_text or "")
    if text_len > 12000:
        chunks = _chunk_text(contract_text, max_chars=1400)
        total = len(chunks)
        # Se houver muitos chunks, limitar a quantidade para evitar travamentos
        if total > max_chunks:
            # Seleção simples: usar apenas os primeiros ou selecionar relevantes
            try:
                question = (
                    "Extrair datas de vencimento, multas, partes, cláusulas padrão/comprometedoras, riscos e resumo"
                )
                relevant = retrieve_relevant_chunks(question, contract_text, top_k=max_chunks)
                chunks = relevant
                total = len(chunks)
            except Exception:
                chunks = chunks[:max_chunks]
                total = len(chunks)

        aggregated = {
            "datas_vencimento": [],
            "valores_multas": [],
            "partes": [],
            "clausulas_comprometedoras": [],
            "clausulas_padrao": [],
            "analise_risco": {},
            "resumo_juridico": "",
        }
        summaries = []
        seen_venc = set()
        seen_multas = set()
        seen_partes = set()
        seen_comp = set()
        seen_padrao = set()

        for idx, ch in enumerate(chunks):
            messages = [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": build_extraction_user_prompt(ch)},
            ]
            output = llm.complete(messages, model=model, temperature=temperature, max_output_tokens=max_output_tokens, response_mime_type="application/json")
            data_chunk = _safe_json_loads(_clean_output(output))

            if _is_empty_result(data_chunk):
                strict_messages = [
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT + " Responda SOMENTE com JSON válido, sem markdown e sem texto fora."},
                    {"role": "user", "content": build_extraction_user_prompt(ch) + "\nRetorne apenas o JSON começando com '{' e terminando com '}'."},
                ]
                output2 = llm.complete(strict_messages, model=model, temperature=max(0.0, temperature - 0.1), max_output_tokens=max_output_tokens, response_mime_type="application/json")
                data_chunk = _safe_json_loads(_clean_output(output2))

            data_chunk = _ensure_schema(data_chunk)

            # Merge datas_vencimento
            for it in data_chunk.get("datas_vencimento", []) or []:
                key = (it.get("descricao"), it.get("data_iso"))
                if key not in seen_venc:
                    aggregated["datas_vencimento"].append(it)
                    seen_venc.add(key)

            # Merge valores_multas
            for it in data_chunk.get("valores_multas", []) or []:
                key = (it.get("tipo"), it.get("percentual"), it.get("valor_monetario"), it.get("texto_origem"))
                if key not in seen_multas:
                    aggregated["valores_multas"].append(it)
                    seen_multas.add(key)

            # Merge partes
            for it in data_chunk.get("partes", []) or []:
                key = (it.get("nome"), it.get("papel"))
                if key not in seen_partes:
                    aggregated["partes"].append(it)
                    seen_partes.add(key)

            # Merge clausulas_comprometedoras
            for it in data_chunk.get("clausulas_comprometedoras", []) or []:
                key = (it.get("titulo"), it.get("parte_afetada"), it.get("gravidade"), it.get("texto_origem"))
                if key not in seen_comp:
                    aggregated["clausulas_comprometedoras"].append(it)
                    seen_comp.add(key)

            # Merge clausulas_padrao
            for it in data_chunk.get("clausulas_padrao", []) or []:
                key = (it.get("tipo"), it.get("presente"), it.get("texto_origem"))
                if key not in seen_padrao:
                    aggregated["clausulas_padrao"].append(it)
                    seen_padrao.add(key)

            # Coleta resumo do chunk, se houver
            summary = data_chunk.get("resumo_juridico")
            if summary:
                summaries.append(summary.strip())

            # Progresso
            if progress_hook:
                try:
                    progress_hook(idx + 1, total)
                except Exception:
                    pass

        # Constrói resumo final a partir dos resumos parciais (limitado)
        if summaries:
            aggregated["resumo_juridico"] = _clean_summary_text("\n\n".join(summaries)[:4000])

        aggregated = _ensure_schema(aggregated)
        aggregated["partes"] = _normalize_partes(aggregated.get("partes", []))
        aggregated = _normalize_values_multas(aggregated)
        aggregated = _expand_vencimento_dates(aggregated)
        return aggregated

    # Contratos pequenos: comportamento original
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": build_extraction_user_prompt(contract_text)},
    ]
    output = llm.complete(messages, model=model, temperature=temperature, max_output_tokens=max_output_tokens, response_mime_type="application/json")
    data = _safe_json_loads(_clean_output(output))

    # Se resultado vazio ou sem as principais listas, tenta uma segunda chamada mais estrita
    if _is_empty_result(data):
        strict_messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT + " Responda SOMENTE com JSON válido, sem markdown e sem texto fora."},
            {"role": "user", "content": build_extraction_user_prompt(contract_text) + "\nRetorne apenas o JSON começando com '{' e terminando com '}'."},
        ]
        output2 = llm.complete(strict_messages, model=model, temperature=max(0.0, temperature - 0.1), max_output_tokens=max_output_tokens, response_mime_type="application/json")
        data = _safe_json_loads(_clean_output(output2))

    data = _ensure_schema(data)
    # Limpeza do resumo para legibilidade
    data["resumo_juridico"] = _clean_summary_text(data.get("resumo_juridico", ""))

    # Normalização de partes para deduplicação e documentos legíveis
    data["partes"] = _normalize_partes(data.get("partes", []))
    data = _normalize_values_multas(data)
    data = _expand_vencimento_dates(data)
    return data


def _parse_brl_amount(text: str) -> Optional[float]:
    """Extrai o primeiro valor monetário no padrão brasileiro (R$ 1.234,56) do texto.
    Aceita espaços opcionais no valor (ex.: "R$ 2.000.000, 00") e também o símbolo "R" sem "$".
    Retorna float ou None se não encontrado.
    """
    if not text:
        return None
    # Padrões comuns: "R$ 1.234,56", "R$1.234", "R$ 123,45" e variações com espaços: "R$ 1.234, 56"
    # Aceita também "R 100.000,00" e "R2.000.000,00" (sem o cifrão)
    m = re.search(r"R\$?\s*([\d\.\s]+(?:,\s*\d{2})?)", text)
    if not m:
        return None
    raw = m.group(1)
    # Remover espaços dentro do número antes de normalizar
    raw = re.sub(r"\s+", "", raw)
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

            # Tentar extrair sempre que houver texto; parser só reconhece padrões com prefixo "R" (com ou sem "$")
            parsed = _parse_brl_amount(texto)
            if parsed is not None:
                it["valor_monetario"] = _format_brl(parsed)
                it["moeda"] = moeda or "BRL"
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


def _normalize_partes(items):
    """Deduplica e normaliza a lista de partes envolvidas.
    - Agrupa por nome normalizado (case-insensitive, espaços colapsados)
    - Agrega papéis únicos em uma string separada por vírgulas
    - Escolhe o tipo mais frequente (ou o primeiro não-vazio)
    - Converte 'documentos' em string legível, evitando [object Object]
    """
    try:
        parts = items or []
        by_name = {}
        def norm_name(n):
            if not n:
                return ""
            return re.sub(r"\s+", " ", str(n)).strip().lower()
        def doc_to_str(doc):
            try:
                if doc is None:
                    return ""
                if isinstance(doc, str):
                    return doc
                if isinstance(doc, (int, float)):
                    return str(doc)
                if isinstance(doc, dict):
                    # Campos comuns
                    tipo = doc.get("tipo")
                    numero = doc.get("numero")
                    desc = doc.get("descricao")
                    base = ", ".join([x for x in [tipo, numero, desc] if x])
                    return base or json.dumps(doc, ensure_ascii=False)
                if isinstance(doc, (list, tuple)):
                    return "; ".join([doc_to_str(x) for x in doc if x])
                return json.dumps(doc, ensure_ascii=False)
            except Exception:
                return str(doc)
        for it in parts:
            nome = it.get("nome")
            nkey = norm_name(nome)
            rec = by_name.get(nkey)
            tipo = (it.get("tipo") or "").strip()
            papel = (it.get("papel") or "").strip()
            documentos = it.get("documentos")
            doc_str = doc_to_str(documentos)
            if not rec:
                rec = {
                    "nome": (str(nome).strip() if nome else "-"),
                    "tipos": [],
                    "papeis": set(),
                    "docs": [],
                }
                by_name[nkey] = rec
            if tipo:
                rec["tipos"].append(tipo)
            if papel:
                rec["papeis"].add(papel)
            if doc_str:
                rec["docs"].append(doc_str)
        result = []
        for nkey, rec in by_name.items():
            tipos = [t for t in rec.get("tipos", []) if t]
            # tipo mais frequente ou primeiro
            tipo_final = tipos[0] if not tipos else max(set(tipos), key=tipos.count)
            papeis = sorted(list(rec.get("papeis", set())))
            docs = [d for d in rec.get("docs", []) if d]
            result.append({
                "nome": rec.get("nome", "-"),
                "tipo": tipo_final or "-",
                "papel": ", ".join(papeis) if papeis else "-",
                "documentos": "; ".join(docs) if docs else "-",
            })
        # ordenar por nome
        result.sort(key=lambda x: x.get("nome", ""))
        return result
    except Exception:
        return items or []

# Atualiza pontos de retorno para aplicar normalização de partes
def _normalize_partes(items):
    """Deduplica e normaliza a lista de partes envolvidas.
    - Agrupa por nome normalizado (case-insensitive, espaços colapsados)
    - Agrega papéis únicos em uma string separada por vírgulas
    - Escolhe o tipo mais frequente (ou o primeiro não-vazio)
    - Converte 'documentos' em string legível, evitando [object Object]
    """
    try:
        parts = items or []
        by_name = {}
        def norm_name(n):
            if not n:
                return ""
            return re.sub(r"\s+", " ", str(n)).strip().lower()
        def doc_to_str(doc):
            try:
                if doc is None:
                    return ""
                if isinstance(doc, str):
                    return doc
                if isinstance(doc, (int, float)):
                    return str(doc)
                if isinstance(doc, dict):
                    # Campos comuns
                    tipo = doc.get("tipo")
                    numero = doc.get("numero")
                    desc = doc.get("descricao")
                    base = ", ".join([x for x in [tipo, numero, desc] if x])
                    return base or json.dumps(doc, ensure_ascii=False)
                if isinstance(doc, (list, tuple)):
                    return "; ".join([doc_to_str(x) for x in doc if x])
                return json.dumps(doc, ensure_ascii=False)
            except Exception:
                return str(doc)
        for it in parts:
            nome = it.get("nome")
            nkey = norm_name(nome)
            rec = by_name.get(nkey)
            tipo = (it.get("tipo") or "").strip()
            papel = (it.get("papel") or "").strip()
            documentos = it.get("documentos")
            doc_str = doc_to_str(documentos)
            if not rec:
                rec = {
                    "nome": (str(nome).strip() if nome else "-"),
                    "tipos": [],
                    "papeis": set(),
                    "docs": [],
                }
                by_name[nkey] = rec
            if tipo:
                rec["tipos"].append(tipo)
            if papel:
                rec["papeis"].add(papel)
            if doc_str:
                rec["docs"].append(doc_str)
        result = []
        for nkey, rec in by_name.items():
            tipos = [t for t in rec.get("tipos", []) if t]
            # tipo mais frequente ou primeiro
            tipo_final = tipos[0] if not tipos else max(set(tipos), key=tipos.count)
            papeis = sorted(list(rec.get("papeis", set())))
            docs = [d for d in rec.get("docs", []) if d]
            result.append({
                "nome": rec.get("nome", "-"),
                "tipo": tipo_final or "-",
                "papel": ", ".join(papeis) if papeis else "-",
                "documentos": "; ".join(docs) if docs else "-",
            })
        # ordenar por nome
        result.sort(key=lambda x: x.get("nome", ""))
        return result
    except Exception:
        return items or []


def _expand_vencimento_dates(data: Dict[str, Any]) -> Dict[str, Any]:
    """Expande datas de vencimento quando o texto menciona "dia X de cada mês"
    e lista meses com um ano (ex.: "abril, maio, junho... de 2025").
    Cria uma entrada por mês com 'data_iso' = YYYY-MM-X.
    """
    try:
        items = data.get("datas_vencimento", []) or []
        if not items:
            return data

        months_map = {
            "janeiro": 1, "fevereiro": 2,
            "março": 3, "marco": 3,
            "abril": 4, "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
            "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
        }

        new_items = list(items)
        seen = {(i.get("descricao"), i.get("data_iso")) for i in items}

        for it in items:
            text = f"{it.get('descricao', '')} {it.get('texto_origem', '')}".lower()
            # Encontrar o dia
            mday = re.search(r"dia\s+(\d{1,2})", text)
            if not mday:
                continue
            day = int(mday.group(1))

            # Capturar ano próximo a menções de meses
            myear = re.search(r"(de\s*)?(\d{4})", text)
            year = int(myear.group(2)) if myear else None
            if not year:
                # sem ano explícito não expandimos
                continue

            # Encontrar todos os meses citados
            found_months = []
            for name, num in months_map.items():
                if re.search(rf"\b{name}\b", text):
                    found_months.append(num)

            # Também suportar intervalos "de abril a agosto"
            # Procura "de <mes> a <mes>"
            for start_name in months_map.keys():
                m_range = re.search(rf"{start_name}\s*(?:a|até)\s*([a-zç]+)", text)
                if m_range:
                    end_name = m_range.group(1)
                    if end_name in months_map:
                        start_m = months_map[start_name]
                        end_m = months_map[end_name]
                        if start_m <= end_m:
                            found_months.extend(list(range(start_m, end_m + 1)))

            if not found_months:
                continue

            # Deduplicar e criar entradas
            for m in sorted(set(found_months)):
                iso = f"{year}-{m:02d}-{day:02d}"
                key = (it.get("descricao"), iso)
                if key in seen:
                    continue
                new_items.append({
                    "descricao": it.get("descricao") or f"Vencimento mensal dia {day}",
                    "data_iso": iso,
                    "texto_origem": it.get("texto_origem") or it.get("descricao") or "",
                })
                seen.add(key)

        data["datas_vencimento"] = new_items
        return data
    except Exception:
        return data