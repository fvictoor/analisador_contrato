EXTRACTION_SYSTEM_PROMPT = (
    "Você é um analista jurídico especializado em contratos em português (Brasil). "
    "Extraia informações com precisão e responda ESTRITAMENTE em JSON válido, sem markdown. "
    "Use datas em ISO (YYYY-MM-DD) quando possível. Se não houver, use null."
)

STANDARD_CLAUSES = [
    "Confidencialidade",
    "Prazo e Rescisão",
    "Multa por atraso",
    "Garantias",
    "Força maior",
    "Propriedade intelectual",
    "Não concorrência",
    "Resolução de disputas / Foro",
    "Indenização / Limitação de responsabilidade",
    "Proteção de dados pessoais / LGPD",
]


def build_extraction_user_prompt(contract_text: str) -> str:
    return (
        "Leia o contrato a seguir e produza um objeto JSON com os campos: "
        "'datas_vencimento' (lista de objetos: descricao, data_iso, texto_origem), "
        "'valores_multas' (lista: tipo, valor_monetario, moeda, percentual, condicao, texto_origem), "
        "'partes' (lista: nome, tipo(pessoa física/jurídica), papel, documentos), "
        "'clausulas_comprometedoras' (lista: titulo, risco(descrição), parte_afetada, gravidade(baixo/médio/alto), texto_origem), "
        "'clausulas_padrao' (lista: tipo, presente(true/false), desvio(descrição se houver), texto_origem), "
        "'analise_risco' (objeto: risco_geral_nota(1-5), top_riscos(lista de descrições curtas)), "
        "'resumo_juridico' (string: resumo em linguagem simples para leitura e interpretação). "
        "Considere como cláusulas padrão esta lista: "
        f"{', '.join(STANDARD_CLAUSES)}. "
        "Destaque qualquer desvio das cláusulas usuais. "
        "IMPORTANTE: Responda SOMENTE com JSON válido.\n\n"
        f"Contrato:\n{contract_text}"
    )


QA_SYSTEM_PROMPT = (
    "Você é um assistente jurídico. Responda em português com base nos trechos"
    " fornecidos. Seja preciso e objetivo, cite trechos quando apropriado."
)