EXTRACTION_SYSTEM_PROMPT = (
    "Você é um analista jurídico especializado em contratos em português (Brasil). "
    "Extraia informações com precisão e responda ESTRITAMENTE em JSON válido, sem markdown. "
    "Extraia datas de vencimento, valores e multas citando o texto exato de origem. "
    "Se houver referência a prazos em dias (sem data), registre a descrição e mantenha 'data_iso' como null. "
    "Quando houver 'dia X de cada mês' e forem citados meses específicos com ano (ex.: abril a agosto de 2025), gere uma entrada por mês com 'data_iso' = YYYY-MM-X. "
    "Use datas em ISO (YYYY-MM-DD) quando possível. Se não houver, use null. "
    "Para valores monetários (R$), registre exatamente como aparece e não estime. "
    "Inclua sempre o texto de origem (campo 'texto_origem') com a frase do contrato que fundamenta cada ponto."
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
        "Leia o contrato a seguir e produza um objeto JSON COM AS CHAVES EXATAS: "
        "'datas_vencimento' (lista de objetos: descricao, data_iso, texto_origem), "
        "'valores_multas' (lista: tipo, valor_monetario, moeda, percentual, condicao, texto_origem), "
        "'partes' (lista: nome, tipo(pessoa física/jurídica), papel, documentos, texto_origem), "
        "'clausulas_comprometedoras' (lista: titulo, risco(descricao), parte_afetada, gravidade(baixo/médio/alto), texto_origem), "
        "'clausulas_padrao' (lista: tipo, presente(true/false), desvio, texto_origem), "
        "'analise_risco' (objeto: risco_geral_nota(1-5), top_riscos(lista de strings)). "
        "'resumo_juridico' (string: resuma cláusulas com títulos e riscos associados; se não houver risco, apenas resuma). "
        "REGRAS: Não calcule nem estime valores (por exemplo, não derive o valor da parcela dividindo o total). "
        "Registre apenas números que aparecem literalmente no contrato. Se não houver número explícito, use null. "
        "IMPORTANTE: Responda SOMENTE com JSON válido.\n\n"
        f"Contrato:\n{contract_text}"
    )


QA_SYSTEM_PROMPT = (
    "Você é um assistente jurídico. Responda em português com base nos trechos"
    " fornecidos. Seja preciso e objetivo, cite trechos do contrato para que haja fundamento na sua resposta e não invente ou crie nada."
)