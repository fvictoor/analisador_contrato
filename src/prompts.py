EXTRACTION_SYSTEM_PROMPT = (
    "Você é um analista jurídico especializado em contratos em português (Brasil). "
    "Extraia informações com precisão e responda ESTRITAMENTE em JSON válido, sem markdown. "
    "Use datas em ISO (YYYY-MM-DD) quando possível. Se não houver, use null. "
    "NÃO INFERIR nem CALCULAR valores que não estão presentes no contrato, parcelas ou multas: só registre o que está explicitamente escrito no contrato. Se um valor por parcela não estiver informado, mantenha null. "
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
        "REGRAS: Não calcule nem estime valores (por exemplo, não derive o valor da parcela dividindo o total), "
        "registre apenas números que aparecem literalmente no contrato. Se não houver número explícito, use null. "
        "IMPORTANTE: Responda SOMENTE com JSON válido.\n\n"
        f"Contrato:\n{contract_text}"
    )


QA_SYSTEM_PROMPT = (
    "Você é um assistente jurídico. Responda em português com base nos trechos"
    " fornecidos. Seja preciso e objetivo, cite trechos quando apropriado."
)