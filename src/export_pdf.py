import io
from typing import Dict, Any, Optional, List
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
from reportlab.lib import colors


def _p(text: str, styles):
    return Paragraph(text.replace("\n", "<br/>"), styles["BodyText"])  # simples conversão de quebra de linha


def _table(data: List[List[str]]):
    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    return tbl


def _list(items: List[str], styles):
    return ListFlowable([ListItem(Paragraph(i, styles["BodyText"])) for i in items], bulletType="bullet")


def generate_pdf_analysis(
    results: Dict[str, Any],
    resumo_por_clausulas: Optional[str] = None,
    resumo_detalhado: Optional[str] = None,
) -> bytes:
    """Gera PDF da análise completa a partir do dicionário de resultados.
    Retorna bytes para uso em download.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="Análise de Contrato")
    styles = getSampleStyleSheet()
    story = []

    # Título
    story.append(Paragraph("Análise de Contrato", styles["Title"]))
    story.append(Spacer(1, 12))

    # Resumo de métricas
    datas = results.get("datas_vencimento", [])
    valores = results.get("valores_multas", [])
    partes = results.get("partes", [])
    riscos = results.get("analise_risco", {})

    metrics_tbl = _table([
        ["Resumo"],
        [f"Datas de vencimento: {len(datas)}"],
        [f"Valores/Multas: {len(valores)}"],
        [f"Partes: {len(partes)}"],
        [f"Cláusulas de risco: {len(results.get('clausulas_comprometedoras', []))}"],
        [f"Nota de risco (1-5): {riscos.get('risco_geral_nota', '-')}"]
    ])
    story.append(metrics_tbl)
    story.append(Spacer(1, 18))

    # Datas de vencimento
    story.append(Paragraph("Datas de vencimento", styles["Heading2"]))
    if datas:
        rows = [["Descrição", "Data (ISO)"]] + [[d.get("descricao", "-"), d.get("data_iso", "-")] for d in datas]
        story.append(_table(rows))
    else:
        story.append(_p("Nenhuma data encontrada.", styles))
    story.append(Spacer(1, 12))

    # Valores e Multas
    story.append(Paragraph("Valores e Multas", styles["Heading2"]))
    if valores:
        rows = [["Tipo", "Valor", "Percentual", "Moeda"]]
        for v in valores:
            rows.append([
                str(v.get("tipo", "-")),
                str(v.get("valor_monetario", "-")),
                str(v.get("percentual", "-")),
                str(v.get("moeda", "-")),
            ])
        story.append(_table(rows))
    else:
        story.append(_p("Nenhum valor/multa encontrado.", styles))
    story.append(Spacer(1, 12))

    # Partes envolvidas
    story.append(Paragraph("Partes envolvidas", styles["Heading2"]))
    if partes:
        rows = [["Nome", "Tipo", "Papel", "Documentos"]]
        for p in partes:
            rows.append([
                str(p.get("nome", "-")),
                str(p.get("tipo", "-")),
                str(p.get("papel", "-")),
                str(p.get("documentos", "-")),
            ])
        story.append(_table(rows))
    else:
        story.append(_p("Partes não identificadas claramente.", styles))
    story.append(Spacer(1, 12))

    # Cláusulas comprometedoras
    story.append(Paragraph("Cláusulas comprometedoras", styles["Heading2"]))
    comp = results.get("clausulas_comprometedoras", [])
    if comp:
        rows = [["Título", "Parte afetada", "Gravidade", "Origem"]]
        for c in comp:
            rows.append([
                str(c.get("titulo", "-")),
                str(c.get("parte_afetada", "-")),
                str(c.get("gravidade", "-")),
                str(c.get("texto_origem", "-")),
            ])
        story.append(_table(rows))
    else:
        story.append(_p("Nenhuma cláusula potencialmente comprometedora destacada.", styles))
    story.append(Spacer(1, 12))

    # Cláusulas padrão e desvios
    story.append(Paragraph("Cláusulas padrão e desvios", styles["Heading2"]))
    padrao = results.get("clausulas_padrao", [])
    if padrao:
        rows = [["Tipo", "Presente", "Desvio", "Origem"]]
        for c in padrao:
            rows.append([
                str(c.get("tipo", "-")),
                str(c.get("presente", "-")),
                str(c.get("desvio", "-")),
                str(c.get("texto_origem", "-")),
            ])
        story.append(_table(rows))
    else:
        story.append(_p("Nenhuma cláusula padrão encontrada ou analisada.", styles))
    story.append(Spacer(1, 12))

    # Resumo por cláusulas (objetivo)
    if resumo_por_clausulas:
        story.append(Paragraph("Resumo por cláusulas (objetivo)", styles["Heading2"]))
        story.append(_p(resumo_por_clausulas, styles))
        story.append(Spacer(1, 12))

    # Resumo jurídico
    story.append(Paragraph("Resumo jurídico", styles["Heading2"]))
    story.append(_p(results.get("resumo_juridico", "Resumo não disponível."), styles))
    story.append(Spacer(1, 12))

    # Resumo detalhado
    if resumo_detalhado:
        story.append(Paragraph("Resumo detalhado", styles["Heading2"]))
        story.append(_p(resumo_detalhado, styles))
        story.append(Spacer(1, 12))

    doc.build(story)
    return buf.getvalue()