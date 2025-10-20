import os
import json
import hashlib
import streamlit as st
from streamlit_authenticator.utilities.exceptions import LoginError

from src.text_loader import extract_text
from src.llm_client import GroqLLM, GeminiLLM
from src.analyzer import analyze_contract
from src.calendar import make_ics_from_dates, make_google_links_from_dates, make_outlook_links_from_dates
from src.rag import retrieve_relevant_chunks
from src.auth import init_authenticator
from src.export_pdf import generate_pdf_analysis


st.set_page_config(page_title="Analisador de Contratos (IA)", layout="wide")


def sidebar_config():
    st.sidebar.header("Configurações")
    provider = st.sidebar.selectbox("Provedor de IA", ["Groq", "Gemini"], index=0)

    if provider == "Groq":
        api_key = st.sidebar.text_input(
            "GROQ API Key",
            value=os.environ.get("GROQ_API_KEY", ""),
            type="password",
            help=(
                "Informe sua chave de API da Groq. Você também pode definir a variável"
                " de ambiente `GROQ_API_KEY`. A chave não é armazenada pelo app;"
                " é lida apenas localmente."
            ),
        )
        options_models = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
        ]
        default_model = "llama-3.3-70b-versatile"
        model_help = (
            "Modelos recomendados pela Groq. Os modelos antigos (por ex. 'llama3-8b-8192') "
            "foram descontinuados. Veja mais em https://console.groq.com/docs/deprecations."
        )
    else:
        api_key = st.sidebar.text_input(
            "Gemini API Key",
            value=os.environ.get("GEMINI_API_KEY", ""),
            type="password",
            help=(
                "Informe sua chave de API do Gemini (Google). Você também pode definir a variável"
                " de ambiente `GEMINI_API_KEY`. A chave não é armazenada pelo app;"
                " é lida apenas localmente."
            ),
        )
        options_models = [
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
        ]
        default_model = "gemini-2.5-flash-lite"
        model_help = (
            "Modelos Gemini (v1beta). Se ocorrer 404/not supported, tente outra variante ou use o botão de listagem."
        )
        try:
            import google.generativeai as genai  # opcional
            if api_key:
                genai.configure(api_key=api_key)
                if st.sidebar.button("Listar modelos disponíveis (Gemini)", help="Consulta modelos na sua conta e métodos suportados"):
                    with st.spinner("Consultando modelos do Gemini..."):
                        try:
                            models = list(genai.list_models())
                            supports = []
                            for m in models:
                                name = getattr(m, "name", "")
                                methods = set(getattr(m, "supported_generation_methods", []) or [])
                                supported = "generateContent" in methods or "generate_content" in methods
                                supports.append((name, supported))
                            st.sidebar.success("Modelos consultados. Veja abaixo os que suportam geração de conteúdo.")
                            st.sidebar.markdown("**Modelos compatíveis**")
                            for name, ok in supports:
                                if ok:
                                    st.sidebar.write(f"• {name}")
                        except Exception as e:
                            st.sidebar.error(f"Falha ao listar modelos: {e}")
        except Exception:
            pass

    model = st.sidebar.selectbox(
        "Modelo LLM",
        options=options_models,
        index=(options_models.index(default_model) if default_model in options_models else 0),
        help=model_help,
    )
    temperature = st.sidebar.slider(
        "Temperatura",
        0.0,
        1.0,
        0.2,
        0.05,
        help=(
            "Controla a criatividade/variação das respostas. Valores baixos (0.0–0.3)"
            " geram respostas mais objetivas e estáveis. Valores altos (≥0.7)"
            " tornam as respostas mais criativas e menos determinísticas."
        ),
    )
    max_output_tokens = st.sidebar.slider(
        "Máx. tokens de saída",
        200,
        4096,
        2000,
        50,
        help=(
            "Limite de tokens na resposta gerada. Reduza se encontrar limites do provedor."
        ),
    )
    max_chunks = st.sidebar.slider(
        "Máx. chunks para textos longos",
        4,
        24,
        12,
        1,
        help=(
            "Para contratos muito grandes, limita quantos trechos serão analisados para evitar limites."
        ),
    )
    return provider, api_key, model, temperature, max_output_tokens, max_chunks


def render_header():
    st.title("Analisador de Contratos com IA")
    st.caption(
        "Faça upload de um contrato em PDF ou DOCX e obtenha extração de informações, detecção de cláusulas padrão e desvios, análise de risco, resumo jurídico e campo de perguntas."
    )


def render_upload_and_preview():
    uploaded = st.file_uploader("Envie o contrato (PDF ou DOCX)", type=["pdf", "docx"], accept_multiple_files=False)
    if uploaded:
        try:
            text = extract_text(uploaded)
        except Exception as e:
            st.error(f"Falha ao extrair texto: {e}")
            return None

        st.success("Arquivo carregado e texto extraído com sucesso.")
        with st.expander("Pré-visualização do texto extraído (parcial)", expanded=False):
            st.text(text[:4000] + ("\n..." if len(text) > 4000 else ""))
        return text
    return None


def _metric(value, label):
    col = st.container()
    with col:
        st.metric(label=label, value=value)


def render_analysis_sections(
    results: dict,
    text: str,
    llm: GroqLLM,
    model: str,
    temperature: float,
    max_output_tokens: int,
):
    if not results:
        st.info("Nenhum resultado para exibir.")
        return

    # Persistir a aba ativa durante reruns
    tab_labels = [
        "Resumo",
        "Datas de vencimento",
        "Valores e Multas",
        "Partes envolvidas",
        "Cláusulas comprometedoras",
        "Cláusulas padrão e desvios",
        "Análise de risco",
        "Resumo jurídico",
    ]
    active_tab = st.session_state.get("active_tab", tab_labels[0])
    ordered_labels = [active_tab] + [l for l in tab_labels if l != active_tab]
    tabs_list = st.tabs(ordered_labels)
    tabs_by_label = {label: tabs_list[i] for i, label in enumerate(ordered_labels)}

    with tabs_by_label["Resumo"]:
        st.subheader("Resumo da análise")
        datas = results.get("datas_vencimento", [])
        valores = results.get("valores_multas", [])
        partes = results.get("partes", [])
        comp = results.get("clausulas_comprometedoras", [])
        padrao = results.get("clausulas_padrao", [])
        risco = results.get("analise_risco", {})

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Datas de vencimento", len(datas))
        with c2:
            st.metric("Valores/Multas", len(valores))
        with c3:
            st.metric("Partes", len(partes))
        with c4:
            st.metric("Cláusulas de risco", len(comp))

        c5, c6 = st.columns(2)
        with c5:
            nota = risco.get("risco_geral_nota")
            st.metric("Nota de risco (1-5)", nota if nota is not None else "-")
        with c6:
            top_riscos = risco.get("top_riscos", [])
            if top_riscos:
                st.markdown("**Principais riscos**")
                for r in top_riscos:
                    st.write(f"• {r}")
            else:
                st.write("Sem riscos destacados.")

        st.divider()
        st.subheader("Exportar análise completa")
        resumo_por_clausulas = st.session_state.get("resumo_por_clausulas")
        pdf_bytes = None
        try:
            pdf_bytes = generate_pdf_analysis(
                results,
                resumo_por_clausulas=resumo_por_clausulas,
                resumo_detalhado=st.session_state.get("resumo_detalhado"),
            )
        except Exception as e:
            st.warning(f"Falha ao preparar PDF: {e}")
        st.download_button(
            label="Baixar análise completa (PDF)",
            data=pdf_bytes or b"",
            file_name="analise_contrato.pdf",
            mime="application/pdf",
            disabled=pdf_bytes is None,
        )

    with tabs_by_label["Datas de vencimento"]:
        st.subheader("Datas de vencimento")
        datas = results.get("datas_vencimento", [])
        if not datas:
            st.write("Nenhuma data encontrada.")
        else:
            st.dataframe(datas, use_container_width=True)

            st.divider()
            st.subheader("Exportar para calendário")
            titulo_base = st.text_input("Título base do evento", value="Vencimento de contrato")
            detalhes = st.text_area("Detalhes/Descrição do evento", value="Gerado pelo Analisador de Contratos")
            incluir_sem_data = st.checkbox("Incluir entradas sem 'data_iso' (serão ignoradas no calendário)", value=False)

            validas = [d for d in datas if d.get("data_iso")] if not incluir_sem_data else datas
            if st.button("Gerar links e arquivo ICS", disabled=len(validas) == 0):
                # Garantir que após o rerun continue nesta aba
                st.session_state["active_tab"] = "Datas de vencimento"
                with st.spinner("Gerando links e arquivo ICS..."):
                    links = make_google_links_from_dates(validas, titulo_base=titulo_base, detalhes=detalhes)
                    outlook_links = make_outlook_links_from_dates(validas, titulo_base=titulo_base, detalhes=detalhes)
                    ics_content = make_ics_from_dates(validas, titulo_base=titulo_base, detalhes=detalhes)
                    st.success("Links e arquivo ICS gerados.")
                    st.markdown("**Links de calendário (Google + Outlook)**")

                    # Mapear por descrição para montar linhas combinadas
                    def _key(desc: str, date_iso: str) -> str:
                        return f"{(desc or '').strip()}|{(date_iso or '').strip()}"

                    g_map = {_key(i.get("descricao"), i.get("date_iso")): i.get("link") for i in links}
                    o_map = {
                        _key(i.get("descricao"), i.get("date_iso")): {"live": i.get("live"), "office": i.get("office")}
                        for i in outlook_links
                    }

                    for v in validas:
                        desc = v.get("descricao") or v.get("data_iso") or "Data de vencimento"
                        key = _key(v.get("descricao"), v.get("data_iso"))
                        g = g_map.get(key)
                        o = o_map.get(key, {})
                        live = o.get("live")
                        office = o.get("office")
                        parts = []
                        if live:
                            parts.append(f"[Outlook.com]({live})")
                        if office:
                            parts.append(f"[Outlook365.com]({office})")
                        if g:
                            parts.append(f"[Agenda.com]({g})")
                        # Se algum link estiver ausente, ainda mostramos os disponíveis
                        if parts:
                            st.markdown(f"- {desc}: " + " | ".join(parts))
                    st.download_button(
                        label="Baixar arquivo .ics",
                        data=ics_content,
                        file_name="vencimentos_contrato.ics",
                        mime="text/calendar",
                    )

    with tabs_by_label["Valores e Multas"]:
        st.subheader("Valores e Multas")
        valores = results.get("valores_multas", [])
        if valores:
            st.dataframe(valores, use_container_width=True)
        else:
            st.write("Nenhum valor/multa encontrado.")

    with tabs_by_label["Partes envolvidas"]:
        st.subheader("Partes envolvidas")
        partes = results.get("partes", [])
        if partes:
            st.dataframe(partes, use_container_width=True)
        else:
            st.write("Partes não identificadas claramente.")

    with tabs_by_label["Cláusulas comprometedoras"]:
        st.subheader("Cláusulas que podem comprometer as partes")
        comp = results.get("clausulas_comprometedoras", [])
        if comp:
            st.dataframe(comp, use_container_width=True)
        else:
            st.write("Nenhuma cláusula potencialmente comprometedora destacada.")

    with tabs_by_label["Cláusulas padrão e desvios"]:
        st.subheader("Cláusulas padrão e desvios")
        padrao = results.get("clausulas_padrao", [])
        if padrao:
            st.dataframe(padrao, use_container_width=True)
        else:
            st.write("Nenhuma cláusula padrão encontrada ou analisada.")

    with tabs_by_label["Análise de risco"]:
        st.subheader("Análise de risco")
        risco = results.get("analise_risco", {})
        if risco:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Nota de risco (1-5)", risco.get("risco_geral_nota", "-"))
            with c2:
                top_riscos = risco.get("top_riscos", [])
                st.markdown("**Principais riscos**")
                if top_riscos:
                    for r in top_riscos:
                        st.write(f"• {r}")
                else:
                    st.write("Sem riscos destacados.")
        else:
            st.write("Análise de risco não disponível.")

    with tabs_by_label["Resumo jurídico"]:
        st.subheader("Resumo jurídico")
        resumo = results.get("resumo_juridico", "")
        if resumo:
            st.write(resumo)
        else:
            st.write("Resumo não disponível.")

        st.divider()
        st.markdown("**Resumo detalhado (opcional)**")
        if st.button("Gerar resumo detalhado", key="btn_resumo_detalhado", disabled=not bool(text)):
            st.session_state["active_tab"] = "Resumo jurídico"
            with st.spinner("Gerando resumo detalhado com IA..."):
                try:
                    # Montar prompt para resumo detalhado priorizando trechos relevantes para reduzir tokens
                    sys_prompt = (
                        "Você é um assistente jurídico em português. Gere um resumo detalhado do contrato, "
                        "claro e estruturado. Inclua: obrigações de cada parte, prazos importantes, valores e multas, "
                        "mecanismos de rescisão, garantias, foro, riscos relevantes e pontos de atenção. Evite linguagem excessivamente técnica."
                    )
                    try:
                        top_chunks = retrieve_relevant_chunks("Resumo detalhado do contrato", text, top_k=6)
                        context = "\n\n".join(top_chunks)
                    except Exception:
                        context = (text or "")[:6000]
                    user_content = (
                        "Trechos relevantes:\n" + context + "\n\n"
                        "Resultados extraídos (JSON):\n" + json.dumps(results, ensure_ascii=False) + "\n\n"
                        "Produza um texto corrido, com seções e marcadores quando útil, sem inventar informações não presentes."
                    )
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_content},
                    ]
                    detailed = llm.complete(
                        messages,
                        model=model,
                        temperature=max(0.1, min(temperature, 0.7)),
                        max_output_tokens=min(max_output_tokens, 1200),
                    )
                    st.session_state["resumo_detalhado"] = detailed
                    st.success("Resumo detalhado gerado.")
                except Exception as e:
                    msg = str(e)
                    low = msg.lower()
                    if ("rate_limit" in low) and ("tokens per day" in low or "tpd" in low):
                        st.warning(
                            "Limite diário de tokens atingido. Aguarde alguns minutos ou reduza o custo: "
                            "use modelo menor, diminua os tokens de saída e o limite de chunks."
                        )
                    else:
                        st.error(f"Falha ao gerar resumo detalhado: {e}")

        st.markdown("**Resumo por cláusulas (objetivo)**")
        if st.button("Gerar resumo por cláusulas", key="btn_resumo_clausulas", disabled=not bool(text)):
            st.session_state["active_tab"] = "Resumo jurídico"
            with st.spinner("Gerando resumo por cláusulas com IA..."):
                try:
                    comp = results.get("clausulas_comprometedoras", []) or []
                    padrao = results.get("clausulas_padrao", []) or []

                    sections = []
                    # Gera um resumo por cláusula, baseado estritamente no texto da própria cláusula
                    for c in comp:
                        titulo = c.get("titulo") or c.get("tipo") or "Cláusula"
                        trecho = (c.get("texto_origem") or "").strip()
                        sys_prompt = (
                            "Você é um analista jurídico. Resuma cada cláusula de forma OBJETIVA, "
                            "SEM inventar informações. USE APENAS o texto fornecido da cláusula. "
                            "Se algo não estiver no texto, escreva 'não informado'. Formato EXATO:\n"
                            "- Obrigações: <texto>\n- Condições: <texto>\n- Penalidades: <texto>\n- Riscos: <texto>\n"
                        )
                        user_content = f"Cláusula: {titulo}\nTexto da cláusula:\n{trecho}"
                        messages = [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_content},
                        ]
                        summary = llm.complete(
                            messages,
                            model=model,
                            temperature=max(0.0, min(temperature, 0.2)),
                            max_output_tokens=min(max_output_tokens, 220),
                        )
                        sections.append(f"### {titulo}\n{summary.strip()}")

                    for c in padrao:
                        titulo = c.get("tipo") or c.get("titulo") or "Cláusula"
                        trecho = (c.get("texto_origem") or c.get("desvio") or "").strip()
                        sys_prompt = (
                            "Você é um analista jurídico. Resuma cada cláusula de forma OBJETIVA, "
                            "SEM inventar informações. USE APENAS o texto fornecido da cláusula. "
                            "Se algo não estiver no texto, escreva 'não informado'. Formato EXATO:\n"
                            "- Obrigações: <texto>\n- Condições: <texto>\n- Penalidades: <texto>\n- Riscos: <texto>\n"
                        )
                        user_content = f"Cláusula: {titulo}\nTexto da cláusula:\n{trecho}"
                        messages = [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_content},
                        ]
                        summary = llm.complete(
                            messages,
                            model=model,
                            temperature=max(0.0, min(temperature, 0.2)),
                            max_output_tokens=min(max_output_tokens, 200),
                        )
                        sections.append(f"### {titulo}\n{summary.strip()}")

                    clause_summary_md = "\n\n".join(sections)
                    st.session_state["resumo_por_clausulas"] = clause_summary_md
                    st.success("Resumo por cláusulas gerado.")
                except Exception as e:
                    st.error(f"Falha ao gerar resumo por cláusulas: {e}")

        if st.session_state.get("resumo_por_clausulas"):
            st.markdown("**Resumo por cláusulas (objetivo)**")
            st.markdown(st.session_state.get("resumo_por_clausulas"))
            st.download_button(
                label="Baixar resumo por cláusulas (.md)",
                data=st.session_state.get("resumo_por_clausulas"),
                file_name="resumo_por_clausulas.md",
                mime="text/markdown",
            )

        if st.session_state.get("resumo_detalhado"):
            st.markdown("**Resumo detalhado**")
            st.write(st.session_state.get("resumo_detalhado"))
            st.download_button(
                label="Baixar resumo detalhado (.md)",
                data=st.session_state.get("resumo_detalhado"),
                file_name="resumo_detalhado.md",
                mime="text/markdown",
            )


def render_qa_section(text: str, llm: GroqLLM, model: str, temperature: float, max_output_tokens: int):
    st.header("Perguntas sobre o contrato")
    question = st.text_input("Digite sua pergunta")
    if st.button("Responder", disabled=not bool(text)):
        if not text:
            st.warning("Carregue um contrato primeiro.")
            return
        with st.spinner("Buscando trechos relevantes e consultando a IA..."):
            top_chunks = retrieve_relevant_chunks(question, text, top_k=5)
            context = "\n\n".join(top_chunks)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Você é um assistente jurídico em português. Responda com base nos trechos"
                        " do contrato fornecidos abaixo. Se a resposta não estiver claramente no"
                        " contrato, diga explicitamente que não há evidência suficiente. Seja"
                        " preciso, cite trechos quando possível."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Pergunta: {question}\n\nTrechos relevantes selecionados:\n{context}\n\n"
                        "Se necessário, considere o restante do contrato, mas priorize os trechos."
                    ),
                },
            ]
            answer = llm.complete(messages, model=model, temperature=temperature, max_output_tokens=max_output_tokens)
            st.markdown("**Resposta:**")
            st.write(answer)


def main():
    render_header()

    # Autenticação
    authenticator = init_authenticator()
    # Compatível com versões recentes do streamlit-authenticator: login não retorna tupla,
    # armazena em st.session_state ('name', 'authentication_status', 'username').
    try:
        authenticator.login(location="main")
    except LoginError:
        # Tratar falhas de autorização sem quebrar a aplicação
        st.error("Usuário não autorizado. Verifique usuário/senha ou permissões.")
        return
    authentication_status = st.session_state.get("authentication_status")
    name = st.session_state.get("name")
    username = st.session_state.get("username")

    if authentication_status is False:
        st.error("Usuário ou senha incorretos.")
        return
    elif authentication_status is None:
        st.info("Informe seu usuário e senha para acessar o analisador.")
        return

    # Usuário autenticado
    authenticator.logout("Sair", location="sidebar")
    st.sidebar.success(f"Logado como: {name}")

    provider, api_key, model, temperature, max_output_tokens, max_chunks = sidebar_config()

    if not api_key:
        st.info("Informe sua API Key do provedor selecionado nas configurações para usar a IA.")

    llm = GroqLLM(api_key=api_key) if provider == "Groq" else GeminiLLM(api_key=api_key)
    text = render_upload_and_preview()

    if text:
        # Se o texto mudar, limpar resultados anteriores
        try:
            current_text_id = hashlib.md5(text.encode("utf-8")).hexdigest()
            if st.session_state.get("last_text_id") != current_text_id:
                st.session_state["last_text_id"] = current_text_id
                st.session_state.pop("analysis_results", None)
                st.session_state.pop("resumo_detalhado", None)
        except Exception:
            pass

        if st.button("Analisar contrato", type="primary"):
            with st.spinner("Analisando contrato com IA (Groq)..."):
                try:
                    # Barra de progresso para contratos longos
                    progress_bar = st.progress(0)
                    def _progress(done: int, total: int):
                        try:
                            frac = 0.0
                            if total > 0:
                                frac = min(1.0, done / total)
                            progress_bar.progress(frac)
                        except Exception:
                            pass

                    results = analyze_contract(
                        text,
                        llm,
                        model=model,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        max_chunks=max_chunks,
                        progress_hook=_progress,
                    )
                    st.session_state["analysis_results"] = results
                    st.success("Análise concluída.")
                except Exception as e:
                    st.error(f"Erro durante a análise: {e}")

        saved_results = st.session_state.get("analysis_results")
        if saved_results:
            render_analysis_sections(saved_results, text, llm, model, temperature, max_output_tokens)

        render_qa_section(text, llm, model, temperature, max_output_tokens)


if __name__ == "__main__":
    main()