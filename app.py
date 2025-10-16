import os
import json
import hashlib
import streamlit as st
from streamlit_authenticator.utilities.exceptions import LoginError

from src.text_loader import extract_text
from src.llm_client import GroqLLM
from src.analyzer import analyze_contract
from src.calendar import make_ics_from_dates, make_google_links_from_dates, make_outlook_links_from_dates
from src.rag import retrieve_relevant_chunks
from src.auth import init_authenticator


st.set_page_config(page_title="Analisador de Contratos (IA - GROQ)", layout="wide")


def get_api_key():
    key_in_env = os.environ.get("GROQ_API_KEY", "")
    return key_in_env


def sidebar_config():
    st.sidebar.header("Configurações")
    api_key = st.sidebar.text_input(
        "GROQ API Key",
        value=get_api_key(),
        type="password",
        help=(
            "Informe sua chave de API da Groq. Você também pode definir a variável"
            " de ambiente `GROQ_API_KEY`. A chave não é armazenada pelo app;"
            " é lida apenas localmente."
        ),
    )
    model = st.sidebar.selectbox(
        "Modelo LLM",
        options=[
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
        ],
        index=0,
        help=(
            "Modelos recomendados pela Groq. Os modelos antigos (por ex. 'llama3-8b-8192') "
            "foram descontinuados. Veja mais em https://console.groq.com/docs/deprecations."
        ),
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
        256,
        8000,
        2000,
        256,
        help=(
            "Limite de tamanho da resposta (tokens). Aumentar permite respostas mais longas"
            " e detalhadas, mas consome mais tempo/custo. Se o limite for atingido,"
            " a saída pode ser truncada."
        ),
    )
    return api_key, model, temperature, max_output_tokens


def render_header():
    st.title("Analisador de Contratos com IA (GROQ)")
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


def render_analysis_sections(results: dict):
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
        st.subheader("Resumo jurídico (linguagem simples)")
        resumo = results.get("resumo_juridico", "")
        if resumo:
            st.write(resumo)
        else:
            st.write("Resumo não disponível.")


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

    api_key, model, temperature, max_output_tokens = sidebar_config()

    if not api_key:
        st.info("Informe sua GROQ API Key nas configurações para usar a IA.")

    llm = GroqLLM(api_key=api_key)
    text = render_upload_and_preview()

    if text:
        # Se o texto mudar, limpar resultados anteriores
        try:
            current_text_id = hashlib.md5(text.encode("utf-8")).hexdigest()
            if st.session_state.get("last_text_id") != current_text_id:
                st.session_state["last_text_id"] = current_text_id
                st.session_state.pop("analysis_results", None)
        except Exception:
            pass

        if st.button("Analisar contrato", type="primary"):
            with st.spinner("Analisando contrato com IA (Groq)..."):
                try:
                    results = analyze_contract(
                        text,
                        llm,
                        model=model,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    )
                    st.session_state["analysis_results"] = results
                    st.success("Análise concluída.")
                except Exception as e:
                    st.error(f"Erro durante a análise: {e}")

        saved_results = st.session_state.get("analysis_results")
        if saved_results:
            render_analysis_sections(saved_results)

        render_qa_section(text, llm, model, temperature, max_output_tokens)


if __name__ == "__main__":
    main()