
import os
import time
from openai import OpenAI
from config import Config

client = OpenAI(api_key=Config.OPENAI_API_KEY)

def analisar_contrato(texto):
    assistant_id = Config.ASSISTANT_ID
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"Analise este contrato e extraia informações relevantes em formato JSON: {texto}"
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id
    )

    timeout = 300
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError("Tempo de análise excedido")
        status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if status.status == "completed":
            break
        elif status.status == "failed":
            raise RuntimeError(f"Falha na análise: {status.last_error.message if status.last_error else 'Erro desconhecido'}")
        elif status.status in ["cancelled", "expired"]:
            raise RuntimeError(f"Processo {status.status}")
        time.sleep(5)

    messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc", limit=1)
    if not messages.data or not messages.data[0].content:
        raise ValueError("Nenhuma resposta válida recebida")

    return messages.data[0].content[0].text.value
