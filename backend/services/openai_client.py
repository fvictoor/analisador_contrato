import os
import time
from openai import OpenAI
from config import Config

client = OpenAI(api_key=Config.OPENAI_API_KEY)

def analisar_contrato(texto):
    print("[INÍCIO] Iniciando análise de contrato.")
    
    assistant_id = Config.ASSISTANT_ID
    print("[ASSISTANT] Usando Assistant ID:", assistant_id)

    print("[THREAD] Criando nova thread...")
    thread = client.beta.threads.create()
    print("[THREAD] Thread criada com ID:", thread.id)

    print("[MENSAGEM] Enviando contrato para análise...")
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"Analise este contrato e extraia informações relevantes em formato JSON: {texto}"
    )
    print("[MENSAGEM] Mensagem enviada.")

    print("[RUN] Iniciando run para análise com o Assistant...")
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id
    )
    print("[RUN] Run iniciado com ID:", run.id)

    timeout = 300
    start_time = time.time()
    print("[AGUARDANDO] Aguardando conclusão do run (timeout de 5 minutos)...")
    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError("Tempo de análise excedido")
        
        status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        print(f"[STATUS] Status atual do run: {status.status}")

        if status.status == "completed":
            print("[CONCLUÍDO] Análise concluída.")
            break
        elif status.status == "failed":
            raise RuntimeError(f"[ERRO] Falha na análise: {status.last_error.message if status.last_error else 'Erro desconhecido'}")
        elif status.status in ["cancelled", "expired"]:
            raise RuntimeError(f"[ERRO] Processo {status.status}")
        
        time.sleep(5)

    print("[RESPOSTA] Buscando mensagem de resposta...")
    messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc", limit=1)
    if not messages.data or not messages.data[0].content:
        raise ValueError("[ERRO] Nenhuma resposta válida recebida")

    print("[SUCESSO] Análise finalizada com sucesso.")
    return messages.data[0].content[0].text.value
