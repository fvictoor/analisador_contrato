from flask import Flask, request, jsonify
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token,
    get_jwt_identity
)
from openai import OpenAI
import time
import os
import tempfile
from werkzeug.utils import secure_filename
from functools import wraps
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

app = Flask(__name__)

# Configurações JWT
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'sua_chave_secreta_super_segura_123')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 3600  # 1 hora de validade
jwt = JWTManager(app)

# Configurações
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.pptx', '.xlsx'}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
API_ACCESS_TOKENS = {
    "admin": os.getenv('ADMIN_API_TOKEN', 'admin_token_seguro_123'),
    "user": os.getenv('USER_API_TOKEN', 'user_token_seguro_456')
}

# Inicializa cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Decorator personalizado para verificar token de API
def api_token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1] if 'Bearer' in request.headers['Authorization'] else None
        
        if not token:
            return jsonify({'message': 'Token está faltando!'}), 401
            
        if token not in API_ACCESS_TOKENS.values():
            return jsonify({'message': 'Token inválido!'}), 401
            
        return f(*args, **kwargs)
        
    return decorated

# Rota para gerar token JWT
@app.route('/login', methods=['POST'])
@api_token_required
def login():
    identity = 'admin' if request.headers['Authorization'].split(" ")[1] == API_ACCESS_TOKENS["admin"] else 'user'
    access_token = create_access_token(identity=identity)
    return jsonify({'token': access_token}), 200

@app.route("/analisar", methods=["POST"])
@jwt_required()
def analisar():
    current_user = get_jwt_identity()
    
    # Inicializa last_error no início da função
    last_error = None
    temp_dir = None
    uploaded_file = None
    
    try:
        # Verificação inicial do arquivo
        if 'file' not in request.files:
            raise ValueError("Nenhum arquivo enviado na requisição")

        file = request.files['file']
        if file.filename == '':
            raise ValueError("Nenhum arquivo selecionado")

        # Validação do arquivo
        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Extensão {ext} não suportada. Use: {', '.join(SUPPORTED_EXTENSIONS)}")

        # Cria diretório temporário
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)

        # Upload do arquivo para OpenAI
        with open(temp_path, "rb") as f:
            uploaded_file = client.files.create(
                file=f,
                purpose="assistants"
            )

        # Criação do assistente
        assistant = client.beta.assistants.create(
            name="Analisador de Contratos",
            instructions="Analise contratos e extraia informações importantes como partes, valores, datas e cláusulas.",
            tools=[{"type": "file_search"}],
            model="gpt-4o"
        )

        # Criação da thread e mensagem
        thread = client.beta.threads.create()
        
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="Analise este contrato e extraia informações relevantes em formato JSON.",
            attachments=[{
                "file_id": uploaded_file.id,
                "tools": [{"type": "file_search"}]
            }]
        )

        # Execução do assistente
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )

        # Monitoramento da execução
        timeout = 300  # 5 minutos
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("Tempo de análise excedido")
                
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            
            if run_status.status == "completed":
                break
            elif run_status.status == "failed":
                error_details = getattr(run_status, "last_error", None)
                error_msg = error_details.message if error_details else "Erro desconhecido"
                raise RuntimeError(f"Falha na análise: {error_msg}")
            elif run_status.status in ["cancelled", "expired"]:
                raise RuntimeError(f"Processo {run_status.status}")
                
            time.sleep(5)

        # Obtenção da resposta
        messages = client.beta.threads.messages.list(
            thread_id=thread.id,
            order="desc",
            limit=1
        )
        
        if not messages.data or not messages.data[0].content:
            raise ValueError("Nenhuma resposta válida recebida")
            
        response = messages.data[0].content[0].text.value
        
        return jsonify({
            "status": "sucesso",
            "analise": response,
            "user": current_user
        })

    except Exception as e:
        last_error = str(e)
        app.logger.error(f"Erro na análise: {last_error}")
        return jsonify({
            "status": "erro",
            "mensagem": last_error,
            "dica": "Verifique se o arquivo é válido e tente novamente"
        }), 500

    finally:
        # Limpeza dos recursos
        if temp_dir and os.path.exists(temp_dir):
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            os.rmdir(temp_dir)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)