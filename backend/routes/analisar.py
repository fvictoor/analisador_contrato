
from flask import Blueprint, request, jsonify, current_app
import tempfile
import os
import time
from werkzeug.utils import secure_filename
from services.pdf_reader import extract_text_from_pdf
from services.openai_client import analisar_contrato
from utils.cleanup import cleanup_temp_dir

analisar_bp = Blueprint('analisar', __name__)

@analisar_bp.route("/analisar", methods=["POST"])
def analisar():
    temp_dir = None
    try:
        current_app.logger.info("Recebendo requisição...")

        if 'file' not in request.files:
            raise ValueError("Nenhum arquivo enviado na requisição")

        file = request.files['file']
        if file.filename == '':
            raise ValueError("Nenhum arquivo selecionado")

        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext != '.pdf':
            raise ValueError("Somente arquivos PDF são suportados.")

        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)

        text = extract_text_from_pdf(temp_path)
        if not text:
            raise ValueError("Não foi possível extrair texto do PDF.")

        response = analisar_contrato(text)
        return jsonify({"status": "sucesso", "analise": response})

    except Exception as e:
        current_app.logger.error(f"Erro: {str(e)}")
        return jsonify({
            "status": "erro",
            "mensagem": str(e),
            "dica": "Verifique se o arquivo é válido e tente novamente"
        }), 500
    finally:
        if temp_dir:
            cleanup_temp_dir(temp_dir)
