from flask import Blueprint, request, jsonify, current_app
import tempfile
import os
import time
import traceback  # Importante para registrar stack trace
from werkzeug.utils import secure_filename
from services.pdf_reader import extract_text_from_pdf
from services.openai_client import analisar_contrato
from utils.cleanup import cleanup_temp_dir

analisar_bp = Blueprint('analisar', __name__)

@analisar_bp.route("/analisar", methods=["POST"])
def analisar():
    temp_dir = None
    try:
        current_app.logger.info("➡️ Início do endpoint /analisar")

        if 'file' not in request.files:
            current_app.logger.warning("⚠️ Nenhum arquivo enviado")
            raise ValueError("Nenhum arquivo enviado na requisição")

        file = request.files['file']
        if file.filename == '':
            current_app.logger.warning("⚠️ Arquivo enviado sem nome")
            raise ValueError("Nenhum arquivo selecionado")

        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext != '.pdf':
            current_app.logger.warning(f"❌ Tipo de arquivo não suportado: {ext}")
            raise ValueError("Somente arquivos PDF são suportados.")

        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        current_app.logger.info(f"📄 Arquivo salvo temporariamente em: {temp_path}")

        text = extract_text_from_pdf(temp_path)
        if not text:
            current_app.logger.warning("⚠️ Nenhum texto extraído do PDF")
            raise ValueError("Não foi possível extrair texto do PDF.")

        current_app.logger.info("💬 Texto extraído com sucesso. Enviando para OpenAI...")
        response = analisar_contrato(text)
        current_app.logger.info("✅ Análise concluída com sucesso.")

        return jsonify({"status": "sucesso", "analise": response})

    except Exception as e:
        trace = traceback.format_exc()
        current_app.logger.error(f"🔥 Erro inesperado: {str(e)}\n{trace}")
        return jsonify({
            "status": "erro",
            "mensagem": str(e),
            "dica": "Verifique se o arquivo é válido e tente novamente"
        }), 500
    finally:
        if temp_dir:
            cleanup_temp_dir(temp_dir)
            current_app.logger.info("🧹 Diretório temporário limpo.")
