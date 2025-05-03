
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'sua_chave_secreta_super_segura_123')
    JWT_ACCESS_TOKEN_EXPIRES = 3600
    SUPPORTED_EXTENSIONS = {'.pdf'}
    MAX_FILE_SIZE = 20 * 1024 * 1024
    ADMIN_API_TOKEN = os.getenv('ADMIN_API_TOKEN', 'admin_token_seguro_123')
    USER_API_TOKEN = os.getenv('USER_API_TOKEN', 'user_token_seguro_456')
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ASSISTANT_ID = "asst_dDoRbTFUsPL65DOMgnxWX6xo"
