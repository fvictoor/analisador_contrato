
from flask import Flask
from flask_jwt_extended import JWTManager
from config import Config
from routes.analisar import analisar_bp

app = Flask(__name__)
app.config.from_object(Config)

jwt = JWTManager(app)

# Registra as rotas
app.register_blueprint(analisar_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
