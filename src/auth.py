from typing import Tuple, Dict, Any
from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth


def _has_secrets_file() -> bool:
    # Verifica locais comuns do secrets.toml para evitar acessar st.secrets quando não existe
    candidates = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
    ]
    return any(p.exists() for p in candidates)


def _load_credentials_and_cookie() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    creds = None
    cookie = None
    if _has_secrets_file():
        try:
            _secrets = st.secrets
            creds = _secrets.get("credentials", None)
            cookie = _secrets.get("cookie", None)
        except Exception:
            creds = None
            cookie = None

    if not creds:
        # Fallback seguro para desenvolvimento local (não usar em produção)
        # Gera hashes de senha para compatibilidade com diferentes versões.
        try:
            hashed = stauth.Hasher(["demo123", "admin123"]).generate()
        except Exception:
            # Se Hasher falhar, usa texto plano (algumas versões suportam)
            hashed = ["demo123", "admin123"]

        creds = {
            "usernames": {
                "demo": {
                    "name": "Demo",
                    "password": hashed[0],
                    "roles": ["viewer"],
                },
                "admin": {
                    "name": "Administrador",
                    "password": hashed[1],
                    "roles": ["admin"],
                },
            }
        }
    else:
        # Não forçar hash manual: o Authenticate auto-hash de senhas em texto.
        # Mantemos quaisquer senhas já hashadas como estão.
        pass

    if not cookie:
        cookie = {
            "cookie_name": "analisador_contrato_auth",
            "signature_key": "change_this_signature_key",
            "cookie_expiry_days": 7,
        }

    return creds, cookie


def init_authenticator() -> stauth.Authenticate:
    creds, cookie = _load_credentials_and_cookie()
    authenticator = stauth.Authenticate(
        creds,
        cookie.get("cookie_name", "analisador_contrato_auth"),
        cookie.get("signature_key", "change_this_signature_key"),
        int(cookie.get("cookie_expiry_days", 7)),
    )
    return authenticator