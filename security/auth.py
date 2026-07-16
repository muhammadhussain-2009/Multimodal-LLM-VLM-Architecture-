import os
import secrets
import time
import uuid
from typing import Dict, Optional

from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Load secret key or generate a fallback for local testing
SECRET_KEY = os.getenv("SESSION_SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_hex(32)

SESSION_MAX_AGE = 3600 * 8  # 8 hours

serializer = URLSafeTimedSerializer(SECRET_KEY)

def generate_session_token(device_fingerprint: Optional[str] = None) -> str:
    """Generates a signed session token."""
    payload = {
        "session_id": str(uuid.uuid4()),
        "created_at": time.time(),
        "device_hint": device_fingerprint[:8] if device_fingerprint else "unknown"
    }
    return serializer.dumps(payload)

def verify_session_token(token: str) -> Optional[Dict]:
    """Verifies the token signature and age."""
    try:
        payload = serializer.loads(token, max_age=SESSION_MAX_AGE)
        return payload
    except (BadSignature, SignatureExpired):
        return None

class SessionMiddleware(BaseHTTPMiddleware):
    """Stateless session management middleware."""
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        
        # Skip auth for healthcheck and static files
        if path == "/api/health" or path.startswith("/static"):
            return await call_next(request)

        token = request.headers.get("X-Session-Token") or request.cookies.get("session_token")
        
        payload = None
        if token:
            payload = verify_session_token(token)

        new_token_needed = payload is None

        if new_token_needed:
            # Generate a new session
            token = generate_session_token()
            payload = verify_session_token(token)
            
        request.state.session = payload

        response: Response = await call_next(request)

        if new_token_needed:
            response.set_cookie(
                key="session_token",
                value=token,
                httponly=True,
                samesite="strict",
                path="/",
                max_age=SESSION_MAX_AGE
            )

        return response
