import re
from typing import Dict, Tuple, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

try:
    import bleach
    HAS_BLEACH = True
except ImportError:
    HAS_BLEACH = False
    import html

MAX_UPLOAD_BYTES = 5 * 1024 * 1024

INJECTION_PATTERNS = [
    ("ignore_instructions", re.compile(r'ignore (all )?(previous|prior|above) instructions', re.IGNORECASE)),
    ("disregard_instructions", re.compile(r'disregard (your|the|all) (system|initial) (prompt|instructions)', re.IGNORECASE)),
    ("you_are_now", re.compile(r'you are now', re.IGNORECASE)),
    ("act_as", re.compile(r'act as', re.IGNORECASE)),
    ("pretend_to_be", re.compile(r'pretend (to be|you are)', re.IGNORECASE)),
    ("leak_prompt", re.compile(r'output (the|your) (system|initial) prompt', re.IGNORECASE)),
    ("repeat_instructions", re.compile(r'repeat (the|your) instructions', re.IGNORECASE)),
    ("dan_mode", re.compile(r'DAN mode', re.IGNORECASE)),
    ("jailbreak", re.compile(r'jailbreak', re.IGNORECASE)),
]

def sanitize_text(text: str) -> str:
    """Strips HTML tags, null bytes, control characters."""
    if not isinstance(text, str):
        return str(text)
    
    # Strip null bytes and control chars (except newline and tab)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    
    if HAS_BLEACH:
        text = bleach.clean(text, tags=[], attributes={}, protocols=[], strip=True)
    else:
        text = html.escape(text)
        
    return text

def sanitize_metadata(metadata: Dict) -> Dict:
    """Recursively sanitizes all string values in nested dicts."""
    sanitized = {}
    for k, v in metadata.items():
        clean_k = sanitize_text(str(k))
        if isinstance(v, dict):
            sanitized[clean_k] = sanitize_metadata(v)
        elif isinstance(v, list):
            sanitized[clean_k] = [sanitize_text(str(i)) if isinstance(i, str) else i for i in v]
        elif isinstance(v, str):
            sanitized[clean_k] = sanitize_text(v)
        else:
            sanitized[clean_k] = v
    return sanitized

def validate_student_id(student_id: str) -> str:
    """Validates student_id is alphanumeric + underscore + hyphen only, max 64 chars."""
    if not student_id:
        raise ValueError("Student ID cannot be empty.")
    if len(student_id) > 64:
        raise ValueError("Student ID exceeds maximum length of 64 characters.")
    if not re.match(r'^[a-zA-Z0-9_\-]+$', student_id):
        raise ValueError("Student ID contains invalid characters.")
    return student_id

def validate_upload_size(content_length: int) -> bool:
    """Returns True if content_length is within MAX_UPLOAD_BYTES."""
    if content_length is None:
        return True # Handled at reading phase if unknown
    return content_length <= MAX_UPLOAD_BYTES

def validate_file_magic_bytes(raw_bytes: bytes) -> Tuple[bool, str]:
    """Checks magic bytes for common image formats."""
    if not raw_bytes:
        return False, "File is empty."
    
    if raw_bytes.startswith(b'\xff\xd8\xff'):
        return True, "JPEG"
    elif raw_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
        return True, "PNG"
    elif raw_bytes.startswith(b'GIF8'):
        return True, "GIF"
    elif raw_bytes.startswith(b'RIFF') and raw_bytes[8:12] == b'WEBP':
        return True, "WEBP"
        
    return False, "Invalid or unsupported file type."

def detect_prompt_injection(text: str) -> Tuple[bool, Optional[str]]:
    """Detects adversarial prompt injection attempts."""
    for name, pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            return True, name
    return False, None

def sanitize_llm_input(text: str, max_length: int = 500) -> str:
    """Sanitizes input intended for the LLM to prevent injections."""
    text = sanitize_text(text)
    if len(text) > max_length:
        text = text[:max_length]
        
    is_injection, matched = detect_prompt_injection(text)
    if is_injection:
        raise ValueError(f"Adversarial input detected: {matched}")
        
    return text

class CSPMiddleware(BaseHTTPMiddleware):
    """Injects restrictive Content Security Policy and security headers."""
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
            "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response
