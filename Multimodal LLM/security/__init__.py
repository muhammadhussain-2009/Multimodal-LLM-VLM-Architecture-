"""
Security module for Socratica STEM Feedback System.
"""
from .protection import (
    sanitize_text,
    sanitize_metadata,
    validate_student_id,
    CSPMiddleware,
    validate_upload_size,
    validate_file_magic_bytes,
    detect_prompt_injection,
    sanitize_llm_input,
)
from .auth import (
    generate_session_token,
    verify_session_token,
    SessionMiddleware,
)
from .database_sec import (
    apply_secure_pragmas,
    set_file_permissions,
    RetryingDatabaseConnection,
    check_database_integrity,
    prepare_sqlcipher_hooks,
)

__all__ = [
    "sanitize_text",
    "sanitize_metadata",
    "validate_student_id",
    "CSPMiddleware",
    "validate_upload_size",
    "validate_file_magic_bytes",
    "detect_prompt_injection",
    "sanitize_llm_input",
    "generate_session_token",
    "verify_session_token",
    "SessionMiddleware",
    "apply_secure_pragmas",
    "set_file_permissions",
    "RetryingDatabaseConnection",
    "check_database_integrity",
    "prepare_sqlcipher_hooks",
]
