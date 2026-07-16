"""
run.py — Socratica launcher
===========================
Validates dependencies, displays setup guidance, then starts the FastAPI server.
"""
import os
import sys
import subprocess
import threading
import time
import webbrowser

REQUIRED_PACKAGES = ["fastapi", "uvicorn", "httpx", "aiosqlite", "slowapi", "PIL"]


def check_requirements():
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"\n[!] Missing packages: {', '.join(missing)}")
        print("    Run:  pip install -r requirements.txt\n")
        answer = input("Attempt auto-install now? [y/N]: ").strip().lower()
        if answer == "y":
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        else:
            print("Please install requirements and restart.")
            sys.exit(1)


def print_banner():
    print("=" * 66)
    print("    SOCRATICA — Multimodal AI STEM Diagram Feedback System v2.0   ")
    print("=" * 66)
    print("  VLM model  : llava:13b  (via Ollama)")
    print("  LLM model  : llama3.1:8b  (via Ollama)")
    print("  Database   : SQLite (socratica.db)")
    print("  Dashboard  : http://127.0.0.1:8000")
    print("  API docs   : http://127.0.0.1:8000/docs")
    print("=" * 66)
    print()
    print("  [!] Make sure Ollama is running: ollama serve")
    print("  [!] Pull models if needed:")
    print("        ollama pull llava:13b")
    print("        ollama pull llama3.1:8b")
    print()


def open_browser_after_delay():
    time.sleep(2.0)
    webbrowser.open("http://127.0.0.1:8000/")


def main():
    print_banner()
    check_requirements()

    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    threading.Thread(target=open_browser_after_delay, daemon=True).start()

    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=int(os.getenv("APP_PORT", "8000")),
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
