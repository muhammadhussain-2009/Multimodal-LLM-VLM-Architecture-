import os
import sys
import subprocess
import webbrowser
import time

def check_requirements():
    """Validates if critical server libraries are installed."""
    required = ["fastapi", "uvicorn", "httpx"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
            
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Please install them using the following command:")
        print(f"pip install {' '.join(missing)}")
        print("\nAttempting automatic installation...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing, check=True)
            print("Successfully installed missing requirements.\n")
        except subprocess.CalledProcessError:
            print("Auto-installation failed. Please run pip install manually and restart.")
            sys.exit(1)

def main():
    print("==================================================================")
    print("      SOCRATICA: MULTIMODAL AI STEM DIAGRAM FEEDBACK SYSTEM       ")
    print("==================================================================")
    
    check_requirements()
    
    # Run the pre-processing data pipeline
    print("Running initial dataset preprocessing checks...")
    try:
        from backend.data_pipeline import DatasetPipeline
        dp = DatasetPipeline()
        dp.preprocess_all()
    except Exception as e:
        print(f"Warning: Dataset preprocess helper failed (ignoring for server launch): {str(e)}")

    print("\nStarting local server on http://127.0.0.1:8000...")
    
    # Start FastAPI server in a background subprocess or direct run
    # Opening browser after a short delay
    def open_browser():
        time.sleep(1.5)
        print("Launching default browser...")
        webbrowser.open("http://127.0.0.1:8000/")

    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.start()
    
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
