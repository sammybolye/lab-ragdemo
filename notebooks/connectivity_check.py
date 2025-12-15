import os
import requests
import sys

def check_ollama():
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    print(f"Checking Ollama at: {ollama_url}")
    try:
        # Ollama API usually has a /api/tags or just / to check version
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print("[\u2705] Ollama is CONNECTED!")
            print(f"Models available: {[m['name'] for m in response.json()['models'][:3]]} ...")
        else:
            print(f"[\u274c] Ollama returned status {response.status_code}")
    except Exception as e:
        print(f"[\u274c] Failed to connect to Ollama: {e}")
        print("Tip: Ensure Ollama is running on your host machine and capable of accepting connections.")

def check_qdrant():
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    print(f"\nChecking Qdrant at: {qdrant_url}")
    try:
        response = requests.get(f"{qdrant_url}/collections", timeout=5)
        if response.status_code == 200:
            print("[\u2705] Qdrant is CONNECTED!")
            print(f"Collections: {response.json()['result']['collections']}")
        else:
            print(f"[\u274c] Qdrant returned status {response.status_code}")
    except Exception as e:
        print(f"[\u274c] Failed to connect to Qdrant: {e}")

if __name__ == "__main__":
    print("--- RAG Lab Connectivity Check ---\n")
    check_ollama()
    check_qdrant()
    print("\n----------------------------------")
