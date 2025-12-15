# RAG Lab Demo: ETL Design Validator

This project demonstrates a **RAG (Retrieval-Augmented Generation)** system built to automate the validation of ETL (Extract, Transform, Load) design specifications. It ingests raw design documents (Markdown, Text), indexes them using a Vector Database (Qdrant), and uses an LLM (Llama3 via Ollama) to generate structured, machine-actionable design validation outputs (JSON) and human-readable reports.

## Features
- **Ingestion Pipeline**: Loads heterogeneous documents (.md, .txt) and chunks them for semantic search.
- **RAG Generation**: Retrieves relevant context to populate a strict Pydantic schema for ETL designs.
- **Dual-Output**: Produces both `design.json` (for automated testing) and `design_report.md` (for business review).
- **Containerized**: Fully Dockerized environment with JupyterLab, Qdrant, and all Python dependencies.

## Prerequisites & Setup (Detailed)

This project requires **Docker** and **Ollama** running locally.

### 1. Install Docker
*   **Mac/Windows**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
*   **Linux**: Install Docker Engine and Docker Compose.
*   **Verification**: Run `docker --version` in your terminal to ensure it is installed.

### 2. Install & Configure Ollama
The project uses Ollama to run the LLM (Llama3) and Embeddings (Nomic) locally.

1.  **Download**: Visit [ollama.com](https://ollama.com) and download the installer for your OS.
2.  **Start Service**:
    *   **Mac/Win**: Open the Ollama application. It runs in the background.
    *   **Linux**: Run `ollama serve`.
3.  **Verify Status**: Open your browser to `http://127.0.0.1:11434`. You should see strictly: `Ollama is running`.
4.  **Pull Required Models**:
    Open your terminal and run these commands **exactly**:
    ```bash
    ollama pull llama3
    ollama pull nomic-embed-text
    ```
    *(Note: The `nomic-embed-text` model is critical for vectorizing the documents.)*

## Running the Lab

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd rag_lab_demo
    ```

2.  **Start the environment**:
    ```bash
    docker-compose up --build
    ```
    *   This spins up two containers: `rag-lab-jupyter` (The Lab) and `rag-lab-qdrant` (Vector DB).
    *   **Wait** until you see logs indicating Jupyter Server has started.

3.  **Access Jupyter Lab**:
    *   Open your browser to: **[`http://127.0.0.1:8889/lab?token=raglab`](http://127.0.0.1:8889/lab?token=raglab)**
    *   The Lab is running on host port **8889**.
    *   **Token**: `raglab`

## Usage Guide

Navigate to the `notebooks/` folder in Jupyter Lab:

1.  **Step 1: Ingestion (`04_ETL_Spec_Ingestion.ipynb`)**
    *   Open and Run All Cells.
    *   **What it does**: Scanning `design_docs/` for any `.md` or `.txt` files, chunking them, and indexing them into the Qdrant Vector Database.
    *   **Success**: Look for "Ingestion Complete!" in the output.

2.  **Step 2: Generation (`05_ETL_Design_Generator.ipynb`)**
    *   Open and Run All Cells.
    *   **What it does**: Queries the Index for a specific pipeline (e.g., "Customer 360"), retrieves the context, and uses Llama3 to extract a structured design.
    *   **Results**: Check the `results/` folder for `design.json` and `design_report.md`.

## Project Structure
- `src/`: Python source code (e.g., `etl_design_schema.py` Pydantic models).
- `design_docs/`: Input folder for raw ETL specifications.
- `notebooks/`: Interactive notebooks for ingestion and generation.
- `results/`: Output folder for generated artifacts.
- `rag_lab_demo-jupyter`: The custom Docker image definition.
