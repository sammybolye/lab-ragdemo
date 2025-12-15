import json
import os

file_path = 'notebooks/02_Vector_DB_RAG.ipynb'

with open(file_path, 'r') as f:
    nb = json.load(f)

# The cell causing the error is likely at index 4 (0:Title, 1:Imports, 2:CleanupMD, 3:CleanupCode, 4:BaselineMD, 5:BaselineCode)
# Let's target the cell with id="baseline_code" to be safe.

target_id = "baseline_code"
found = False

for cell in nb['cells']:
    if cell.get('id') == target_id:
        found = True
        # Replace the source
        cell['source'] = [
            "# Baseline Query\n",
            "from langchain_ollama import ChatOllama\n",
            "\n",
            "# Initialize LLM for this cell\n",
            "llm = ChatOllama(\n",
            "    base_url=os.environ.get('OLLAMA_BASE_URL'),\n",
            "    model='llama3'\n",
            ")\n",
            "\n",
            f"query = 'what is the best partitioning strategy for a lookup stage in datastage?'\n",
            "print(f'Question: {query}')\n",
            "print(llm.invoke(query).content)"
        ]
        break

if found:
    with open(file_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("Notebook 02 fixed successfully.")
else:
    print("Could not find baseline_code cell.")
