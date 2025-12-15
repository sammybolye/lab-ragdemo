import nbformat as nbf
import os

def create_hello_rag():
    nb = nbf.v4.new_notebook()
    
    nb.cells.append(nbf.v4.new_markdown_cell("# 01. Hello RAG (Simple In-Memory)\n"
                                             "Welcome to your first RAG Lab! In this notebook, we will:\n"
                                             "1. Connect to Llama3 using `langchain-ollama`.\n"
                                             "2. Perform a simple generation.\n"
                                             "3. Build a RAG system using `langchain-qdrant`."))

    nb.cells.append(nbf.v4.new_code_cell("import os\n"
                                         "from langchain_ollama import ChatOllama\n"
                                         "from langchain_core.prompts import ChatPromptTemplate\n"
                                         "from langchain_core.output_parsers import StrOutputParser"))

    nb.cells.append(nbf.v4.new_code_cell("# Initialize Llama3\n"
                                         "llm = ChatOllama(\n"
                                         "    base_url=os.environ.get('OLLAMA_BASE_URL'),\n"
                                         "    model='llama3'\n"
                                         ")\n"
                                         "print('LLM Initialized')"))

    nb.cells.append(nbf.v4.new_code_cell("# 1. Simple Generation Test\n"
                                         "response = llm.invoke('Tell me a one-sentence joke about Docker.')\n"
                                         "print(response.content)"))

    nb.cells.append(nbf.v4.new_markdown_cell("## Simple RAG using Qdrant\n"
                                             "We will use `QdrantVectorStore` connected to our Qdrant container."))
    
    nb.cells.append(nbf.v4.new_code_cell("from langchain_huggingface import HuggingFaceEmbeddings\n"
                                         "from langchain_qdrant import QdrantVectorStore\n"
                                         "from langchain_text_splitters import CharacterTextSplitter\n"
                                         "from langchain_core.documents import Document\n\n"
                                         "# Initialize Embeddings (running locally on CPU)\n"
                                         "embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')\n"
                                         "print('Embeddings model loaded')"))

    nb.cells.append(nbf.v4.new_code_cell("# Create some dummy data\n"
                                         "text_data = [\n"
                                         "    'SnapLogic is a powerful integration platform.',\n"
                                         "    'Docker helps you containerize applications.',\n"
                                         "    'RAG stands for Retrieval-Augmented Generation.',\n"
                                         "    'Llama3 is a large language model by Meta.'\n"
                                         "]\n"
                                         "docs = [Document(page_content=t) for t in text_data]"))

    nb.cells.append(nbf.v4.new_code_cell("# Create a vector store\n"
                                         "url = os.environ.get('QDRANT_URL')\n"
                                         "qdrant = QdrantVectorStore.from_documents(\n"
                                         "    docs,\n"
                                         "    embeddings,\n"
                                         "    url=url,\n"
                                         "    prefer_grpc=False,\n"
                                         "    collection_name='hello_rag'\n"
                                         ")\n"
                                         "print('Documents indexed in Qdrant!')"))

    nb.cells.append(nbf.v4.new_code_cell("# Perform a Search\n"
                                         "query = 'What is RAG?'\n"
                                         "found_docs = qdrant.similarity_search(query, k=1)\n"
                                         "print(f'Top result: {found_docs[0].page_content}')"))

    nb.cells.append(nbf.v4.new_code_cell("# Connect it to the LLM (The RAG part)\n"
                                         "from langchain_core.runnables import RunnablePassthrough\n\n"
                                         "retriever = qdrant.as_retriever()\n"
                                         "template = \"\"\"Answer the question based only on the following context:\n"
                                         "{context}\n\n"
                                         "Question: {question}\n"
                                         "\"\"\"\n"
                                         "prompt = ChatPromptTemplate.from_template(template)\n\n"
                                         "chain = (\n"
                                         "    {'context': retriever, 'question': RunnablePassthrough()}\n"
                                         "    | prompt\n"
                                         "    | llm\n"
                                         "    | StrOutputParser()\n"
                                         ")\n\n"
                                         "print(chain.invoke('What does RAG stand for?'))"))

    with open('notebooks/01_Hello_RAG.ipynb', 'w') as f:
        nbf.write(nb, f)


def create_vector_db_rag():
    nb = nbf.v4.new_notebook()
    
    nb.cells.append(nbf.v4.new_markdown_cell("# 02. Advanced RAG with PDF\n"
                                             "In this notebook, we will:\n"
                                             "1. Download a sample PDF.\n"
                                             "2. Ingest and split the text.\n"
                                             "3. Store embeddings in Qdrant (Persistent).\n"
                                             "4. Perform RAG questions against the PDF."))

    nb.cells.append(nbf.v4.new_code_cell("import os\n"
                                         "import requests\n"
                                         "from langchain_community.document_loaders import PyPDFLoader\n"
                                         "from langchain_text_splitters import RecursiveCharacterTextSplitter\n"
                                         "from langchain_huggingface import HuggingFaceEmbeddings\n"
                                         "from langchain_qdrant import QdrantVectorStore\n"
                                         "from langchain_ollama import ChatOllama\n"
                                         "from langchain_core.prompts import ChatPromptTemplate\n"
                                         "from langchain_core.runnables import RunnablePassthrough\n"
                                         "from langchain_core.output_parsers import StrOutputParser\n"
                                         "\n"
                                         "# Ensure data directory exists\n"
                                         "os.makedirs('data', exist_ok=True)"))

    nb.cells.append(nbf.v4.new_code_cell("# 1. Download a Sample PDF\n"
                                         "pdf_url = 'https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/how_to/state_of_the_union.pdf' \n"
                                         "pdf_path = 'data/sample.pdf'\n\n"
                                         "if not os.path.exists(pdf_path):\n"
                                         "    print('Downloading PDF...')\n"
                                         "    response = requests.get(pdf_url)\n"
                                         "    with open(pdf_path, 'wb') as f:\n"
                                         "        f.write(response.content)\n"
                                         "    print('PDF Downloaded')\n"
                                         "else:\n"
                                         "    print('PDF already exists')"))

    nb.cells.append(nbf.v4.new_code_cell("# 2. Load and Split PDF\n"
                                         "loader = PyPDFLoader(pdf_path)\n"
                                         "pages = loader.load()\n"
                                         "print(f'Loaded {len(pages)} pages')\n\n"
                                         "text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)\n"
                                         "splits = text_splitter.split_documents(pages)\n"
                                         "print(f'Created {len(splits)} splits')"))

    nb.cells.append(nbf.v4.new_code_cell("# 3. Index in Qdrant (Persistent)\n"
                                         "# We use a specific collection name 'pdf_rag'\n"
                                         "embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')\n"
                                         "url = os.environ.get('QDRANT_URL')\n\n"
                                         "qdrant = QdrantVectorStore.from_documents(\n"
                                         "    splits,\n"
                                         "    embeddings,\n"
                                         "    url=url,\n"
                                         "    prefer_grpc=False,\n"
                                         "    collection_name='pdf_rag',\n"
                                         "    force_recreate=True  # Clean start for this tutorial\n"
                                         ")\n"
                                         "print('PDF Content Indexed!')"))

    nb.cells.append(nbf.v4.new_code_cell("# 4. Perform RAG\n"
                                         "llm = ChatOllama(\n"
                                         "    base_url=os.environ.get('OLLAMA_BASE_URL'),\n"
                                         "    model='llama3'\n"
                                         ")\n\n"
                                         "retriever = qdrant.as_retriever(search_kwargs={'k': 3})\n"
                                         "template = \"\"\"Answer the question based only on the following context:\n"
                                         "{context}\n\n"
                                         "Question: {question}\n"
                                         "\"\"\"\n"
                                         "prompt = ChatPromptTemplate.from_template(template)\n\n"
                                         "chain = (\n"
                                         "    {'context': retriever, 'question': RunnablePassthrough()}\n"
                                         "    | prompt\n"
                                         "    | llm\n"
                                         "    | StrOutputParser()\n"
                                         ")\n\n"
                                         "query = 'What did the president say about Ukraine?'\n"
                                         "print(f'Question: {query}')\n"
                                         "print('Answer:')\n"
                                         "for chunk in chain.stream(query):\n"
                                         "    print(chunk, end='', flush=True)"))
    
    with open('notebooks/02_Vector_DB_RAG.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_fine_tuning_lab():
    nb = nbf.v4.new_notebook()
    
    nb.cells.append(nbf.v4.new_markdown_cell("# 03. Fine-Tuning Primer (PEFT/LoRA)\n"
                                             "**Warning: CPU Training is Slow**\n\n"
                                             "In this notebook, we welcome you to the world of Fine-Tuning!\n"
                                             "We will use **TinyLlama-1.1B** and **LoRA** (Low-Rank Adaptation) to demonstrate the mechanics.\n\n"
                                             "**Steps:**\n"
                                             "1. Load Model & Tokenizer (TinyLlama).\n"
                                             "2. Prepare a tiny dataset.\n"
                                             "3. Configure LoRA.\n"
                                             "4. Run a Training Loop (1-2 steps only)."))

    nb.cells.append(nbf.v4.new_code_cell("import torch\n"
                                         "from transformers import AutoTokenizer, AutoModelForCausalLM\n"
                                         "from peft import LoraConfig, get_peft_model, TaskType\n"
                                         "from trl import SFTTrainer, SFTConfig\n"
                                         "from datasets import Dataset\n"
                                         "\n"
                                         "# Use CPU (or mps if available on Mac host, but inside Docker usually CPU)\n"
                                         "device = 'cpu'\n"
                                         "print(f'Using device: {device}')"))

    nb.cells.append(nbf.v4.new_code_cell("# 1. Load Model (TinyLlama)\n"
                                         "model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'\n"
                                         "print(f'Loading {model_id}...')\n"
                                         "tokenizer = AutoTokenizer.from_pretrained(model_id)\n"
                                         "model = AutoModelForCausalLM.from_pretrained(model_id)\n"
                                         "print('Model loaded')"))

    nb.cells.append(nbf.v4.new_code_cell("# 2. Prepare Dummy Dataset\n"
                                         "data = [\n"
                                         "    {'text': 'User: How do I learn RAG? Assistant: Start with the RAG Lab! '},\n"
                                         "    {'text': 'User: What is Docker? Assistant: A tool to containerize apps. '},\n"
                                         "    {'text': 'User: Who is Antigravity? Assistant: An agentic AI coding assistant. '}\n"
                                         "] * 5  # Repeat to make it slightly bigger\n"
                                         "\n"
                                         "dataset = Dataset.from_list(data)\n"
                                         "print(f'Dataset size: {len(dataset)}')"))

    nb.cells.append(nbf.v4.new_code_cell("# 3. Configure LoRA\n"
                                         "peft_config = LoraConfig(\n"
                                         "    task_type=TaskType.CAUSAL_LM, \n"
                                         "    inference_mode=False, \n"
                                         "    r=4,            # Rank\n"
                                         "    lora_alpha=16, \n"
                                         "    lora_dropout=0.1\n"
                                         ")\n\n"
                                         "model = get_peft_model(model, peft_config)\n"
                                         "model.print_trainable_parameters()"))

    nb.cells.append(nbf.v4.new_code_cell("# 4. Train (1 Step Demo)\n"
                                         "# Updated for TRL >= 0.25 (Using SFTConfig)\n"
                                         "sft_config = SFTConfig(\n"
                                         "    dataset_text_field='text',\n"
                                         "    output_dir='./results',\n"
                                         "    num_train_epochs=3,\n"
                                         "    per_device_train_batch_size=1,\n"
                                         "    max_steps=30,  # Increased to 30 steps for better learning (approx 1-2 mins on CPU)\n"
                                         "    logging_steps=5,\n"
                                         "    use_cpu=True\n"
                                         ")\n\n"
                                         "trainer = SFTTrainer(\n"
                                         "    model=model,\n"
                                         "    train_dataset=dataset,\n"
                                         "    args=sft_config,\n"
                                         ")\n\n"
                                         "print('Starting training (this might take a minute)...')\n"
                                         "trainer.train()\n"
                                         "print('Training complete!')"))

    nb.cells.append(nbf.v4.new_code_cell("# 5. Save the Component (Adapter)\n"
                                         "# We don't save the whole 1GB model, just the tiny difference (LoRA)\n"
                                         "output_adapter_dir = './tinyllama_lora_adapter'\n"
                                         "trainer.save_model(output_adapter_dir)\n"
                                         "print(f'Adapter saved to {output_adapter_dir}')"))

    nb.cells.append(nbf.v4.new_code_cell("# 6. Inference (Try it out!)\n"
                                         "from peft import PeftModel\n"
                                         "\n"
                                         "# Inference usually runs in a fresh process, but here we reload\n"
                                         "print('Reloading base model...')\n"
                                         "base_model = AutoModelForCausalLM.from_pretrained(model_id)\n"
                                         "\n"
                                         "print('Loading your new adapters...')\n"
                                         "finetuned_model = PeftModel.from_pretrained(base_model, output_adapter_dir)\n"
                                         "\n"
                                         "# Test prompt\n"
                                         "test_prompt = 'User: Who is Antigravity? Assistant: '\n"
                                         "inputs = tokenizer(test_prompt, return_tensors='pt')\n"
                                         "\n"
                                         "print('Generating...')\n"
                                         "outputs = finetuned_model.generate(**inputs, max_new_tokens=30)\n"
                                         "result = tokenizer.decode(outputs[0], skip_special_tokens=True)\n"
                                         "print('-'*20)\n"
                                         "print(result)\n"
                                         "print('-'*20)"))

    with open('notebooks/03_Fine_Tuning_Primer.ipynb', 'w') as f:
        nbf.write(nb, f)



if __name__ == "__main__":
    create_hello_rag()
    create_vector_db_rag()
    create_fine_tuning_lab()
    print("Notebooks created.")
