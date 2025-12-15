import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

# Initialize Llama3
print("Initializing LLM...")
try:
    llm = ChatOllama(
        base_url=os.environ.get('OLLAMA_BASE_URL'),
        model='llama3'
    )
    print("LLM Initialized")
except Exception as e:
    print(f"Error initializing LLM: {e}")

# 1. Simple Generation Test
print("Testing Generation...")
try:
    response = llm.invoke('Tell me a one-sentence joke about Docker.')
    print(f"Generation Response: {response.content}")
except Exception as e:
    print(f"Error in Generation: {e}")

# Initialize Embeddings
print("Initializing Embeddings...")
try:
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    print('Embeddings model loaded')
except Exception as e:
    print(f"Error loading embeddings: {e}")

# Create some dummy data
text_data = [
    'SnapLogic is a powerful integration platform.',
    'Docker helps you containerize applications.',
    'RAG stands for Retrieval-Augmented Generation.',
    'Llama3 is a large language model by Meta.'
]
docs = [Document(page_content=t) for t in text_data]

# Create a vector store
print("Creating Vector Store...")
try:
    url = os.environ.get('QDRANT_URL')
    qdrant = QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url=url,
        prefer_grpc=False,
        collection_name='hello_rag'
    )
    print('Documents indexed in Qdrant!')
except Exception as e:
    print(f"Error creating/indexing Qdrant: {e}")

# Perform a Search
print("Performing Search...")
try:
    query = 'What is RAG?'
    found_docs = qdrant.similarity_search(query, k=1)
    print(f'Top result: {found_docs[0].page_content}')
except Exception as e:
    print(f"Error in search: {e}")

# Connect it to the LLM
print("Running RAG Chain...")
try:
    retriever = qdrant.as_retriever()
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(chain.invoke('What does RAG stand for?'))
except Exception as e:
    print(f"Error in RAG chain: {e}")
