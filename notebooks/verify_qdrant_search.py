from qdrant_client import QdrantClient
import qdrant_client
print(f"Version: {qdrant_client.__version__ if hasattr(qdrant_client, '__version__') else 'UNKNOWN'}")
print(f"File: {qdrant_client.__file__}")

try:
    c = QdrantClient("http://localhost:6333")
    print("Instance attributes:")
    print([d for d in dir(c) if 'search' in d or 'query' in d])
    print("Type:", type(c))
    print("MRO:", type(c).mro())
except Exception as e:
    print(e)
