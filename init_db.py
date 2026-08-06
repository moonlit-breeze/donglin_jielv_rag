"""初始化向量数据库。"""

from rag.loader import load_and_split
from rag.vector_store import create_vectorstore

if __name__ == "__main__":
    docs = load_and_split("data/jielv.txt")
    create_vectorstore(docs)
    print("向量库创建完成")