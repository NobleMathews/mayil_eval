from pymilvus import utility
from mayil.integrations.milvus import MilvusDB
from pymilvus import Collection

_ = MilvusDB()
conn = utility.list_collections()
for name in conn:
    collection = Collection(name)
    collection.release()