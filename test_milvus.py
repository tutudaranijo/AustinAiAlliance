from pymilvus import connections

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

print("Milvus connected")
