import os
from datetime import datetime
import uuid
from pymongo import MongoClient
from dotenv import load_dotenv

DATABASE_NAME = "vcon_database"  
COLLECTION_NAME = "vcon_data"  
load_dotenv()
Mongo_connect = os.getenv("MONGO_URI")
# Connect to MongoDB
client = MongoClient(Mongo_connect)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

def save_vcon(vcon):
    try:
        # Generate a unique ID for the vCon
        vcon["uuid"] = str(uuid.uuid4())
        # Add created_at timestamp
        vcon["created_at"] = datetime.utcnow().isoformat()

        # Insert the vCon into the collection
        result = collection.insert_one(vcon)
        print(f"Data inserted with ID: {result.inserted_id}")
        return result.inserted_id
    except Exception as e:
        print(f"Error saving vCon: {e}")
        return None


def close_connection():
    client.close()

