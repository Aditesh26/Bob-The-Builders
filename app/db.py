import os

from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()


_DEFAULT_MONGODB_URI = (
    "mongodb+srv://aditeshpatro_db_user:admin123@cluster0.gxftxhm.mongodb.net/"
)


def get_mongo_client() -> MongoClient:
    mongodb_uri = os.getenv("MONGODB_URI", "").strip() or _DEFAULT_MONGODB_URI
    return MongoClient(mongodb_uri)


def get_collection():
    client = get_mongo_client()
    db_name = os.getenv("MONGODB_DB", "smart_drain")
    collection_name = os.getenv("MONGODB_COLLECTION", "readings")
    return client[db_name][collection_name]
