from pymongo import MongoClient

client = MongoClient("mongodb+srv://aditeshpatro_db_user:admin123@cluster0.gxftxhm.mongodb.net/")
db = client["smart_drain"]
readings_collection = db["readings"]


# ---- TEST CONNECTION ----
if __name__ == "__main__":
    readings_collection.insert_one({"test": "connected"})
    print("MongoDB Atlas connected successfully!")
