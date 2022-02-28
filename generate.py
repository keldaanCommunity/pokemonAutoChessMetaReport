import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

mongoURI = os.getenv('MONGO_URI')
client = MongoClient(mongoURI)
db = client.dev
collection = db.detailledstatisticv2
print(collection.find_one())
