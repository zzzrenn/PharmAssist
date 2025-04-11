from pymongo import MongoClient
import uuid
import random
import string

from config import settings


def insert_data_to_mongodb(uri, database_name, collection_name, data):
    """
    Insert data into a MongoDB collection.

    :param uri: MongoDB URI
    :param database_name: Name of the database
    :param collection_name: Name of the collection
    :param data: Data to be inserted (dict)
    """
    client = MongoClient(uri)
    db = client[database_name]
    collection = db[collection_name]

    try:
        # Generate a UUID for the document
        doc_id = str(uuid.uuid4())
        data["_id"] = doc_id
        result = collection.insert_one(data)
        print(f"Data inserted with _id: {doc_id}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client.close()

def update_data_in_mongodb(uri, database_name, collection_name, data):
    """
    Update data in a MongoDB collection.

    :param uri: MongoDB URI
    :param database_name: Name of the database
    :param collection_name: Name of the collection
    :param data: Data to be updated (dict)
    """
    client = MongoClient(uri)
    db = client[database_name]
    collection = db[collection_name]

    try:
        old_data = collection.find_one({"url": data["url"]})
        if old_data:
            data["_id"] = old_data["_id"]  # Preserve the existing UUID
            result = collection.update_one({"_id": old_data["_id"]}, {"$set": data})
            print(f"Data updated with _id: {old_data['_id']}")
        else:
            print("No document found to update")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client.close()

def generate_random_text(num_words=100):
    """Generate random text with specified number of words."""
    words = []
    for _ in range(num_words):
        # Generate random word of length 3-10 characters
        word_length = random.randint(3, 10)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
        words.append(word)
    return ' '.join(words)


if __name__ == "__main__":
    insert_data_to_mongodb(
        settings.MONGO_DATABASE_HOST,
        "pharmassist",
        "test_collection",
        {"url": "https://www.nice.org.uk/guidance/ng133", "title": "Test title", "last_updated": "2024-01-01", "chapters": [{"markdown": "Test chapter"}]}
    )

    random_year = random.randint(1900, 2025)
    random_text = generate_random_text(100)
    update_data_in_mongodb(
        settings.MONGO_DATABASE_HOST,
        "pharmassist",
        "test_collection",
        {"url": "https://www.nice.org.uk/guidance/ng133", "title": "Test title", "last_updated": f"{random_year}-01-01", "chapters": [{"markdown": random_text}]}
    )
