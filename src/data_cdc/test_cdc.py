from pymongo import MongoClient

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
        result = collection.insert_one(data)
        print(f"Data inserted with _id: {result.inserted_id}")
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
        result = collection.update_one(old_data, {"$set": data})
        print(f"Data updated with _id: {old_data['_id']}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    insert_data_to_mongodb(
        settings.MONGO_DATABASE_HOST,
        "pharmassist",
        "test_collection",
        {"url": "https://www.nice.org.uk/guidance/ng133", "title": "Test title", "last_updated": "2024-01-01"}
    )

    import random
    random_year = random.randint(1900, 2025)
    update_data_in_mongodb(
        settings.MONGO_DATABASE_HOST,
        "pharmassist",
        "test_collection",
        {"url": "https://www.nice.org.uk/guidance/ng133", "title": "Test title", "last_updated": f"{random_year}-01-01"}
    )
