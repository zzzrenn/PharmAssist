import sys
import os
# Add the project root to path to resolve module imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import logging

from bson import json_util
from config import settings
from core.db.mongo import MongoDatabaseConnector
from core.logger_utils import get_logger
from core.mq import publish_to_rabbitmq

logger = get_logger(__file__)


def stream_process():
    try:
        client = MongoDatabaseConnector()
        db = client["pharmassist"]
        logging.info("Connected to MongoDB.")

        # Watch changes in a specific collection
        changes = db.watch([{"$match": {"operationType": {"$in": ["insert", "update"]}}}])
        for change in changes:
            data_type = change["ns"]["coll"]
            
            # Handle different operation types
            if change["operationType"] == "insert":
                entry_id = str(change["fullDocument"]["_id"])
                document = change["fullDocument"]
            elif change["operationType"] == "update":
                entry_id = str(change["documentKey"]["_id"])
                # For updates, we need to fetch the full document
                document = db[data_type].find_one({"_id": change["documentKey"]["_id"]})
                if not document:
                    logger.warning(f"Document with id {entry_id} not found after update")
                    continue
            else:
                logger.warning(f"Unsupported operation type: {change['operationType']}")
                continue

            document.pop("_id")
            document["type"] = data_type
            document["entry_id"] = entry_id

            if data_type not in ["NICE_GUIDELINE", "test_collection"]:
                logging.info(f"Unsupported data type: '{data_type}'")
                continue

            # Use json_util to serialize the document
            data = json.dumps(document, default=json_util.default)
            logger.info(
                f"Change detected and serialized for a data sample of type {data_type}."
            )

            # Send data to rabbitmq
            publish_to_rabbitmq(queue_name=settings.RABBITMQ_QUEUE_NAME, data=data)
            logger.info(f"Data of type '{data_type}' published to RabbitMQ.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    stream_process()