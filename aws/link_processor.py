import os

import boto3

s3 = boto3.client("s3")
sqs = boto3.client("sqs")


def lambda_handler(event, context):
    bucket = os.environ["BUCKET_NAME"]
    key = os.environ["FILE_KEY"]
    queue_url = os.environ["SQS_QUEUE_URL"]

    # Read file from S3
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8")
    links = [line.strip() for line in content.split("\n") if line.strip()]

    # Send to SQS
    for link in links:
        sqs.send_message(QueueUrl=queue_url, MessageBody=link)

    return f"Successfully sent {len(links)} links to SQS"
