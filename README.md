# PharmAssist Microservices Architecture

PharmAssist is a RAG-based chatbot for pharmacist guidelines. It has a distributed system built using microservices architecture.

## Available Microservices

### 1. Data Crawlers Service
- **Port**: 9010
- **Description**: Responsible for crawling and collecting data from various pharmaceutical sources.
- **Features**:
  - Web scraping capabilities
  - Data collection and normalization

### 2. Message Queue (RabbitMQ)
- **Ports**:
  - AMQP: 5673
  - Management UI: 15673
- **Description**: Central message broker for communication between changes in MongoDB and QdrantDB.
- **Features**:
  - Message queuing and routing
  - Web-based management interface
  - Persistent message storage

### 3. Data CDC (Change Data Capture) Service
- **Description**: Handles data synchronization and change detection across the MongoDB.
- **Features**:
  - Real-time data change detection (currently supports insert and update operations)
  - Event-driven architecture
  - Integration with message queue
- **Dependencies**: RabbitMQ

### 4. Feature Pipeline Service
- **Description**: Processes and transforms raw data, and save cleaned data and vector embeddings into QdrantDB.
- **Features**:
  - Data cleaning, chunking, and embedding
  - Read messages from RabbitMQ
- **Dependencies**: RabbitMQ

## Integration with AWS
### Data crawler
- Save the data/links.txt in a S3 bucket
- Create a SQS
- Create a lambda that reads the links.txt from this bucket and send message to SQS. Copy aws/link_processor.py to the created lambda. Add the required environment variables.
- Push the pharmassist-data-crawler image to ECR. Create a lambda using this image. Add the environment variable MONGO_DATABASE_HOST. Set the memory to 512 MB and timeout to 5 minutes. Add the SQS as trigger.
- Finally, create a EventBridge Scheduler to create a cron job that triggers the lambda to process the links in the text file.

## Reference

https://github.com/decodingml/llm-twin-course/tree/main
