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

### 5. Inference Pipeline
- **Description**: Handles user queries and provides responses using the RAG model.
- **Features**:
  - Query processing
  - Retrieval of relevant documents from QdrantDB
  - Response generation using LLM (Huggingface QWEN model)
  - Queries, retrieved documents, and responses will be randomly saved to a monitoring dataset for evaluation.

### 6. Evaluation
- **Description**: Evaluates the performance of the RAG system and LLM using Opik.
- **Features**:
  - Evaluate LLM using test dataset, metrics: Hallucination, LevenshteinRatio, Moderation.
  - Evaluate RAG system using test dataset, metrics: ContextPrecision, ContextRecall, Hallucination.
  - Evaluate chatbot using monitoring dataset, metrics: AnswerRelevance, Hallucination, Moderation.

## Installation and Setup
### Install dependencies
```
pip install -r requirements.txt
```

### Initialize microservices
build and run microservices (1-4) in docker.
```
make local-start
```

### Data ingestion
crawl nice guideline and save it to mongoDB, then preprocess and save to qdrantDB
```
# one link
make local-test-nice

# all links
make local-ingest-data
```

### Run inference
```
python src/inference_pipeline/main.py --query "What is the treatment for pregnant women with hypertension?"
```

### Run evaluation
```
# Evaluate LLM
make evaluate-llm

# Evaluate RAG system
make evaluate-rag

# Evaluate Chatbot on monitoring dataset
make evaluate-llm-monitoring
```

## Integration with AWS
### Data crawler
- Save the data/links.txt in a S3 bucket
- Create a SQS
- Create a lambda that reads the links.txt from this bucket and send message to SQS. Copy aws/link_processor.py to the created lambda. Add the required environment variables.
- Push the pharmassist-data-crawler image to ECR. Create a lambda using this image. Add the environment variable MONGO_DATABASE_HOST. Set the memory to 512 MB and timeout to 5 minutes. Add the SQS as trigger.
- Finally, create a EventBridge Scheduler to create a cron job that triggers the lambda to process the links in the text file.

## Reference

https://github.com/decodingml/llm-twin-course/tree/main
