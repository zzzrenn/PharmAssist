# ======================================
# ------- Docker Infrastructure --------
# ======================================

local-start: # Build and start your local Docker infrastructure.
	docker compose -f docker-compose.yml up --build

local-stop: # Stop your local Docker infrastructure.
	docker compose -f docker-compose.yml down --remove-orphans

# ======================================
# ---------- Crawling Data -------------
# ======================================

local-test-nice: # Make a call to your local AWS Lambda (hosted in Docker) to crawl a NICE guideline.
	curl -X POST "http://localhost:9010/2015-03-31/functions/function/invocations" \
	  	-d "{\"Records\": [{\"body\": \"https://www.nice.org.uk/guidance/ng106\"}]}"

local-ingest-data: # Ingest all links from data/links.txt by calling your local AWS Lambda hosted in Docker.
	while IFS= read -r link; do \
		link=`echo "$$link" | tr -d '\r'`; \
		echo "Processing: $$link"; \
		curl -X POST "http://localhost:9010/2015-03-31/functions/function/invocations" \
			-d "{\"Records\": [{\"body\": \"$$link\"}]}"; \
		echo "\n"; \
	done < data/links.txt

# ======================================
# ---------- Feature Pipeline ---------
# ======================================

local-test-inference-pipeline: # Test the inference pipeline.
	docker compose -f docker-compose-inference.yml up --build

# ======================================
# ---------- Training Pipeline ---------
# ======================================
download-instruct-dataset: # Download the NICE Guideline Instruct Dataset from Hugging Face.
	python src/training_pipeline/download_dataset.py

create-sagemaker-execution-role: # Create an AWS SageMaker execution role you need for the training and inference pipelines.
	cd src && python -m aws.create_execution_role

start-training-pipeline-dummy-mode: # Start the training pipeline in AWS SageMaker.
	python src/training_pipeline/run_on_sagemaker.py --is-dummy


# ======================================
# ---------- Inference Pipeline ---------
# ======================================

call-inference-pipeline: # Call the inference pipeline client
	python src/inference_pipeline/main.py

# ======================================
# ---------- Evaluation models ---------
# ======================================

evaluate-llm: # Run evaluation tests on the LLM model's performance
	cd src/inference_pipeline && python -m evaluation.evaluate

evaluate-rag: # Run evaluation tests specifically on the RAG system's performance
	cd src/inference_pipeline && python -m evaluation.evaluate_rag

evaluate-llm-monitoring: # Run evaluation tests for monitoring the LLM system
	cd src/inference_pipeline && python -m evaluation.evaluate_monitoring
