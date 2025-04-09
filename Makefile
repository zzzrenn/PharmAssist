# ======================================
# ------- Docker Infrastructure --------
# ======================================

local-start: # Build and start your local Docker infrastructure.
	docker compose -f docker-compose.yml up --build -d

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