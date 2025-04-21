import os
import sys

# Add the project root to path to resolve module imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.models.embeddings import embedding_model_factory

model = embedding_model_factory.create_embedding_model()

texts = ["Hello, world!", "This is a test."]
embeddings = model.embed(texts)

print(len(embeddings[0]))
