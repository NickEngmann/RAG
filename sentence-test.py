#!/usr/bin/env python3

from sentence_transformers import SentenceTransformer
import numpy as np

print("Testing sentence_transformers...")

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ['This is a test sentence.', 'Another sentence for embedding.']

embeddings = model.encode(sentences)
print(f"Shape of embeddings: {embeddings.shape}")
print(f"Type of embeddings: {type(embeddings)}")
print(f"Sample embedding:\n{embeddings[0][:5]}...")  # Print first 5 values of first embedding

print("sentence_transformers test completed successfully")