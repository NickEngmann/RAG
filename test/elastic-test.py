#!/usr/bin/env python3

from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

print("Testing elasticsearch...")

load_dotenv()

# Replace the existing Elasticsearch and OpenAI initialization lines with:
es = Elasticsearch([os.getenv('ELASTICSEARCH_URL')])

# Check if the cluster is up
if es.ping():
    print("Connected to Elasticsearch")
    info = es.info()
    print(f"Elasticsearch version: {info['version']['number']}")
else:
    print("Could not connect to Elasticsearch")

print("elasticsearch test completed successfully")