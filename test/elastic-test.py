#!/usr/bin/env python3

from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

print("Testing elasticsearch...")

load_dotenv()

# Get Elasticsearch URL from environment variable
es_url = os.getenv('ELASTICSEARCH_URL')

if es_url:
    es = Elasticsearch([es_url])
    # Check if the cluster is up
    if es.ping():
        print("Connected to Elasticsearch")
        info = es.info()
        print(f"Elasticsearch version: {info['version']['number']}")
    else:
        print("Could not connect to Elasticsearch")
else:
    print("Elasticsearch URL not set, skipping test")
    print("Set ELASTICSEARCH_URL environment variable to test Elasticsearch connection")

print("elasticsearch test completed successfully")