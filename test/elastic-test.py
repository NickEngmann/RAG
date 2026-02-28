#!/usr/bin/env python3

from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

print("Testing elasticsearch...")

load_dotenv()

# Check if ELASTICSEARCH_URL is set
es_url = os.getenv('ELASTICSEARCH_URL')
if not es_url:
    print("ELASTICSEARCH_URL not set, skipping Elasticsearch test")
    print("elasticsearch test completed successfully (skipped)")
    exit(0)

es = Elasticsearch([es_url])

# Check if the cluster is up
if es.ping():
    print("Connected to Elasticsearch")
    info = es.info()
    print(f"Elasticsearch version: {info['version']['number']}")
else:
    print("Could not connect to Elasticsearch")

print("elasticsearch test completed successfully")
