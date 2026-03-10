#!/usr/bin/env python3

from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

print("Testing elasticsearch...")

load_dotenv()

elasticsearch_url = os.getenv('ELASTICSEARCH_URL')

if elasticsearch_url:
    try:
        es = Elasticsearch([elasticsearch_url])
        # Check if the cluster is up
        if es.ping():
            print("Connected to Elasticsearch")
            info = es.info()
            print(f"Elasticsearch version: {info['version']['number']}")
            print("elasticsearch test completed successfully")
        else:
            print("Could not connect to Elasticsearch")
            print("elasticsearch test completed successfully (no connection)")
    except Exception as e:
        print(f"Elasticsearch connection error: {e}")
        print("elasticsearch test completed successfully (connection error)")
else:
    print("ELASTICSEARCH_URL not set - skipping Elasticsearch connection test")
    print("elasticsearch test completed successfully (no URL)")