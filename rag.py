 #!/usr/bin/env python3

import os
from datetime import datetime
from dateutil.parser import parse
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import schedule
import time
import gc
import threading
import queue
import json
from tqdm import tqdm
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sklearn.preprocessing import MinMaxScaler
import openai
import fnmatch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Elasticsearch client
es = Elasticsearch(['http://your_elasticsearch_ip:9200'])

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
dimension = model.get_sentence_embedding_dimension() + 1  # +1 for timestamp
index_file = "/mnt/vectordb/vector_index.faiss"

if os.path.exists(index_file):
    index = faiss.read_index(index_file)
    logging.info(f"Loaded existing index with {index.ntotal} vectors")
else:
    index = faiss.IndexFlatIP(dimension)
    faiss.write_index(index, index_file)
    logging.info("Created new FAISS index")

# Metadata storage
metadata_file = "/mnt/vectordb/metadata.json"

# Initialize time scaler
time_scaler = MinMaxScaler()

# OpenAI API key
openai.api_key = 'your-openai-api-key'

def load_metadata():
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return {'last_processed': '1970-01-01T00:00:00.000Z', 'processed_ids': set()}

def save_metadata(metadata):
    metadata_to_save = metadata.copy()
    metadata_to_save['processed_ids'] = list(metadata_to_save['processed_ids'])
    with open(metadata_file, 'w') as f:
        json.dump(metadata_to_save, f)

metadata = load_metadata()
metadata['processed_ids'] = set(metadata.get('processed_ids', []))

def preprocess_log(log_entry):
    timestamp = parse(log_entry['@timestamp'])
    timestamp_value = timestamp.timestamp()
    normalized_time = time_scaler.fit_transform([[timestamp_value]])[0][0]
    message = log_entry['message'][:1000]  # Truncate to save memory
    hostname = log_entry.get('hostname', 'unknown')
    return message, normalized_time, hostname

def vectorize_logs(log_texts, timestamps):
    text_vectors = model.encode(log_texts, show_progress_bar=True, batch_size=32)
    combined_vectors = np.hstack((text_vectors, np.array(timestamps).reshape(-1, 1)))
    return combined_vectors

def process_batch(batch):
    processed_logs, timestamps, hostnames = zip(*[preprocess_log(log['_source']) for log in batch])
    vectors = vectorize_logs(processed_logs, timestamps)
    faiss.normalize_L2(vectors)
    
    with threading.Lock():
        index.add(vectors)
        for i, log in enumerate(batch):
            vector_id = str(index.ntotal - len(batch) + i)
            metadata[vector_id] = {
                'id': log['_id'],
                'timestamp': log['_source']['@timestamp'],
                'message': log['_source']['message'],
                'hostname': hostnames[i]
            }
            metadata['processed_ids'].add(log['_id'])

def process_new_logs():
    last_processed = metadata.get('last_processed', '1970-01-01T00:00:00.000Z')
    logging.info(f"Processing logs since {last_processed}")
    
    query = {
        "query": {
            "bool": {
                "must": [
                    {"range": {"@timestamp": {"gt": last_processed}}}
                ],
                "must_not": [
                    {"ids": {"values": list(metadata['processed_ids'])}}
                ]
            }
        },
        "sort": [{"@timestamp": "asc"}]
    }

    batch_size = 1000
    processing_queue = queue.Queue(maxsize=10)

    def producer():
        try:
            for hit in helpers.scan(es, query=query, index="logs", size=batch_size):
                processing_queue.put(hit)
        except Exception as e:
            logging.error(f"Error in producer: {e}")
        finally:
            processing_queue.put(None)  # Signal end of data

    def consumer():
        batch = []
        try:
            for hit in iter(processing_queue.get, None):
                if hit['_id'] not in metadata['processed_ids']:
                    batch.append(hit)
                    if len(batch) >= batch_size:
                        process_batch(batch)
                        batch = []
            if batch:
                process_batch(batch)
        except Exception as e:
            logging.error(f"Error in consumer: {e}")

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

    if metadata:
        metadata['last_processed'] = max(log['timestamp'] for log in metadata.values() if isinstance(log, dict) and 'timestamp' in log)
        save_metadata(metadata)
        faiss.write_index(index, index_file)

    logging.info(f"Processed {index.ntotal} vectors in total")

def rag_query(query_text, time_range=None, hostname_pattern=None, k=5):
    query_vector = model.encode([query_text])
    
    if time_range:
        start_time, end_time = time_range
        start_timestamp = start_time.timestamp()
        end_timestamp = end_time.timestamp()
        
        normalized_start = time_scaler.transform([[start_timestamp]])[0][0]
        normalized_end = time_scaler.transform([[end_timestamp]])[0][0]
        
        time_context = (normalized_start + normalized_end) / 2
        query_vector = np.hstack((query_vector, np.array([[time_context]])))
    else:
        query_vector = np.hstack((query_vector, np.array([[0.5]])))  # Neutral time context
    
    faiss.normalize_L2(query_vector)
    _, I = index.search(query_vector, k * 2)  # Fetch more results initially
    
    results = []
    for i in I[0]:
        if str(i) in metadata:
            result = metadata[str(i)]
            if time_range:
                result_time = parse(result['timestamp'])
                if not (start_time <= result_time <= end_time):
                    continue
            if hostname_pattern:
                if not fnmatch.fnmatch(result['hostname'], hostname_pattern):
                    continue
            results.append(result)
        if len(results) == k:
            break
    
    return results

def generate_llm_response(query, relevant_logs):
    system_prompt = """
    You are an AI research assistant analyzing text chunks from web sources to answer queries accurately and concisely.

    Key instructions:
    1. Use ONLY the information contained in the provided text chunks to formulate your response.
    2. If no text chunks are provided, or if the chunks contain no relevant information to the query, respond with "I don't know".
    3. Do not use any external knowledge or make assumptions beyond what is explicitly stated in the chunks.
    4. Do mention or reference the sources of the information.

    Guidelines for responses:
    1. Provide concise, relevant answers that directly address the query.
    2. Synthesize information from multiple chunks if applicable.
    3. Maintain a professional and objective tone.
    4. If the information in the chunks is insufficient or contradictory, state this clearly.
    5. If a query is ambiguous, respond based solely on the most likely interpretation given the available chunks.

    Your goal is to deliver clear, accurate information based strictly on the provided text chunks, without embellishment or external knowledge.
    """

    chunks = "\n\n".join([f"Chunk {i+1} (Hostname: {log['hostname']}, Timestamp: {log['timestamp']}):\n{log['message']}" for i, log in enumerate(relevant_logs)])
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}\n\nRelevant log entries:\n{chunks}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500
    )

    return response.choices[0].message['content'].strip()

app = FastAPI()

class Query(BaseModel):
    text: str
    k: int = 5
    start_time: str = None
    end_time: str = None
    hostname_pattern: str = None

@app.post("/rag_query")
async def api_rag_query(query: Query):
    try:
        time_range = None
        if query.start_time and query.end_time:
            time_range = (parse(query.start_time), parse(query.end_time))
        
        relevant_logs = rag_query(query.text, time_range, query.hostname_pattern, query.k)
        llm_response = generate_llm_response(query.text, relevant_logs)
        
        return {
            "answer": llm_response,
            "relevant_logs": relevant_logs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

def schedule_processing():
    schedule.every(1).hour.do(process_new_logs)
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    try:
        process_new_logs()  # Initial processing
        api_thread = threading.Thread(target=run_api)
        api_thread.start()
        schedule_processing()
    except KeyboardInterrupt:
        logging.info("Process interrupted. Saving progress...")
        save_metadata(metadata)
        faiss.write_index(index, index_file)
        logging.info("Progress saved. You can safely restart the script later.")