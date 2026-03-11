 #!/usr/bin/env python3

from dotenv import load_dotenv
import os
from datetime import datetime
from dateutil.parser import parse
from elasticsearch import Elasticsearch, helpers
# sentence_transformers import is lazy - moved inside function to avoid import errors when not needed
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

# Initialize Elasticsearch client (lazy)
load_dotenv()

# Global ES client - initialized lazily
_es = None

def _get_es_client():
    """Lazy initialization of the Elasticsearch client."""
    global _es
    if _es is None:
        es_url = os.getenv('ELASTICSEARCH_URL')
        if es_url:
            _es = Elasticsearch([es_url])
        else:
            _es = None
    return _es

# Initialize FAISS index (lazy - will be created when needed)
index_file = "/mnt/vectordb/vector_index.faiss"

# Metadata storage
metadata_file = "/mnt/vectordb/metadata.json"

# Initialize time scaler (lazy - will be created when needed)
time_scaler = None

# OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Global model and index - initialized lazily
_model = None
_index = None
_time_scaler = None

def _get_model():
    """Lazy initialization of the embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def _get_index():
    """Lazy initialization of the FAISS index."""
    global _index, _time_scaler
    if _index is None:
        model = _get_model()
        dimension = model.get_sentence_embedding_dimension() + 1  # +1 for timestamp
        if os.path.exists(index_file):
            _index = faiss.read_index(index_file)
            logging.info(f"Loaded existing index with {_index.ntotal} vectors")
        else:
            _index = faiss.IndexFlatIP(dimension)
            faiss.write_index(_index, index_file)
            logging.info("Created new FAISS index")
    return _index

def _get_time_scaler():
    """Lazy initialization of the time scaler."""
    global _time_scaler
    if _time_scaler is None:
        from sklearn.preprocessing import MinMaxScaler
        _time_scaler = MinMaxScaler()
    return _time_scaler

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
            for hit in helpers.scan(_get_es_client(), query=query, index="logs", size=batch_size):
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

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message['content'].strip()
    except openai.error.InvalidRequestError as e:
        logging.error(f"Invalid request to OpenAI API: {e}")
        return "I don't know"
    except openai.error.RateLimitError as e:
        logging.error(f"Rate limit exceeded: {e}")
        return "I don't know"
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "I don't know"

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

class SummaryQuery(BaseModel):
    hostname_pattern: str = None
    start_time: str = None
    end_time: str = None
    top_n: int = 10

@app.post("/summary")
async def api_summary(query: SummaryQuery):
    """Generate a summary of logs by hostname or time period.
    
    Returns statistics about log entries including:
    - Total log count
    - Logs per hostname
    - Time range of logs
    - Most common log patterns
    """
    try:
        # Build filter criteria
        filtered_logs = []
        
        for log_id, log_data in metadata.items():
            if not isinstance(log_data, dict):
                continue
                
            # Apply hostname filter
            if query.hostname_pattern:
                if not fnmatch.fnmatch(log_data.get('hostname', ''), query.hostname_pattern):
                    continue
            
            # Apply time range filter
            if query.start_time or query.end_time:
                log_time = parse(log_data['timestamp'])
                if query.start_time:
                    start = parse(query.start_time)
                    if log_time < start:
                        continue
                if query.end_time:
                    end = parse(query.end_time)
                    if log_time > end:
                        continue
            
            filtered_logs.append(log_data)
        
        if not filtered_logs:
            return {
                "total_logs": 0,
                "hostnames": {},
                "time_range": None,
                "message": "No logs found matching the criteria"
            }
        
        # Calculate statistics
        hostname_counts = {}
        for log in filtered_logs:
            hostname = log.get('hostname', 'unknown')
            hostname_counts[hostname] = hostname_counts.get(hostname, 0) + 1
        
        # Sort by count and get top N
        top_hostnames = dict(sorted(hostname_counts.items(), key=lambda x: x[1], reverse=True)[:query.top_n])
        
        # Calculate time range
        timestamps = [parse(log['timestamp']) for log in filtered_logs if 'timestamp' in log]
        time_range = {
            "start": min(timestamps).isoformat() if timestamps else None,
            "end": max(timestamps).isoformat() if timestamps else None
        } if timestamps else None
        
        # Extract log message patterns (first 50 chars as pattern)
        patterns = {}
        for log in filtered_logs:
            message = log.get('message', '')[:50]
            patterns[message] = patterns.get(message, 0) + 1
        
        top_patterns = dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5])
        
        return {
            "total_logs": len(filtered_logs),
            "hostnames": top_hostnames,
            "time_range": time_range,
            "top_patterns": top_patterns,
            "message": f"Summary of {len(filtered_logs)} logs"
        }
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

def schedule_processing():
    schedule.every(1).hour.do(process_new_logs)
    while True:
        schedule.run_pending()
        time.sleep(60)

# Helper functions for document summarization

def get_embedding_model():
    """Get the sentence transformer embedding model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_faiss_index():
    """Load the Faiss index from disk."""
    index_file = 'faiss.index'
    if os.path.exists(index_file):
        return faiss.read_index(index_file)
    return None

def get_elasticsearch_client():
    """Get the Elasticsearch client."""
    return Elasticsearch(
        ['http://localhost:9200'],
        timeout=30
    )

def summarize_documents(query: str, top_k: int = 5, use_faiss: bool = True, use_elasticsearch: bool = True) -> dict:
    """Summarize documents based on a query using semantic search.
    
    Args:
        query: The search query text
        top_k: Number of documents to retrieve
        use_faiss: Whether to use Faiss for semantic search
        use_elasticsearch: Whether to use Elasticsearch for document retrieval
        
    Returns:
        Dictionary containing:
        - summary: Generated summary of relevant documents
        - sources: List of source documents
        - query: The original query
    """
    try:
        # Get embedding model
        embedding_model = get_embedding_model()
        
        # Encode the query
        query_embedding = embedding_model.encode(query)
        
        # Retrieve relevant documents
        sources = []
        
        if use_faiss:
            faiss_index = get_faiss_index()
            if faiss_index is not None:
                # Search Faiss index
                distances, indices = faiss_index.search(query_embedding.reshape((1, -1)), top_k)
                # Load metadata and get document content
                for idx in indices[0]:
                    if idx < len(metadata):
                        meta = list(metadata.values())[idx]
                        sources.append({
                            'text': meta.get('text', ''),
                            'metadata': meta.get('metadata', {})
                        })
        
        if use_elasticsearch and not sources:
            # Fall back to Elasticsearch
            es_client = get_elasticsearch_client()
            response = es_client.search(
                index='documents',
                body={
                    'query': {
                        'match': {
                            'text': query
                        }
                    },
                    'size': top_k
                }
            )
            for hit in response['hits']['hits']:
                source = hit['_source']
                sources.append({
                    'text': source.get('text', ''),
                    'metadata': source.get('metadata', {})
                })
        
        # Generate summary from retrieved documents
        if sources:
            # Combine document texts
            document_texts = '\n\n'.join([s['text'] for s in sources])
            
            # Create a simple summary (in production, this would use an LLM)
            summary = f"Summary of {len(sources)} relevant documents for query '{query}':\n\n"
            summary += document_texts[:1000] + ('...' if len(document_texts) > 1000 else '')
            
            return {
                'summary': summary,
                'sources': sources,
                'query': query,
                'num_documents': len(sources)
            }
        else:
            return {
                'summary': 'No relevant documents found.',
                'sources': [],
                'query': query,
                'num_documents': 0
            }
    
    except Exception as e:
        logging.error(f"Error in summarize_documents: {e}")
        return {
            'summary': 'Error generating summary.',
            'sources': [],
            'query': query,
            'error': str(e)
        }


def generate_summary_stats(logs, hostname_pattern=None, start_time=None, end_time=None):
    """Generate summary statistics from log data with optional filters."""
    # Filter by hostname if provided
    if hostname_pattern:
        logs = [log for log in logs if hostname_pattern in log.get('hostname', '')]
    
    # Filter by time range if provided
    if start_time or end_time:
        filtered_logs = []
        for log in logs:
            ts = log.get('timestamp', '')
            if start_time and ts < start_time:
                continue
            if end_time and ts > end_time:
                continue
            filtered_logs.append(log)
        logs = filtered_logs
    
    # Calculate statistics
    total_logs = len(logs)
    
    # Get unique hostnames and counts
    hostnames = {}
    for log in logs:
        hostname = log.get('hostname', 'unknown')
        hostnames[hostname] = hostnames.get(hostname, 0) + 1
    
    # Get time range
    time_range = {'start': None, 'end': None}
    if logs:
        timestamps = [log.get('timestamp', '') for log in logs if log.get('timestamp')]
        if timestamps:
            time_range['start'] = min(timestamps)
            time_range['end'] = max(timestamps)
    
    # Count error and warning messages
    error_count = sum(1 for log in logs if 'error' in log.get('message', '').lower())
    warning_count = sum(1 for log in logs if 'warn' in log.get('message', '').lower())
    
    return {
        'total_logs': total_logs,
        'hostnames': hostnames,
        'time_range': time_range,
        'error_count': error_count,
        'warning_count': warning_count
    }


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