 #!/usr/bin/env python3

from dotenv import load_dotenv
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
from pydantic import BaseModel, Field
import uvicorn
from sklearn.preprocessing import MinMaxScaler
import openai
import fnmatch
from typing import Optional, Dict, Any

# Set up logging with better structure
LOG_DIR = '/mnt/vectordb'
LOG_FILE = os.path.join(LOG_DIR, 'rag_system.log')

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# Global lock for thread-safe operations
index_lock = threading.Lock()

# Initialize Elasticsearch client
load_dotenv()

# Initialize Elasticsearch with proper error handling
# Defer connection until needed to allow testing without ES
def get_elasticsearch_client() -> Elasticsearch:
    """Get or create Elasticsearch client.
    
    Returns:
        Elasticsearch client instance
        
    Raises:
        ConnectionError: If unable to connect to Elasticsearch
    """
    global es
    if es is None:
        es_url = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
        try:
            es = Elasticsearch([es_url], verify_certs=True, timeout=30)
            if not es.ping():
                raise ConnectionError(f"Failed to connect to Elasticsearch at {es_url}")
            logger.info(f"Connected to Elasticsearch at {es_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise
    return es

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
openai.api_key = os.getenv('OPENAI_API_KEY')

def load_metadata():
    """Load metadata from file or return default.
    
    Returns:
        Dictionary containing metadata
    """
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                # Convert list back to set for processed_ids
                data['processed_ids'] = set(data.get('processed_ids', []))
                return data
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error loading metadata: {e}. Using default.")
    return {'last_processed': '1970-01-01T00:00:00.000Z', 'processed_ids': set()}

def save_metadata(metadata):
    """Save metadata to file with proper serialization.
    
    Args:
        metadata: Metadata dictionary to save
    """
    try:
        metadata_to_save = metadata.copy()
        metadata_to_save['processed_ids'] = list(metadata_to_save.get('processed_ids', []))
        with open(metadata_file, 'w') as f:
            json.dump(metadata_to_save, f)
        logger.info("Metadata saved successfully")
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")
        raise

def cleanup_old_processed_ids(max_ids: int = 10000):
    """Clean up old processed IDs to prevent memory leaks.
    
    Args:
        max_ids: Maximum number of processed IDs to keep
    """
    if len(metadata.get('processed_ids', set())) > max_ids:
        # Keep only the most recent IDs
        processed_list = list(metadata['processed_ids'])
        metadata['processed_ids'] = set(processed_list[-max_ids:])
        save_metadata(metadata)
        logger.info(f"Cleaned up processed IDs, now tracking {len(metadata['processed_ids'])}")

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

def process_batch(batch: list) -> None:
    """Process a batch of logs and add them to the FAISS index.
    
    Args:
        batch: List of log entries from Elasticsearch
    """
    try:
        processed_logs, timestamps, hostnames = zip(*[preprocess_log(log['_source']) for log in batch])
        vectors = vectorize_logs(processed_logs, timestamps)
        faiss.normalize_L2(vectors)
        
        with index_lock:
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
        
        logger.info(f"Processed batch of {len(batch)} logs, total vectors: {index.ntotal}")
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise

def process_new_logs():
    """Process new logs from Elasticsearch and add to FAISS index."""
    global es
    last_processed = metadata.get('last_processed', '1970-01-01T00:00:00.000Z')
    logger.info(f"Processing logs since {last_processed}")
    
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
            es_client = get_elasticsearch_client()
            for hit in helpers.scan(es_client, query=query, index="logs", size=batch_size):
                processing_queue.put(hit)
        except Exception as e:
            logger.error(f"Error in producer: {e}")
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
            logger.error(f"Error in consumer: {e}")

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

    logger.info(f"Processed {index.ntotal} vectors in total")

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

def generate_llm_response(query: str, relevant_logs: list) -> str:
    """Generate LLM response using the updated OpenAI API.
    
    Args:
        query: The user's query text
        relevant_logs: List of relevant log entries
        
    Returns:
        Generated response string
    """
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
        # Use the new OpenAI API format (v1.0+)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        if relevant_logs:
            return "I was unable to generate a response due to an error with the AI service. Here are the relevant logs:\n" + "\n\n".join([f"{log['hostname']} at {log['timestamp']}: {log['message'][:200]}" for log in relevant_logs[:3]])
        return "I don't know"

app = FastAPI(
    title="Log Analysis RAG System",
    description="A Retrieval-Augmented Generation system for log analysis",
    version="1.1.0"
)

class Query(BaseModel):
    """Query model for RAG system."""
    text: str = Field(..., description="The query text")
    k: int = Field(default=5, ge=1, le=20, description="Number of relevant logs to retrieve")
    start_time: Optional[str] = Field(default=None, description="Start time filter (ISO format)")
    end_time: Optional[str] = Field(default=None, description="End time filter (ISO format)")
    hostname_pattern: Optional[str] = Field(default=None, description="Hostname pattern filter")

class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str = Field(..., description="The generated response")
    relevant_logs: list = Field(..., description="List of relevant log entries")
    query_time: str = Field(..., description="Query execution timestamp")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    index_size: int
    metadata_size: int

def get_query_time() -> str:
    """Get current query timestamp."""
    return datetime.utcnow().isoformat()

@app.post("/rag_query", response_model=QueryResponse)
async def api_rag_query(query: Query):
    """Process a RAG query and return relevant logs with LLM-generated answer."""
    try:
        time_range = None
        if query.start_time and query.end_time:
            time_range = (parse(query.start_time), parse(query.end_time))
        
        relevant_logs = rag_query(query.text, time_range, query.hostname_pattern, query.k)
        llm_response = generate_llm_response(query.text, relevant_logs)
        
        return QueryResponse(
            answer=llm_response,
            relevant_logs=relevant_logs,
            query_time=get_query_time()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=get_query_time(),
        index_size=index.ntotal,
        metadata_size=len(metadata)
    )

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    return {
        "index_size": index.ntotal,
        "metadata_size": len(metadata),
        "processed_ids_count": len(metadata.get('processed_ids', set())),
        "last_processed": metadata.get('last_processed'),
        "timestamp": get_query_time()
    }

def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

def schedule_processing():
    schedule.every(1).hour.do(process_new_logs)
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    """Main entry point for the RAG System.
    
    Handles initialization, processing, and graceful shutdown.
    """
    shutdown_event = threading.Event()
    
    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received")
        shutdown_event.set()
        logger.info("Saving progress before shutdown...")
        save_metadata(metadata)
        faiss.write_index(index, index_file)
        logger.info("Progress saved. Exiting gracefully.")
        os._exit(0)
    
    try:
        import signal
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        process_new_logs()  # Initial processing
        
        # Start API server in a separate thread
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        logger.info("API server started")
        
        # Schedule periodic cleanup
        schedule.every().day.at("02:00").do(cleanup_old_processed_ids)
        schedule.every().day.at("04:00").do(cleanup_old_processed_ids)
        
        # Main scheduling loop
        while not shutdown_event.is_set():
            try:
                schedule.run_pending()
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}")
                time.sleep(5)
                
    except KeyboardInterrupt:
        logger.info("Process interrupted")
        shutdown_event.set()
    finally:
        logger.info("Saving progress...")
        save_metadata(metadata)
        faiss.write_index(index, index_file)
        logger.info("Progress saved. You can safely restart the script later.")