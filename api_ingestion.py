#!/usr/bin/env python3
"""
Log Ingestion API Module

This module provides endpoints for ingesting logs into Elasticsearch.
It extends the RAG system with manual log upload capabilities.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import logging
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize Elasticsearch client
ES_URL = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
es = Elasticsearch([ES_URL])

app = FastAPI(
    title="Log Ingestion API",
    description="API for ingesting logs into Elasticsearch for RAG processing",
    version="1.0.0"
)


class LogEntry(BaseModel):
    """Single log entry model"""
    message: str
    timestamp: Optional[str] = None
    hostname: Optional[str] = None
    level: Optional[str] = None
    source: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Error: Connection timeout to database",
                "timestamp": "2024-01-15T10:30:00.000Z",
                "hostname": "server-01",
                "level": "ERROR",
                "source": "app.log"
            }
        }


class BulkLogRequest(BaseModel):
    """Bulk log request model"""
    logs: List[LogEntry]
    index_name: Optional[str] = "logs"
    
    class Config:
        json_schema_extra = {
            "example": {
                "logs": [
                    {
                        "message": "User login successful",
                        "timestamp": "2024-01-15T10:30:00.000Z",
                        "hostname": "web-server-01",
                        "level": "INFO"
                    },
                    {
                        "message": "Database connection established",
                        "timestamp": "2024-01-15T10:31:00.000Z",
                        "hostname": "db-server-01",
                        "level": "INFO"
                    }
                ],
                "index_name": "application-logs"
            }
        }


class IngestionResponse(BaseModel):
    """Response model for log ingestion"""
    success: bool
    ingested_count: int
    failed_count: int
    errors: Optional[List[Dict[str, Any]]] = None
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "ingested_count": 10,
                "failed_count": 0,
                "errors": None,
                "message": "Successfully ingested 10 log entries"
            }
        }


def validate_elasticsearch_connection():
    """Check if Elasticsearch is accessible"""
    try:
        if not es.ping():
            raise HTTPException(status_code=503, detail="Elasticsearch is not available")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Elasticsearch connection failed: {str(e)}")


def normalize_timestamp(timestamp_str: str) -> str:
    """Normalize timestamp to ISO format"""
    try:
        # Try parsing common timestamp formats
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    except ValueError:
        # If parsing fails, return as-is
        return timestamp_str


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        es.ping()
        return {
            "status": "healthy",
            "elasticsearch": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "elasticsearch": "disconnected",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.post("/logs/bulk", response_model=IngestionResponse)
async def ingest_bulk_logs(request: BulkLogRequest):
    """
    Ingest multiple log entries at once.
    
    This endpoint accepts a list of log entries and ingests them into Elasticsearch.
    Each log entry should contain at minimum a 'message' field.
    """
    validate_elasticsearch_connection()
    
    errors = []
    success_count = 0
    
    actions = []
    for i, log in enumerate(request.logs):
        try:
            # Normalize timestamp if provided
            log_data = {
                "message": log.message,
                "timestamp": normalize_timestamp(log.timestamp) if log.timestamp else datetime.utcnow().isoformat(),
                "hostname": log.hostname or "unknown",
                "level": log.level or "INFO",
                "source": log.source or "api-upload"
            }
            
            action = {
                "_index": request.index_name,
                "_source": log_data
            }
            actions.append(action)
        except Exception as e:
            errors.append({"index": i, "error": str(e), "log": log.dict()})
    
    if not actions:
        raise HTTPException(status_code=400, detail="No valid log entries to ingest")
    
    try:
        # Use Elasticsearch bulk API
        response = helpers.bulk(es, actions, raise_on_error=False)
        success_count = response[0]
        failures = response[1]
        
        if failures:
            for failure in failures:
                if "create" in failure:
                    errors.append({
                        "index": len(errors),
                        "error": failure["create"].get("error", "Unknown error"),
                        "status": failure["create"].get("status")
                    })
        
        return IngestionResponse(
            success=len(errors) == 0,
            ingested_count=success_count,
            failed_count=len(errors),
            errors=errors if errors else None,
            message=f"Successfully ingested {success_count} log entries"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest logs: {str(e)}")


@app.post("/logs/single", response_model=IngestionResponse)
async def ingest_single_log(log: LogEntry):
    """
    Ingest a single log entry.
    
    This is a convenience endpoint for ingesting one log at a time.
    """
    validate_elasticsearch_connection()
    
    try:
        log_data = {
            "message": log.message,
            "timestamp": normalize_timestamp(log.timestamp) if log.timestamp else datetime.utcnow().isoformat(),
            "hostname": log.hostname or "unknown",
            "level": log.level or "INFO",
            "source": log.source or "api-upload"
        }
        
        response = es.index(index="logs", document=log_data)
        
        return IngestionResponse(
            success=True,
            ingested_count=1,
            failed_count=0,
            errors=None,
            message=f"Successfully ingested log entry with id: {response['_id']}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest log: {str(e)}")


@app.post("/logs/upload", response_model=IngestionResponse)
async def upload_log_file(
    file: UploadFile = File(..., description="Log file to upload (JSON or plain text)"),
    index_name: str = Form("logs", description="Elasticsearch index name")
):
    """
    Upload a log file for ingestion.
    
    Supports JSON format (array of log objects) and plain text format (one log per line).
    """
    validate_elasticsearch_connection()
    
    try:
        content = await file.read()
        
        # Try to parse as JSON first
        try:
            logs_data = json.loads(content)
            if isinstance(logs_data, list):
                logs = logs_data
            else:
                logs = [logs_data]
        except json.JSONDecodeError:
            # Parse as plain text (one log per line)
            logs = [line.strip() for line in content.decode('utf-8').split('\n') if line.strip()]
        
        if not logs:
            raise HTTPException(status_code=400, detail="No log entries found in file")
        
        actions = []
        for i, log in enumerate(logs):
            # Handle both string logs and dict logs
            if isinstance(log, str):
                log_data = {
                    "message": log,
                    "timestamp": datetime.utcnow().isoformat(),
                    "hostname": "unknown",
                    "level": "INFO",
                    "source": f"file-upload:{file.filename}"
                }
            else:
                log_data = {
                    "message": log.get("message", str(log)),
                    "timestamp": log.get("timestamp", datetime.utcnow().isoformat()),
                    "hostname": log.get("hostname", "unknown"),
                    "level": log.get("level", "INFO"),
                    "source": f"file-upload:{file.filename}"
                }
            
            action = {
                "_index": index_name,
                "_source": log_data
            }
            actions.append(action)
        
        response = helpers.bulk(es, actions, raise_on_error=False)
        success_count = response[0]
        failures = response[1]
        
        errors = []
        if failures:
            for failure in failures:
                if "create" in failure:
                    errors.append({
                        "index": len(errors),
                        "error": failure["create"].get("error", "Unknown error"),
                        "status": failure["create"].get("status")
                    })
        
        return IngestionResponse(
            success=len(errors) == 0,
            ingested_count=success_count,
            failed_count=len(errors),
            errors=errors if errors else None,
            message=f"Successfully ingested {success_count} log entries from {file.filename}"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.get("/logs/stats", response_model=Dict[str, Any])
async def get_log_stats():
    """
    Get statistics about ingested logs.
    
    Returns count of logs per index and overall statistics.
    """
    validate_elasticsearch_connection()
    
    try:
        # Get all indices
        indices_response = es.cat.indices(format="json")
        
        stats = {
            "total_indices": len(indices_response),
            "indices": [],
            "total_logs": 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for index_info in indices_response:
            index_name = index_info.get("index", "unknown")
            doc_count = int(index_info.get("docs.count", 0))
            
            stats["indices"].append({
                "name": index_name,
                "document_count": doc_count
            })
            stats["total_logs"] += doc_count
        
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
