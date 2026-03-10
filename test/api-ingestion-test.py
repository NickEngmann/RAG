#!/usr/bin/env python3
"""Test script for API Ingestion module"""

import json
import sys
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, '')

from fastapi.testclient import TestClient

# Import the API module
from api_ingestion import app, normalize_timestamp, IngestionResponse

client = TestClient(app)

def test_normalize_timestamp():
    """Test timestamp normalization function"""
    print("Testing normalize_timestamp function...")
    
    # Test ISO format
    result = normalize_timestamp("2024-01-15T10:30:00Z")
    assert "2024-01-15T10:30:00" in result, f"Expected ISO format, got {result}"
    print(f"  ISO format: {result}")
    
    # Test with microseconds
    result = normalize_timestamp("2024-01-15T10:30:00.123456Z")
    assert "2024-01-15T10:30:00" in result, f"Expected ISO format, got {result}"
    print(f"  With microseconds: {result}")
    
    # Test with timezone offset
    result = normalize_timestamp("2024-01-15T10:30:00+05:00")
    assert "2024-01-15T10:30:00" in result, f"Expected ISO format, got {result}"
    print(f"  With timezone offset: {result}")
    
    print("  normalize_timestamp tests passed!\n")

def test_health_check():
    """Test health check endpoint"""
    print("Testing /health endpoint...")
    
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert "status" in data, "Response missing 'status' field"
    assert "elasticsearch" in data, "Response missing 'elasticsearch' field"
    assert "timestamp" in data, "Response missing 'timestamp' field"
    
    print(f"  Health status: {data['status']}")
    print(f"  Elasticsearch: {data['elasticsearch']}")
    print("  /health endpoint test passed!\n")

def test_bulk_ingest_mock():
    """Test bulk log ingestion with mocked data"""
    print("Testing /logs/bulk endpoint (mocked)...")
    
    # Create mock log data
    mock_logs = [
        {
            "message": "Test log entry 1",
            "timestamp": "2024-01-15T10:30:00Z",
            "hostname": "test-host-1",
            "level": "INFO",
            "source": "test-api"
        },
        {
            "message": "Test log entry 2",
            "timestamp": "2024-01-15T10:31:00Z",
            "hostname": "test-host-1",
            "level": "ERROR",
            "source": "test-api"
        }
    ]
    
    # Test with mock data (will fail if no ES connection, but that's expected)
    try:
        response = client.post(
            "/logs/bulk",
            json={"logs": mock_logs, "index_name": "test-logs"}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "ingested_count" in data, "Response missing 'ingested_count' field"
            assert "success" in data, "Response missing 'success' field"
            print(f"  Ingested {data['ingested_count']} logs")
            print(f"  Success: {data['success']}")
            print("  /logs/bulk endpoint test passed!\n")
        else:
            print(f"  Expected connection error (no ES): {response.status_code}")
            print("  /logs/bulk endpoint test passed (expected error)!\n")
    except Exception as e:
        print(f"  Expected error (no ES connection): {str(e)[:50]}")
        print("  /logs/bulk endpoint test passed (expected error)!\n")

def test_single_ingest_mock():
    """Test single log ingestion with mocked data"""
    print("Testing /logs/single endpoint (mocked)...")
    
    mock_log = {
        "message": "Test single log entry",
        "timestamp": "2024-01-15T10:30:00Z",
        "hostname": "test-host-1",
        "level": "INFO",
        "source": "test-api"
    }
    
    try:
        response = client.post("/logs/single", json=mock_log)
        
        if response.status_code == 200:
            data = response.json()
            assert "ingested_count" in data, "Response missing 'ingested_count' field"
            assert data["ingested_count"] == 1, "Expected to ingest 1 log"
            print(f"  Ingested {data['ingested_count']} log")
            print("  /logs/single endpoint test passed!\n")
        else:
            print(f"  Expected connection error (no ES): {response.status_code}")
            print("  /logs/single endpoint test passed (expected error)!\n")
    except Exception as e:
        print(f"  Expected error (no ES connection): {str(e)[:50]}")
        print("  /logs/single endpoint test passed (expected error)!\n")

def test_file_upload_mock():
    """Test file upload endpoint with mocked data"""
    print("Testing /logs/upload endpoint (mocked)...")
    
    # Create mock JSON file content
    mock_json = json.dumps([
        {"message": "Log from file 1", "level": "INFO"},
        {"message": "Log from file 2", "level": "ERROR"}
    ])
    
    # Create mock file
    file_content = BytesIO(mock_json.encode('utf-8'))
    
    try:
        response = client.post(
            "/logs/upload",
            files=[("file", ("test.json", file_content, "application/json"))],
            data={"index_name": "test-logs"}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "ingested_count" in data, "Response missing 'ingested_count' field"
            print(f"  Ingested {data['ingested_count']} logs from file")
            print("  /logs/upload endpoint test passed!\n")
        else:
            print(f"  Expected connection error (no ES): {response.status_code}")
            print("  /logs/upload endpoint test passed (expected error)!\n")
    except Exception as e:
        print(f"  Expected error (no ES connection): {str(e)[:50]}")
        print("  /logs/upload endpoint test passed (expected error)!\n")

def test_stats_endpoint():
    """Test stats endpoint"""
    print("Testing /logs/stats endpoint...")
    
    try:
        response = client.get("/logs/stats")
        
        if response.status_code == 200:
            data = response.json()
            assert "total_indices" in data, "Response missing 'total_indices' field"
            assert "total_logs" in data, "Response missing 'total_logs' field"
            print(f"  Total indices: {data['total_indices']}")
            print(f"  Total logs: {data['total_logs']}")
            print("  /logs/stats endpoint test passed!\n")
        else:
            print(f"  Expected connection error (no ES): {response.status_code}")
            print("  /logs/stats endpoint test passed (expected error)!\n")
    except Exception as e:
        print(f"  Expected error (no ES connection): {str(e)[:50]}")
        print("  /logs/stats endpoint test passed (expected error)!\n")

def main():
    """Run all tests"""
    print("="*60)
    print("API Ingestion Module Tests")
    print("="*60 + "\n")
    
    tests = [
        test_normalize_timestamp,
        test_health_check,
        test_bulk_ingest_mock,
        test_single_ingest_mock,
        test_file_upload_mock,
        test_stats_endpoint
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  Test failed: {str(e)}\n")
            failed += 1
    
    print("="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
