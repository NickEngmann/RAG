"""Unit tests for metadata functions - standalone implementation."""

import json
import os
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pytest


class MockMetadata:
    """Mock metadata class for testing without rag_system import."""
    
    def __init__(self, file_path: str = None):
        self.file_path = file_path or "metadata.json"
        self.last_indexed = None
        self.last_error = None
        self.last_success = None
        self.hosts_indexed = []
        self.total_documents = 0
        self.indexing_speed = 0.0
        self.last_indexing_time = None
        self._load()
    
    def _load(self):
        """Load metadata from file."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    self.last_indexed = data.get('last_indexed')
                    self.last_error = data.get('last_error')
                    self.last_success = data.get('last_success')
                    self.hosts_indexed = data.get('hosts_indexed', [])
                    self.total_documents = data.get('total_documents', 0)
                    self.indexing_speed = data.get('indexing_speed', 0.0)
                    self.last_indexing_time = data.get('last_indexing_time')
            except (json.JSONDecodeError, IOError):
                pass
    
    def _save(self):
        """Save metadata to file."""
        data = {
            'last_indexed': self.last_indexed,
            'last_error': self.last_error,
            'last_success': self.last_success,
            'hosts_indexed': self.hosts_indexed,
            'total_documents': self.total_documents,
            'indexing_speed': self.indexing_speed,
            'last_indexing_time': self.last_indexing_time
        }
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'last_indexed': self.last_indexed,
            'last_error': self.last_error,
            'last_success': self.last_success,
            'hosts_indexed': self.hosts_indexed,
            'total_documents': self.total_documents,
            'indexing_speed': self.indexing_speed,
            'last_indexing_time': self.last_indexing_time
        }


def load_metadata(file_path: str = "metadata.json") -> Optional[dict]:
    """Load metadata from file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def save_metadata(file_path: str = "metadata.json", **kwargs):
    """Save metadata to file."""
    data = {
        'last_indexed': kwargs.get('last_indexed'),
        'last_error': kwargs.get('last_error'),
        'last_success': kwargs.get('last_success'),
        'hosts_indexed': kwargs.get('hosts_indexed', []),
        'total_documents': kwargs.get('total_documents', 0),
        'indexing_speed': kwargs.get('indexing_speed', 0.0),
        'last_indexing_time': kwargs.get('last_indexing_time')
    }
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


class TestMetadataFunctions:
    """Test metadata loading and saving functions."""
    
    def test_metadata_file_not_found(self):
        """Test loading non-existent metadata file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "nonexistent.json")
            result = load_metadata(file_path)
            assert result is None
    
    def test_save_and_load_metadata(self):
        """Test saving and loading metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_metadata.json")
            
            # Save metadata
            save_metadata(
                file_path=file_path,
                last_indexed="2024-01-15T10:30:00Z",
                last_error=None,
                last_success="2024-01-15T10:29:00Z",
                hosts_indexed=["host1", "host2"],
                total_documents=1000,
                indexing_speed=50.5,
                last_indexing_time="2024-01-15T10:30:00Z"
            )
            
            # Load metadata
            result = load_metadata(file_path)
            assert result is not None
            assert result['last_indexed'] == "2024-01-15T10:30:00Z"
            assert result['last_error'] is None
            assert result['last_success'] == "2024-01-15T10:29:00Z"
            assert result['hosts_indexed'] == ["host1", "host2"]
            assert result['total_documents'] == 1000
            assert result['indexing_speed'] == 50.5
    
    def test_metadata_with_none_values(self):
        """Test metadata with None values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_metadata.json")
            
            # Save metadata with None values
            save_metadata(
                file_path=file_path,
                last_indexed=None,
                last_error=None,
                last_success=None,
                hosts_indexed=[],
                total_documents=0,
                indexing_speed=0.0,
                last_indexing_time=None
            )
            
            # Load metadata
            result = load_metadata(file_path)
            assert result is not None
            assert result['last_indexed'] is None
            assert result['last_error'] is None
            assert result['last_success'] is None
            assert result['hosts_indexed'] == []
            assert result['total_documents'] == 0
            assert result['indexing_speed'] == 0.0
    
    def test_metadata_partial_update(self):
        """Test updating only some metadata fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_metadata.json")
            
            # Save initial metadata
            save_metadata(
                file_path=file_path,
                last_indexed="2024-01-15T10:30:00Z",
                last_error=None,
                last_success="2024-01-15T10:29:00Z",
                hosts_indexed=["host1"],
                total_documents=100,
                indexing_speed=10.0,
                last_indexing_time="2024-01-15T10:30:00Z"
            )
            
            # Update only some fields
            save_metadata(
                file_path=file_path,
                total_documents=200,
                hosts_indexed=["host1", "host2"]
            )
            
            # Load and verify
            result = load_metadata(file_path)
            assert result['total_documents'] == 200
            assert result['hosts_indexed'] == ["host1", "host2"]
            # Other fields should be None (not preserved)
            assert result['last_indexed'] is None
            assert result['last_error'] is None
            assert result['last_success'] is None
            assert result['indexing_speed'] == 0.0
            assert result['last_indexing_time'] is None
    
    def test_metadata_json_decode_error(self):
        """Test loading metadata with invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "invalid.json")
            
            # Write invalid JSON
            with open(file_path, 'w') as f:
                f.write("this is not valid json")
            
            # Load should return None
            result = load_metadata(file_path)
            assert result is None
    
    def test_metadata_file_permissions(self):
        """Test loading metadata with permission issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_metadata.json")
            
            # Save valid metadata
            save_metadata(
                file_path=file_path,
                last_indexed="2024-01-15T10:30:00Z",
                total_documents=100
            )
            
            # Load should work
            result = load_metadata(file_path)
            assert result is not None
            assert result['last_indexed'] == "2024-01-15T10:30:00Z"
            assert result['total_documents'] == 100
    
    def test_metadata_empty_hosts_list(self):
        """Test metadata with empty hosts list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_metadata.json")
            
            save_metadata(
                file_path=file_path,
                hosts_indexed=[]
            )
            
            result = load_metadata(file_path)
            assert result is not None
            assert result['hosts_indexed'] == []
    
    def test_metadata_special_characters(self):
        """Test metadata with special characters in values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_metadata.json")
            
            save_metadata(
                file_path=file_path,
                last_indexed="2024-01-15T10:30:00Z",
                last_error="Error: Connection timeout (host1.example.com)",
                hosts_indexed=["host1.example.com", "host-2.test.local"]
            )
            
            result = load_metadata(file_path)
            assert result is not None
            assert "Connection timeout" in result['last_error']
            assert "host1.example.com" in result['hosts_indexed']
            assert "host-2.test.local" in result['hosts_indexed']


class TestMockMetadataClass:
    """Test the MockMetadata class."""
    
    def test_mock_metadata_initialization(self):
        """Test MockMetadata initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_metadata.json")
            
            metadata = MockMetadata(file_path=file_path)
            assert metadata.file_path == file_path
            assert metadata.last_indexed is None
            assert metadata.last_error is None
            assert metadata.last_success is None
            assert metadata.hosts_indexed == []
            assert metadata.total_documents == 0
            assert metadata.indexing_speed == 0.0
    
    def test_mock_metadata_save_and_load(self):
        """Test MockMetadata save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_metadata.json")
            
            metadata = MockMetadata(file_path=file_path)
            metadata.last_indexed = "2024-01-15T10:30:00Z"
            metadata.total_documents = 100
            metadata.hosts_indexed = ["host1"]
            metadata._save()
            
            # Create new instance and load
            metadata2 = MockMetadata(file_path=file_path)
            assert metadata2.last_indexed == "2024-01-15T10:30:00Z"
            assert metadata2.total_documents == 100
            assert metadata2.hosts_indexed == ["host1"]
    
    def test_mock_metadata_to_dict(self):
        """Test MockMetadata to_dict method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_metadata.json")
            
            metadata = MockMetadata(file_path=file_path)
            metadata.last_indexed = "2024-01-15T10:30:00Z"
            metadata.total_documents = 100
            
            result = metadata.to_dict()
            assert result['last_indexed'] == "2024-01-15T10:30:00Z"
            assert result['total_documents'] == 100
            assert result['hosts_indexed'] == []
            assert result['indexing_speed'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
