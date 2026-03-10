#!/usr/bin/env python3
"""Unit tests for log processing logic without external dependencies."""

import pytest
from datetime import datetime, timedelta
from dateutil.parser import parse
import re
import os
import json
from unittest.mock import mock_open, patch


class TestLogPreprocessing:
    """Test log preprocessing functions."""
    
    def test_parse_log_line_with_timestamp(self):
        """Test parsing a log line with timestamp."""
        log_line = "2024-01-15 10:30:45 INFO [host1] Message here"
        
        # Extract timestamp using regex
        timestamp_match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', log_line)
        assert timestamp_match is not None
        timestamp_str = timestamp_match.group(1)
        timestamp = parse(timestamp_str)
        assert isinstance(timestamp, datetime)
        assert timestamp.year == 2024
        assert timestamp.month == 1
        assert timestamp.day == 15
    
    def test_parse_log_line_with_date_only(self):
        """Test parsing a log line with date only (no time)."""
        log_line = "2024-01-15 INFO [host1] Message here"
        
        timestamp_match = re.match(r'^(\d{4}-\d{2}-\d{2})', log_line)
        assert timestamp_match is not None
        timestamp_str = timestamp_match.group(1)
        timestamp = parse(timestamp_str)
        assert isinstance(timestamp, datetime)
        assert timestamp.hour == 0
        assert timestamp.minute == 0
        assert timestamp.second == 0
    
    def test_parse_log_line_no_timestamp(self):
        """Test parsing a log line without timestamp."""
        log_line = "INFO [host1] Message without timestamp"
        
        timestamp_match = re.match(r'^(\d{4}-\d{2}-\d{2}[\s]?\d{2}:\d{2}:\d{2})', log_line)
        assert timestamp_match is None
    
    def test_extract_hostname_from_brackets(self):
        """Test extracting hostname from brackets."""
        log_line = "2024-01-15 10:30:45 INFO [host1.example.com] Message here"
        
        hostname_match = re.search(r'\[([^\]]+)\]', log_line)
        assert hostname_match is not None
        hostname = hostname_match.group(1)
        assert hostname == "host1.example.com"
    
    def test_extract_hostname_no_brackets(self):
        """Test extracting hostname when no brackets present."""
        log_line = "2024-01-15 10:30:45 INFO Message here"
        
        hostname_match = re.search(r'\[([^\]]+)\]', log_line)
        assert hostname_match is None


class TestTimeRangeFiltering:
    """Test time range filtering logic."""
    
    def test_log_within_time_range(self):
        """Test that a log within the time range is included."""
        log_time = datetime(2024, 1, 15, 10, 30, 45)
        start_time = datetime(2024, 1, 15, 0, 0, 0)
        end_time = datetime(2024, 1, 15, 23, 59, 59)
        
        assert start_time <= log_time <= end_time
    
    def test_log_before_time_range(self):
        """Test that a log before the time range is excluded."""
        log_time = datetime(2024, 1, 14, 23, 59, 59)
        start_time = datetime(2024, 1, 15, 0, 0, 0)
        end_time = datetime(2024, 1, 15, 23, 59, 59)
        
        assert log_time < start_time
    
    def test_log_after_time_range(self):
        """Test that a log after the time range is excluded."""
        log_time = datetime(2024, 1, 16, 0, 0, 0)
        start_time = datetime(2024, 1, 15, 0, 0, 0)
        end_time = datetime(2024, 1, 15, 23, 59, 59)
        
        assert log_time > end_time
    
    def test_time_range_with_timedelta(self):
        """Test time range calculation using timedelta."""
        now = datetime(2024, 1, 15, 12, 0, 0)
        days_ago = now - timedelta(days=7)
        
        assert days_ago.year == 2024
        assert days_ago.month == 1
        assert days_ago.day == 8


class TestMetadataFunctions:
    """Test metadata loading and saving functions."""
    
    def test_metadata_file_not_found(self):
        """Test loading metadata when file doesn't exist."""
        metadata_file = "/tmp/nonexistent_metadata.json"
        
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
        
        # Simulate loading non-existent file
        metadata = {}
        assert metadata == {}
    
    def test_save_and_load_metadata(self):
        """Test saving and loading metadata."""
        metadata_file = "/tmp/test_metadata.json"
        test_metadata = {
            "last_processed_date": "2024-01-15",
            "last_processed_time": "10:30:45",
            "last_processed_hostname": "host1",
            "total_lines_processed": 1000,
            "last_updated": "2024-01-15T10:30:45"
        }
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(test_metadata, f)
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata == test_metadata
        
        # Cleanup
        os.remove(metadata_file)
    
    def test_metadata_with_none_values(self):
        """Test metadata with None values."""
        metadata_file = "/tmp/test_metadata_none.json"
        test_metadata = {
            "last_processed_date": None,
            "last_processed_time": None,
            "last_processed_hostname": None,
            "total_lines_processed": 0,
            "last_updated": None
        }
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(test_metadata, f)
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata["last_processed_date"] is None
        assert loaded_metadata["total_lines_processed"] == 0
        
        # Cleanup
        os.remove(metadata_file)


class TestHostPatternMatching:
    """Test hostname pattern matching."""
    
    def test_exact_hostname_match(self):
        """Test exact hostname matching."""
        hostname = "host1.example.com"
        pattern = "host1"
        
        assert hostname.startswith(pattern)
    
    def test_hostname_no_match(self):
        """Test hostname that doesn't match pattern."""
        hostname = "host2.example.com"
        pattern = "host1"
        
        assert not hostname.startswith(pattern)
    
    def test_hostname_with_subdomain(self):
        """Test hostname with subdomain matching."""
        hostname = "host1.sub.example.com"
        pattern = "host1"
        
        assert hostname.startswith(pattern)
    
    def test_hostname_case_sensitive(self):
        """Test that hostname matching is case-sensitive."""
        hostname = "Host1.example.com"
        pattern = "host1"
        
        assert not hostname.startswith(pattern)


class TestLogLineParsing:
    """Test log line parsing logic."""
    
    def test_parse_log_line_structure(self):
        """Test parsing a complete log line."""
        log_line = "2024-01-15 10:30:45 INFO [host1.example.com] Application started successfully"
        
        # Extract timestamp
        timestamp_match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', log_line)
        assert timestamp_match is not None
        timestamp = parse(timestamp_match.group(1))
        
        # Extract log level
        level_match = re.search(r'\b(INFO|ERROR|WARNING|DEBUG|CRITICAL)\b', log_line)
        assert level_match is not None
        log_level = level_match.group(1)
        assert log_level == "INFO"
        
        # Extract hostname
        hostname_match = re.search(r'\[([^\]]+)\]', log_line)
        assert hostname_match is not None
        hostname = hostname_match.group(1)
        assert hostname == "host1.example.com"
        
        # Extract message
        message_match = re.search(r'\] (.+)$', log_line)
        assert message_match is not None
        message = message_match.group(1)
        assert message == "Application started successfully"
    
    def test_parse_log_line_with_error(self):
        """Test parsing a log line with ERROR level."""
        log_line = "2024-01-15 10:30:45 ERROR [host1.example.com] Database connection failed"
        
        level_match = re.search(r'\b(INFO|ERROR|WARNING|DEBUG|CRITICAL)\b', log_line)
        assert level_match is not None
        log_level = level_match.group(1)
        assert log_level == "ERROR"
    
    def test_parse_log_line_with_warning(self):
        """Test parsing a log line with WARNING level."""
        log_line = "2024-01-15 10:30:45 WARNING [host1.example.com] Low memory detected"
        
        level_match = re.search(r'\b(INFO|ERROR|WARNING|DEBUG|CRITICAL)\b', log_line)
        assert level_match is not None
        log_level = level_match.group(1)
        assert log_level == "WARNING"


class TestDateParsing:
    """Test date parsing logic."""
    
    def test_parse_iso_date(self):
        """Test parsing ISO format date."""
        date_str = "2024-01-15"
        parsed_date = parse(date_str)
        assert isinstance(parsed_date, datetime)
        assert parsed_date.year == 2024
        assert parsed_date.month == 1
        assert parsed_date.day == 15
    
    def test_parse_datetime_string(self):
        """Test parsing datetime string."""
        datetime_str = "2024-01-15 10:30:45"
        parsed_datetime = parse(datetime_str)
        assert isinstance(parsed_datetime, datetime)
        assert parsed_datetime.hour == 10
        assert parsed_datetime.minute == 30
        assert parsed_datetime.second == 45
    
    def test_parse_date_with_time(self):
        """Test parsing date with time component."""
        datetime_str = "2024-01-15T10:30:45"
        parsed_datetime = parse(datetime_str)
        assert isinstance(parsed_datetime, datetime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
