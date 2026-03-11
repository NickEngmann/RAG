#!/usr/bin/env python3
"""Test the summary endpoint functionality."""

import sys
sys.path.insert(0, '')

from rag_system import generate_summary_stats

def test_generate_summary_stats():
    """Test the summary statistics generation."""
    # Create sample log data
    sample_logs = [
        {
            'id': '1',
            'timestamp': '2024-01-01T10:00:00.000Z',
            'message': 'ERROR: Connection timeout',
            'hostname': 'server1'
        },
        {
            'id': '2',
            'timestamp': '2024-01-01T10:05:00.000Z',
            'message': 'INFO: Request processed',
            'hostname': 'server1'
        },
        {
            'id': '3',
            'timestamp': '2024-01-01T10:10:00.000Z',
            'message': 'ERROR: Connection timeout',
            'hostname': 'server2'
        },
        {
            'id': '4',
            'timestamp': '2024-01-01T10:15:00.000Z',
            'message': 'WARN: High memory usage',
            'hostname': 'server2'
        },
        {
            'id': '5',
            'timestamp': '2024-01-01T10:20:00.000Z',
            'message': 'INFO: Request processed',
            'hostname': 'server1'
        }
    ]
    
    # Test basic summary generation
    result = generate_summary_stats(sample_logs)
    
    # Verify results
    assert result['total_logs'] == 5, f"Expected 5 logs, got {result['total_logs']}"
    assert 'server1' in result['hostnames'], "server1 should be in hostnames"
    assert 'server2' in result['hostnames'], "server2 should be in hostnames"
    assert result['hostnames']['server1'] == 3, "server1 should have 3 logs"
    assert result['hostnames']['server2'] == 2, "server2 should have 2 logs"
    assert result['time_range']['start'] == '2024-01-01T10:00:00.000Z'
    assert result['time_range']['end'] == '2024-01-01T10:20:00.000Z'
    
    # Test with hostname filter
    filtered = generate_summary_stats(sample_logs, hostname_pattern='server1')
    assert filtered['total_logs'] == 3, f"Expected 3 logs for server1, got {filtered['total_logs']}"
    
    # Test with time range filter
    filtered = generate_summary_stats(sample_logs, start_time='2024-01-01T10:05:00.000Z', end_time='2024-01-01T10:15:00.000Z')
    assert filtered['total_logs'] == 3, f"Expected 3 logs in time range, got {filtered['total_logs']}"
    
    print("All summary tests passed!")

if __name__ == '__main__':
    test_generate_summary_stats()
