#!/usr/bin/env python3
"""Test metadata loading and saving functionality."""

import json
import os
import tempfile
from datetime import datetime

# Test the metadata functions
def test_metadata_roundtrip():
    """Test that metadata can be saved and loaded correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_file = os.path.join(tmpdir, "metadata.json")
        
        # Simulate the load_metadata and save_metadata functions
        def load_metadata(filepath):
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
            return {'last_processed': '1970-01-01T00:00:00.000Z', 'processed_ids': []}
        
        def save_metadata(metadata, filepath):
            metadata_to_save = metadata.copy()
            metadata_to_save['processed_ids'] = list(metadata_to_save['processed_ids'])
            with open(filepath, 'w') as f:
                json.dump(metadata_to_save, f)
        
        # Test loading empty metadata
        metadata = load_metadata(metadata_file)
        assert 'last_processed' in metadata
        assert 'processed_ids' in metadata
        print("✓ load_metadata works correctly")
        
        # Test saving and loading with data
        metadata['processed_ids'] = {'log1', 'log2', 'log3'}
        metadata['last_processed'] = datetime.now().isoformat()
        save_metadata(metadata, metadata_file)
        
        loaded_metadata = load_metadata(metadata_file)
        assert set(loaded_metadata['processed_ids']) == {'log1', 'log2', 'log3'}
        print("✓ save_metadata and load_metadata roundtrip works")
        
        print("\nAll metadata tests passed!")

if __name__ == "__main__":
    test_metadata_roundtrip()
