#!/usr/bin/env python
"""Test imports for the RAG system"""

import sys
import os

# Mock sentence_transformers before importing rag_system
from unittest.mock import MagicMock
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['sentence_transformers.SentenceTransformer'] = MagicMock()
sys.modules['sentence_transformers.util'] = MagicMock()
sys.modules['sentence_transformers.util.SentenceTransformer'] = MagicMock()
sys.modules['sentence_transformers.util.SentenceTransformer'] = MagicMock()
sys.modules['sentence_transformers.util.SentenceTransformer'] = MagicMock()
sys.modules['sentence_transformers.util.SentenceTransformer'] = MagicMock()

print("Testing imports...")

try:
    import logging
    import json
    import time
    import hashlib
    import re
    import numpy as np
    from datetime import datetime
    from typing import Dict, List, Optional, Set, Any, Tuple
    from collections import defaultdict
    from sklearn.preprocessing import MinMaxScaler
    import openai
    import fnmatch
    import gc
    import traceback
    from tqdm import tqdm
    import elasticsearch
    from elasticsearch import Elasticsearch
    import torch
    from torch import tensor, FloatTensor
    print("✓ All basic imports successful")
except Exception as e:
    print(f"✗ Import failed: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

try:
    from rag_system import (
        preprocess_log,
        load_metadata,
        save_metadata,
        TimeScaler,
        RAGSystem,
        ElasticsearchConnector,
        SentenceTransformerEmbedder,
        MetadataManager,
        LogProcessor,
        EmbeddingGenerator,
        TimeNormalizer
    )
    print("✓ All rag_system imports successful")
except Exception as e:
    print(f"✗ rag_system import failed: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

print("\nAll imports successful!")
