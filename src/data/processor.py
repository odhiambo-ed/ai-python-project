import json
from typing import Generator, Any, Dict, List
from pathlib import Path

def read_jsonl(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """Read a JSONL file one line at a time. Memory-efficient."""
    with open(file_path, "r") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_number}: {e}")
                continue

def filter_high_quality(records: Generator[Dict[str, Any], None, None], 
                        threshold: float = 0.8) -> Generator[Dict[str, Any], None, None]:
    """Filter records by quality score."""
    for record in records:
        if record.get("quality_score", 0) > threshold:
            yield record

def count_records(records: Generator[Dict[str, Any], None, None]) -> int:
    """Count total records in generator."""
    return sum(1 for _ in records)