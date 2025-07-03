#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Granular Evaluation scripts for Date of Birth (DoB) property linking.

This scripts performs evaluation of DoB linking at different granularity levels:
- Exact match (year, month, day)
- Year match
- Decade match (e.g. 1980s)
- Century match (e.g. 19th century)

The scripts:
- Processes DoB CSV files with linked entities
- Calculates metrics (precision, recall, F1) at each granularity level
- Generates detailed reports per model/property combination
- Creates comprehensive LaTeX-compatible tables for results
- Supports Boyer-Moore voting algorithms and per-model analysis
"""

import os
import sys
import csv
import argparse
import logging
import glob
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional, Set, Union
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_INPUT_DIR = ""

# Voting algorithm directories (enhanced to include Boyer-Moore per-model)
VOTING_DIRS = [
    "",
    "",
    "",
    ""
]

# Define property-specific folders within granular_analyses
PROPERTY_NAME = "DoB"
GRANULAR_BASE_DIR = ""
DEFAULT_OUTPUT_DIR = f"{GRANULAR_BASE_DIR}/{PROPERTY_NAME}"

# RAG directories
DEFAULT_RAG_NIL_DIR = f"{DEFAULT_OUTPUT_DIR}/RAG/nil"
DEFAULT_RAG_QID_DIR = f"{DEFAULT_OUTPUT_DIR}/RAG/qid"

# ZS directories
DEFAULT_ZS_NIL_DIR = f"{DEFAULT_OUTPUT_DIR}/ZS/nil"
DEFAULT_ZS_QID_DIR = f"{DEFAULT_OUTPUT_DIR}/ZS/qid"

# Report and table directories
DEFAULT_REPORT_DIR = f"{GRANULAR_BASE_DIR}/{PROPERTY_NAME}/reports"
DEFAULT_LATEX_DIR = f"{GRANULAR_BASE_DIR}/{PROPERTY_NAME}/latex_tables"

# Create directories if they don't exist
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
os.makedirs(DEFAULT_RAG_NIL_DIR, exist_ok=True)
os.makedirs(DEFAULT_RAG_QID_DIR, exist_ok=True)
os.makedirs(DEFAULT_ZS_NIL_DIR, exist_ok=True)
os.makedirs(DEFAULT_ZS_QID_DIR, exist_ok=True)
os.makedirs(DEFAULT_REPORT_DIR, exist_ok=True)
os.makedirs(DEFAULT_LATEX_DIR, exist_ok=True)

# Date format patterns
DATE_FORMATS = [
    r'(\d{4})-(\d{2})-(\d{2})T\d{2}:\d{2}:\d{2}Z',  # ISO date with time: 1728-05-11T00:00:00Z
    r'(\d{4})-(\d{2})-(\d{2})',                     # ISO date: 1728-05-11
    r'(\d{4})\/(\d{1,2})\/(\d{1,2})',               # Slash date: 1728/05/11
    r'\b(1\d{3}|20\d{2}|21\d{2})\b'                 # Year only: 1728, 2023, 2100
]

# Additional date patterns
DATE_TEXT_PATTERNS = [
    r'born (?:on )?(?:the )?(\d{1,2})(?:st|nd|rd|th)? (?:of )?([A-Za-z]+),? (\d{4})',  # born on the 3rd of March, 1890
    r'born (?:in )?([A-Za-z]+) (?:of )?(\d{4})',  # born in March of 1890
    r'born (?:in )?(\d{4})',  # born in 1890
    r'(\d{1,2}) ([A-Za-z]+) (\d{4})',  # 3 March 1890
]

# Month name mapping
MONTH_NAMES = {
    'january': 1, 'jan': 1,
    'february': 2, 'feb': 2,
    'march': 3, 'mar': 3,
    'april': 4, 'apr': 4,
    'may': 5,
    'june': 6, 'jun': 6,
    'july': 7, 'jul': 7,
    'august': 8, 'aug': 8,
    'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10,
    'november': 11, 'nov': 11,
    'december': 12, 'dec': 12
}

# Enhanced model mappings to standardize names (optimized for LaTeX tables)
MODEL_MAPPINGS = {
    'google_gemma-2-27b-it': 'Gemma',
    'meta-llama_llama-3.3-70b-instruct': 'LLAMA',
    'microsoft_phi-3-medium-128k-instruct': 'Phi-3',
    'mistralai_mixtral-8x7b-instruct': 'Mixtral',
    'openai_gpt-4o-mini': 'GPT-4o',
    'qwen_qwen-2.5-72b-instruct': 'Qwen',
    # Boyer-Moore per-model specific mappings (with corrected parsing)
    'meta-llama_llama.3.3-70b-instruct': 'LLAMA',
    'qwen_qwen.2.5-72b-instruct': 'Qwen',
    # Voting algorithm names (consistent with main evaluation scripts)
    'majority_voting': 'Majority',
    'boyer_moore_voting': 'Boyer-M',
    'flexible_boyer_voting': 'Flexible-Boyer',
    'adaptive_threshold_voting': 'Adaptive',
    'boyer-moore': 'Boyer-M',
    'adaptive_threshold': 'Adaptive',
    'boyer_moore': 'Boyer-M',
    'flexible_boyer': 'Flexible-B',
    'flexible_boyer_moore_voting': 'Flexible-Boyer',
    'boyer_moore_per_model': 'Boyer-M',
    # Combined results
    'combined_adaptive_threshold': 'Adaptive',
    'combined_boyer_moore': 'Boyer-M',
    'combined_flexible_boyer': 'Flexible-B',
    'combined_adaptive_threshold_voting': 'Adaptive',
    'combined_boyer_moore_voting': 'Boyer-M',
    'combined_flexible_boyer_moore_voting': 'Flexible-B'
}

# Enhanced retriever formatting and ordering (consistent with main evaluation scripts)
RETRIEVER_MAPPINGS = {
    'none': '-',  # Zero-shot (changed from 'ZS' to '-' to avoid overflow issues)
    'bge-large': 'BGE',
    'bm25': 'BM25',
    'instructor-xl': 'Inst',   # Shortened from 'Inst-X'
    'contriever': 'Contr',
    'gtr-xl': 'GTR-X',
    'gtr-large': 'GTR-L',
    # Voting algorithms - consistent naming
    'adaptive_threshold': 'Adaptive',
    'boyer_moore': 'Boyer-M',
    'flexible_boyer': 'Flexible-B',
    'adaptive_threshold_voting': 'Adaptive',
    'boyer_moore_voting': 'Boyer-M',
    'flexible_boyer_moore_voting': 'Flexible-B',
    'boyer_moore_per_model': 'Boyer-M'
}

RETRIEVER_ORDERING = {
    'boyer_moore': -2,  # Boyer-Moore should come first (consistent with main scripts)
    'boyer-moore': -2,
    'boyer_moore_per_model': -2,
    'none': 0,  # Zero-shot (no retriever)
    'adaptive_threshold': 1,
    'flexible_boyer': 2,
    'flexible_boyer_moore': 2,
    'adaptive_threshold_voting': 3,
    'boyer_moore_voting': 4,
    'flexible_boyer_moore_voting': 5,
    'bge-large': 10,
    'bm25': 11,
    'instructor-xl': 12,
    'contriever': 13,
    'gtr-xl': 14,
    'gtr-large': 15
}

# Improved date extraction functions
def normalize_date(date_str: str) -> Optional[Tuple[int, int, int]]:
    """
    Normalize a date string to a standardized (year, month, day) tuple.
    This handles various date formats and extracts the components.
    
    Args:
        date_str (str): A string containing a date
        
    Returns:
        Optional[Tuple[int, int, int]]: (year, month, day), or None if not a valid date
    """
    if not date_str or not isinstance(date_str, str):
        return None
    
    # Remove any time zone indicator and whitespace
    date_str = date_str.strip()
    
    # Try to extract full dates from structured formats
    for pattern in DATE_FORMATS[:3]:  # Only use the patterns with full date components
        match = re.search(pattern, date_str)
        if match:
            try:
                year = int(match.group(1))
                month = int(match.group(2))
                day = int(match.group(3))
                
                # Validate components (simple validation)
                # Using a broader range for years to handle future dates or potential errors
                if 1000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                    return (year, month, day)
            except (ValueError, IndexError):
                continue
    
    # Try to extract dates from text patterns
    for i, pattern in enumerate(DATE_TEXT_PATTERNS):
        match = re.search(pattern, date_str, re.IGNORECASE)
        if match:
            try:
                if i == 0:  # born on the 3rd of March, 1890
                    day = int(match.group(1))
                    month_name = match.group(2).lower()
                    year = int(match.group(3))
                    
                    if month_name in MONTH_NAMES:
                        month = MONTH_NAMES[month_name]
                        return (year, month, day)
                elif i == 1:  # born in March of 1890
                    month_name = match.group(1).lower()
                    year = int(match.group(2))
                    
                    if month_name in MONTH_NAMES:
                        month = MONTH_NAMES[month_name]
                        return (year, month, 1)  # Default to 1st of the month
                elif i == 2:  # born in 1890
                    year = int(match.group(1))
                    return (year, 1, 1)  # Default to January 1st
                elif i == 3:  # 3 March 1890
                    day = int(match.group(1))
                    month_name = match.group(2).lower()
                    year = int(match.group(3))
                    
                    if month_name in MONTH_NAMES:
                        month = MONTH_NAMES[month_name]
                        return (year, month, day)
            except (ValueError, IndexError):
                continue
    
    # Try to extract just the year
    year_match = re.search(r'\b(1\d{3}|20\d{2}|21\d{2})\b', date_str)
    if year_match:
        try:
            year = int(year_match.group(1))
            # Using broader validation range to accommodate future dates and various date formats
            if 1000 <= year <= 2100:
                return (year, 1, 1)  # Default to January 1st when only year is known
        except (ValueError, IndexError):
            pass
            
    return None

def compare_dates_exact(date1: str, date2: str) -> bool:
    """
    Compare two date strings for exact match (same year, month, and day).
    Handles different formatting (ISO with/without time, different separators, etc.)
    
    Args:
        date1 (str): First date string
        date2 (str): Second date string
        
    Returns:
        bool: True if dates match exactly, False otherwise
    """
    norm1 = normalize_date(date1)
    norm2 = normalize_date(date2)
    
    if norm1 is None or norm2 is None:
        return False
    
    # For exact match, all components must match
    return norm1 == norm2

def extract_date_components(date_str: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Extract year, month, and day from a date string in various formats.
    
    Args:
        date_str (str): A string containing a date
        
    Returns:
        Tuple[Optional[int], Optional[int], Optional[int]]: (year, month, day)
    """
    normalized = normalize_date(date_str)
    if normalized:
        return normalized
    
    return None, None, None

def extract_year(date_str: str) -> Optional[int]:
    """
    Extract the year component from a date string.
    
    Args:
        date_str (str): A string containing a date
        
    Returns:
        Optional[int]: The extracted year as an integer, or None if no year could be extracted
    """
    year, _, _ = extract_date_components(date_str)
    return year

def extract_decade(date_str: str) -> Optional[int]:
    """
    Extract the decade from a date string.
    
    Args:
        date_str (str): A string containing a date
        
    Returns:
        Optional[int]: The decade (e.g., 1980 for the 1980s), or None if no decade could be extracted
    """
    year = extract_year(date_str)
    if year is not None:
        return (year // 10) * 10
    
    # Try to extract decade directly (e.g., "1980s")
    decade_match = re.search(r'\b(1\d{3}|20\d{2})s\b', date_str)
    if decade_match:
        try:
            return int(decade_match.group(1))
        except (ValueError, IndexError):
            pass
            
    return None

def extract_century(date_str: str) -> Optional[int]:
    """
    Extract the century from a date string.
    
    Args:
        date_str (str): A string containing a date
        
    Returns:
        Optional[int]: The century (e.g., 19 for the 19th century), or None if no century could be extracted
    """
    year = extract_year(date_str)
    if year is not None:
        return (year - 1) // 100 + 1
    
    # Try to extract century directly (e.g., "19th century")
    century_match = re.search(r'\b(\d{1,2})(st|nd|rd|th)\s+century\b', date_str, re.IGNORECASE)
    if century_match:
        try:
            return int(century_match.group(1))
        except (ValueError, IndexError):
            pass
            
    return None

class GranularEvaluationResult:
    """Class to store evaluation results for a single file with different granularities."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.property_type = 'DoB'  # Always DoB for this scripts
        self.approach = self._extract_approach()
        self.retriever = self._extract_retriever()
        self.model = self._extract_model()
        self.entity_type = self._extract_entity_type()
        
        # Results for each granularity level
        self.granularities = ['exact', 'year', 'decade', 'century']
        self.results = {
            granularity: {
                'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0
            }
            for granularity in self.granularities
        }
            
    def _extract_approach(self) -> str:
        """Extract approach (RAG or ZS) from path or filename."""
        if "RAG" in self.file_path or "rag" in self.file_path:
            return "RAG"
        elif "ZS" in self.file_path or "zs" in self.file_path:
            return "ZS"
        elif "noctx" in self.file_path.lower():  # Zero-shot files often have 'noctx' in the name
            return "ZS"
        elif "answers" in self.file_path:
            return "RAG"
        elif "ZS_answers" in self.file_path:
            return "ZS"
        return "unknown"
    
    def _extract_retriever(self) -> str:
        """Extract retriever from filename (enhanced to support Boyer-Moore per-model)."""
        # Check for per-model Boyer-Moore results first
        if "boyer_moore_per_model" in self.file_path:
            return "boyer_moore"
            
        # Check for other voting algorithm files
        if "boyer_moore" in self.file_path or "boyer-moore" in self.file_path:
            return "boyer_moore"
        if "adaptive_threshold" in self.file_path:
            return "adaptive_threshold"
        if "flexible_boyer" in self.file_path:
            return "flexible_boyer"
            
        # Standard retrievers
        retrievers = ["bge-large", "bm25", "contriever", "gtr-large", "gtr-xl", "instructor-xl"]
        for retriever in retrievers:
            if retriever in self.file_path:
                return retriever
        
        return "none"
    
    def _extract_model(self) -> str:
        """Extract model from filename (enhanced to support Boyer-Moore per-model)."""
        # Check for voting algorithm files
        filename = os.path.basename(self.file_path)
        
        # For per-model Boyer-Moore results, extract the model name
        if "boyer_moore_per_model" in filename:
            # Extract the model name that comes after openrouter_ and before _boyer_moore_per_model
            # Updated pattern to handle hyphens in model names like llama-3-3 and qwen-2-5
            pattern = r"openrouter_([a-zA-Z0-9\-]+_[a-zA-Z0-9\.\-]+)_boyer_moore"
            match = re.search(pattern, filename)
            if match:
                model_name = match.group(1)
                # Convert hyphens back to dots for consistency with other model names
                model_name = model_name.replace('-3-3-', '.3.3-').replace('-2-5-', '.2.5-')
                return model_name
        
        # Standard voting algorithm detection
        if "majority_voting" in filename or "majvote" in filename:
            return "majority_voting"
        if "adaptive_threshold" in filename:
            return "adaptive_threshold_voting"
        if "flexible_boyer" in filename:
            return "flexible_boyer_voting"
        if "boyer_moore" in filename or "boyer-moore" in filename:
            return "boyer_moore_voting"
        
        # Pattern to match model name: openrouter_vendor_model
        pattern = r"openrouter_([a-zA-Z0-9\-]+)_([a-zA-Z0-9\.\-]+)"
        match = re.search(pattern, filename)
        if match:
            vendor = match.group(1)
            model = match.group(2)
            return f"{vendor}_{model}"
        
        # For files with 'combined' in the name, extract the voting algorithm
        if "combined" in self.file_path:
            if "adaptive_threshold" in self.file_path:
                return "combined_adaptive_threshold_voting"
            elif "flexible_boyer_moore" in self.file_path:
                return "combined_flexible_boyer_moore_voting"
            elif "boyer_moore" in self.file_path:
                return "combined_boyer_moore_voting"
        
        return "unknown"
    
    def _extract_entity_type(self) -> str:
        """Extract entity type (NIL or QID) from filename."""
        filename = self.file_path.lower()
        entity_type = "unknown"
        if "nil" in filename:
            entity_type = "NIL"
        elif "qid" in filename:
            entity_type = "QID"
        # Add debug logging
        logger.debug(f"Extracted entity type: {entity_type} from file: {os.path.basename(self.file_path)}")
        return entity_type
    
    def calculate_metrics(self, granularity: str) -> None:
        """Calculate precision, recall, and F1 score for a specific granularity."""
        try:
            tp = self.results[granularity]['tp']
            fp = self.results[granularity]['fp']
            fn = self.results[granularity]['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            self.results[granularity]['precision'] = precision
            self.results[granularity]['recall'] = recall
            self.results[granularity]['f1'] = f1
        except Exception as e:
            logger.warning(f"Error calculating metrics for {granularity} in {self.file_path}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for reporting."""
        result_dict = {
            "file": os.path.basename(self.file_path),
            "property": self.property_type,
            "approach": self.approach,
            "retriever": self.retriever,
            "model": self.model,
            "entity_type": self.entity_type,
        }
        
        # Add metrics for each granularity
        for granularity in self.granularities:
            for metric, value in self.results[granularity].items():
                result_dict[f"{granularity}_{metric}"] = value
                
        return result_dict


def find_dob_files(input_dir: str, include_files: List[str] = None, skip_files: List[str] = None) -> List[str]:
    """Find DoB CSV files in the given directories."""
    all_files = []
    
    # If input_dir is a string, convert to list
    if isinstance(input_dir, str):
        input_dirs = [input_dir]
    else:
        input_dirs = input_dir
        
    for curr_input_dir in input_dirs:
        # Find all CSV files recursively
        csv_files = glob.glob(f"{curr_input_dir}/**/*.csv", recursive=True)
        logger.info(f"Found {len(csv_files)} total CSV files in {curr_input_dir}")
        
        # Filter for DoB files specifically
        dob_files = [f for f in csv_files if "dob" in f.lower() or "date_of_birth" in f.lower()]
        logger.info(f"Filtered to {len(dob_files)} DoB files")
        
        # Apply additional filters
        if include_files:
            filtered_files = []
            for file in dob_files:
                if all(pattern in file for pattern in include_files):
                    filtered_files.append(file)
            dob_files = filtered_files
            logger.info(f"After include_files filter: {len(dob_files)} files")

        if skip_files:
            filtered_files = []
            for file in dob_files:
                if not any(pattern in file for pattern in skip_files):
                    filtered_files.append(file)
            dob_files = filtered_files
            logger.info(f"After skip_files filter: {len(dob_files)} files")
        
        # Count files by entity type for debugging
        qid_files = [f for f in dob_files if "qid" in f.lower()]
        nil_files = [f for f in dob_files if "nil" in f.lower()]
        unknown_files = [f for f in dob_files if "qid" not in f.lower() and "nil" not in f.lower()]
        
        logger.info(f"Files breakdown in {curr_input_dir}: {len(nil_files)} NIL files, {len(qid_files)} QID files, {len(unknown_files)} unknown")
        
        all_files.extend(dob_files)
        
    return all_files


def evaluate_file_granular(file_path: str, out_dir: str, verbose: bool = False) -> GranularEvaluationResult:
    """Evaluate a single CSV file at different granularity levels."""
    result = GranularEvaluationResult(file_path)
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, delimiter=',', low_memory=False)
        
        # Add evaluation columns for each granularity if they don't exist
        for granularity in result.granularities:
            if f'{granularity}_TP' not in df.columns:
                df[f'{granularity}_TP'] = 0
            if f'{granularity}_FP' not in df.columns:
                df[f'{granularity}_FP'] = 0
            if f'{granularity}_FN' not in df.columns:
                df[f'{granularity}_FN'] = 0
            if f'{granularity}_TN' not in df.columns:
                df[f'{granularity}_TN'] = 0
        
        # Extract entity type from filename (authoritative)
        entity_type_from_filename = "unknown"
        if "qid" in file_path.lower():
            entity_type_from_filename = "QID"
        elif "nil" in file_path.lower():
            entity_type_from_filename = "NIL"
            
        # Update result's entity_type to match filename
        result.entity_type = entity_type_from_filename
        
        # Set entity_type column in dataframe
        df['entity_type'] = entity_type_from_filename
        
        # Normalize column names
        # Check for common variations of expected columns
        linked_qid_candidates = ['linked_qid', 'prediction', 'llm_answer']
        gold_qid_candidates = ['qid_gold_true', 'span_gold_qid', 'gold_answer']
        
        linked_qid_col = None
        for col in linked_qid_candidates:
            if col in df.columns:
                linked_qid_col = col
                break
        
        gold_qid_col = None
        for col in gold_qid_candidates:
            if col in df.columns:
                gold_qid_col = col
                break
        
        if not linked_qid_col or not gold_qid_col:
            logger.error(f"Required columns not found in {file_path}")
            return result
        
        # Create columns to store the extracted date components
        extraction_columns = [
            'gold_normalized', 'linked_normalized',
            'gold_year', 'gold_decade', 'gold_century',
            'linked_year', 'linked_decade', 'linked_century'
        ]
        for col in extraction_columns:
            df[col] = None
        
        # Process each row for evaluation
        for granularity in result.granularities:
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            
            for i, row in df.iterrows():
                linked_qid = str(row[linked_qid_col]) if pd.notna(row[linked_qid_col]) else ""
                gold_qid = str(row[gold_qid_col]) if pd.notna(row[gold_qid_col]) else ""
                
                # Skip rows with empty gold_qid for NIL files too (same behavior as QID files)
                if not gold_qid:
                    continue
                
                # Extract date components based on granularity
                # First process gold values
                if gold_qid.upper() == "NIL":
                    gold_value = None
                else:
                    if granularity == 'exact':
                        # For exact match, use the normalized date components
                        normalized_gold = normalize_date(gold_qid)
                        if normalized_gold:
                            df.at[i, 'gold_normalized'] = f"{normalized_gold[0]}-{normalized_gold[1]:02d}-{normalized_gold[2]:02d}"
                            gold_value = normalized_gold
                        else:
                            gold_value = None
                    elif granularity == 'year':
                        gold_value = extract_year(gold_qid)
                        df.at[i, 'gold_year'] = gold_value
                    elif granularity == 'decade':
                        gold_value = extract_decade(gold_qid)
                        df.at[i, 'gold_decade'] = gold_value
                    elif granularity == 'century':
                        gold_value = extract_century(gold_qid)
                        df.at[i, 'gold_century'] = gold_value
                
                # Then process linked values
                if not linked_qid or linked_qid.upper() == "NIL":
                    linked_value = None
                else:
                    if granularity == 'exact':
                        # For exact match, use the normalized date components
                        normalized_linked = normalize_date(linked_qid)
                        if normalized_linked:
                            df.at[i, 'linked_normalized'] = f"{normalized_linked[0]}-{normalized_linked[1]:02d}-{normalized_linked[2]:02d}"
                            linked_value = normalized_linked
                        else:
                            linked_value = None
                    elif granularity == 'year':
                        linked_value = extract_year(linked_qid)
                        df.at[i, 'linked_year'] = linked_value
                    elif granularity == 'decade':
                        linked_value = extract_decade(linked_qid)
                        df.at[i, 'linked_decade'] = linked_value
                    elif granularity == 'century':
                        linked_value = extract_century(linked_qid)
                        df.at[i, 'linked_century'] = linked_value
                
                # Different handling based on file type (NIL vs QID files)
                if result.entity_type.upper() == "NIL":
                    # Case for NIL files
                    
                    # If gold_qid is NIL
                    if gold_qid.upper() == "NIL":
                        # If linked_qid is NIL or empty -> True Negative
                        if not linked_qid or linked_qid.upper() == "NIL":
                            tn += 1
                            df.at[i, f'{granularity}_TN'] = 1
                        # If linked_qid is not NIL or empty -> False Positive
                        else:
                            fp += 1
                            df.at[i, f'{granularity}_FP'] = 1
                    # If gold_qid is empty string, skip (already done above)
                    # If gold_qid has a valid QID (non-NIL, non-empty)
                    else:
                        # If linked_qid is empty or NIL, count as false negative
                        if not linked_qid or linked_qid.upper() == "NIL":
                            fn += 1
                            df.at[i, f'{granularity}_FN'] = 1
                        else:
                            # Check if the values match at the specified granularity
                            match = False
                            if granularity == 'exact':
                                # For exact match, check the normalized components
                                if gold_value and linked_value:
                                    match = gold_value == linked_value
                            else:
                                # For other granularities, compare the extracted values
                                if gold_value is not None and linked_value is not None:
                                    match = gold_value == linked_value
                                    
                            if match:
                                tp += 1
                                df.at[i, f'{granularity}_TP'] = 1
                            else:
                                fp += 1
                                fn += 1
                                df.at[i, f'{granularity}_FP'] = 1
                                df.at[i, f'{granularity}_FN'] = 1
                else:
                    # For QID files
                    # If gold_qid is empty
                    if not gold_qid:
                        # If linked_qid is empty as well -> True Negative
                        if not linked_qid or linked_qid.upper() == "NIL":
                            tn += 1
                            df.at[i, f'{granularity}_TN'] = 1
                        # If linked_qid is not empty -> False Positive
                        else:
                            fp += 1
                            df.at[i, f'{granularity}_FP'] = 1
                    # For QID files, we should not have gold_qid = NIL as per requirements
                    elif gold_qid.upper() == "NIL":
                        logger.warning(f"Unexpected NIL value in gold column for QID file: {file_path}, row {i}")
                        continue
                    # If gold_qid has a valid QID
                    else:
                        # If linked_qid is empty or NIL, count as false negative
                        if not linked_qid or linked_qid.upper() == "NIL":
                            fn += 1
                            df.at[i, f'{granularity}_FN'] = 1
                        else:
                            # Check if the values match at the specified granularity
                            match = False
                            if granularity == 'exact':
                                # For exact match, check the normalized components
                                if gold_value and linked_value:
                                    match = gold_value == linked_value
                            else:
                                # For other granularities, compare the extracted values
                                if gold_value is not None and linked_value is not None:
                                    match = gold_value == linked_value
                                    
                            if match:
                                tp += 1
                                df.at[i, f'{granularity}_TP'] = 1
                            else:
                                fp += 1
                                fn += 1
                                df.at[i, f'{granularity}_FP'] = 1
                                df.at[i, f'{granularity}_FN'] = 1
            
            # Update the result object with granularity-specific metrics
            result.results[granularity]['tp'] = tp
            result.results[granularity]['fp'] = fp
            result.results[granularity]['fn'] = fn
            result.results[granularity]['tn'] = tn
            result.calculate_metrics(granularity)
            
            # Add metrics to the dataframe
            for metric in ['precision', 'recall', 'f1']:
                df[f'{granularity}_{metric}'] = result.results[granularity][metric]
        
        # Determine the appropriate output directory based on entity type and approach
        if result.approach.upper() == "ZS" or "noctx" in file_path.lower():
            # ZS directories
            if result.entity_type.upper() == "NIL":
                entity_output_dir = DEFAULT_ZS_NIL_DIR
            elif result.entity_type.upper() == "QID":
                entity_output_dir = DEFAULT_ZS_QID_DIR
            else:
                entity_output_dir = out_dir
        else:
            # RAG directories
            if result.entity_type.upper() == "NIL":
                entity_output_dir = DEFAULT_RAG_NIL_DIR
            elif result.entity_type.upper() == "QID":
                entity_output_dir = DEFAULT_RAG_QID_DIR
            else:
                entity_output_dir = out_dir
            
        # Create output filename
        base_filename = os.path.basename(file_path)
        out_path = os.path.join(entity_output_dir, base_filename.replace('.csv', '_granular_evaluated.csv'))
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        
        if verbose:
            logger.info(f"Evaluated: {file_path}")
            logger.info(f"  Entity Type: {result.entity_type}")
            logger.info(f"  Property Type: {result.property_type}")
            
            for granularity in result.granularities:
                logger.info(f"  {granularity.capitalize()} granularity:")
                logger.info(f"    TP: {result.results[granularity]['tp']}, FP: {result.results[granularity]['fp']}, FN: {result.results[granularity]['fn']}")
                logger.info(f"    Precision: {result.results[granularity]['precision']:.4f}, Recall: {result.results[granularity]['recall']:.4f}, F1: {result.results[granularity]['f1']:.4f}")
            
            logger.info(f"  Saved to: {out_path}")
    
    except Exception as e:
        logger.error(f"Error evaluating {file_path}: {e}", exc_info=True)
    
    return result


def normalize_model_name(model_str):
    """Normalize model name using the MODEL_MAPPINGS."""
    if not model_str:
        return "unknown"
        
    # Convert to lowercase for case-insensitive matching
    model_str_lower = str(model_str).lower()
    
    # Standard model mapping (consistent with main evaluation scripts)
    for key, value in MODEL_MAPPINGS.items():
        if key.lower() in model_str_lower:
            return value
            
    return model_str


def format_retriever_name(retriever):
    """Format retriever name for reporting."""
    if retriever in RETRIEVER_MAPPINGS:
        return RETRIEVER_MAPPINGS[retriever]
    return retriever


def generate_granular_table_reports(results: List[GranularEvaluationResult], report_dir: str, entity_type: str):
    """Generate structured reports in the format needed for LaTeX tables (similar to main evaluation scripts)."""
    # Create nested dictionaries for each granularity and metric
    granularities = ['exact', 'year', 'decade', 'century']
    metrics = ['precision', 'recall', 'f1']
    
    # Initialize lookup_tables structures
    metric_data = {}
    for granularity in granularities:
        metric_data[granularity] = {}
        for metric in metrics:
            metric_data[granularity][metric] = defaultdict(lambda: defaultdict(dict))

    # Add debugging info
    logger.info(f"Generating table reports for entity type: {entity_type}")
    entity_type_counts = {"NIL": 0, "QID": 0, "unknown": 0}

    # Process results
    for result in results:
        # Count entity types before filtering
        entity_type_counts[result.entity_type] += 1

        # Skip results that don't match the requested entity type
        if result.entity_type.upper() != entity_type.upper():
            logger.debug(f"Skipping result with entity_type={result.entity_type}, expected={entity_type}")
            continue

        # Additional debug info
        logger.debug(f"Processing result: {os.path.basename(result.file_path)}, "
                    f"entity_type={result.entity_type}, property={result.property_type}")

        # Normalize model and retriever names
        model = normalize_model_name(result.model)
        retriever = result.retriever

        # Store metrics for each granularity/model/retriever combination
        for granularity in granularities:
            for metric_name in metrics:
                metric_value = result.results[granularity][metric_name]
                
                # If there are duplicates, use the one with the best F1 score
                property_f1 = result.results[granularity]['f1']
                if granularity not in metric_data or \
                   metric_name not in metric_data[granularity] or \
                   model not in metric_data[granularity]['f1'] or \
                   retriever not in metric_data[granularity]['f1'][model] or \
                   property_f1 > metric_data[granularity]['f1'][model][retriever]:
                    metric_data[granularity][metric_name][model][retriever] = metric_value

    # Log summary of entity type counts
    logger.info(f"Entity type counts in results: {entity_type_counts}")

    # Create separate reports for each granularity and metric combination
    for granularity in granularities:
        for metric_name in metrics:
            # Convert to a list of rows for the CSV report
            rows = []

            # Create header row
            header = ["Model", "Retriever", granularity.capitalize()]

            # Sort models with Boyer-M first and rest alphabetically
            def model_sort_key(model):
                if 'Boyer-M' in model:
                    return (0, model)
                else:
                    return (1, model)
                
            sorted_models = sorted(metric_data[granularity][metric_name].keys(), key=model_sort_key)
            for model in sorted_models:
                # Sort retrievers by predefined order
                sorted_retrievers = sorted(metric_data[granularity][metric_name][model].keys(), 
                                         key=lambda r: RETRIEVER_ORDERING.get(r, 999))

                for retriever in sorted_retrievers:
                    retriever_display = format_retriever_name(retriever)
                    value = metric_data[granularity][metric_name][model][retriever]
                    rows.append([model, retriever_display, value])

            # Create DataFrame
            df = pd.DataFrame(rows, columns=header)

            # Save report
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, f"table_report_{entity_type.lower()}_{granularity}_{metric_name}.csv")
            df.to_csv(report_path, index=False)

            logger.info(f"Generated {granularity} {metric_name} report: {report_path}")

    return report_dir


def generate_combined_granular_report(report_dir: str, entity_type: str, output_dir: str):
    """Generate combined reports with all granularities and metrics (similar to main evaluation scripts)."""
    granularities = ['exact', 'year', 'decade', 'century']
    metrics = ['precision', 'recall', 'f1']
    
    # Read individual metric files for each granularity
    all_data = {}
    
    for granularity in granularities:
        all_data[granularity] = {}
        for metric in metrics:
            file_path = os.path.join(report_dir, f"table_report_{entity_type.lower()}_{granularity}_{metric}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                all_data[granularity][metric] = df
            else:
                logger.warning(f"Missing file: {file_path}")
                all_data[granularity][metric] = pd.DataFrame()

    # Create combined dataset
    combined_rows = []
    
    # Get all unique model/retriever combinations
    all_combinations = set()
    for granularity in granularities:
        for metric in metrics:
            if not all_data[granularity][metric].empty:
                for _, row in all_data[granularity][metric].iterrows():
                    all_combinations.add((row['Model'], row['Retriever']))
    
    # Sort combinations: first by model name, then Boyer-M retriever first within each model
    def combination_sort_key(combination):
        model, retriever = combination
        # Boyer-M should come first within each model group
        if retriever == 'Boyer-M':
            retriever_priority = -1
        else:
            retriever_priority = RETRIEVER_ORDERING.get(retriever.lower().replace('-', '_'), 999)
        return (model, retriever_priority)
    
    sorted_combinations = sorted(all_combinations, key=combination_sort_key)
    
    # Build combined report
    for model, retriever in sorted_combinations:
        row_data = {
            'Model': model,
            'Retriever': retriever
        }
        
        # Add columns for each granularity and metric
        for granularity in granularities:
            for metric in metrics:
                col_name = f"{granularity.capitalize()}_{metric.upper()}"
                
                # Find the value in the corresponding dataframe
                df = all_data[granularity][metric]
                if not df.empty:
                    matching_rows = df[(df['Model'] == model) & (df['Retriever'] == retriever)]
                    if not matching_rows.empty:
                        row_data[col_name] = matching_rows.iloc[0][granularity.capitalize()]
                    else:
                        row_data[col_name] = None
                else:
                    row_data[col_name] = None
        
        combined_rows.append(row_data)
    
    # Create combined DataFrame with reordered columns (P, R, F1 for each granularity)
    column_order = ['Model', 'Retriever']
    for granularity in granularities:
        column_order.extend([
            f"{granularity.capitalize()}_PRECISION",
            f"{granularity.capitalize()}_RECALL", 
            f"{granularity.capitalize()}_F1"
        ])
    
    combined_df = pd.DataFrame(combined_rows)
    combined_df = combined_df.reindex(columns=column_order)
    
    # Save combined report
    os.makedirs(output_dir, exist_ok=True)
    combined_path = os.path.join(output_dir, f"combined_granular_report_{entity_type.lower()}.csv")
    combined_df.to_csv(combined_path, index=False)
    
    logger.info(f"Generated combined granular report for {entity_type}: {combined_path}")
    return combined_path


def generate_latex_table(report_path, entity_type):
    """Generate a LaTeX table from a DoB granularity report."""
    
    # Read the CSV file
    df = pd.read_csv(report_path)
    
    # Filter out any "unknown" rows
    df = df[df['Model'].str.lower() != 'unknown']
    
    # Granularity levels in order of specificity
    granularities = ['Exact', 'Year', 'Decade', 'Century']
    
    # Check which columns are present in the file
    present_granularities = []
    for granularity in granularities:
        if f"{granularity}_PRECISION" in df.columns:
            present_granularities.append(granularity)
    
    # Generate LaTeX table
    latex = []
    
    # Table environment
    latex.append("\\begin{table}[htbp]")
    latex.append("  \\centering")
    latex.append("  \\small")
    latex.append("  \\resizebox{\\textwidth}{!}{")
    
    # Create column spec with vertical separators between granularity groups
    col_spec = "ll|" + "ccc|" * (len(present_granularities) - 1) + "ccc"
    latex.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    
    latex.append("      \\toprule")
    
    # Add header rows
    header1 = ["", ""]
    for i, gran in enumerate(present_granularities):
        if i < len(present_granularities) - 1:
            header1.append(f"\\multicolumn{{3}}{{c|}}{{{gran}}}")
        else:
            header1.append(f"\\multicolumn{{3}}{{c}}{{{gran}}}")
    latex.append("      " + " & ".join(header1) + " \\\\")
    
    # Second header row with P, R, F1
    header2 = ["Model", "Retriever"]
    for _ in present_granularities:
        header2.extend(["P", "R", "F1"])
    latex.append("      " + " & ".join(header2) + " \\\\")
    
    latex.append("      \\midrule")
    
    # Determine maximum values for bold formatting
    max_vals = {}
    for granularity in present_granularities:
        for metric in ['PRECISION', 'RECALL', 'F1']:
            col = f"{granularity}_{metric}"
            if col in df.columns:
                max_vals[col] = round(df[col].max(), 2)
    
    # Group lookup_tables by model for better presentation
    models = df['Model'].unique()
    for i, model in enumerate(models):
        model_rows = df[df['Model'] == model]
        
        for j, (idx, row) in enumerate(model_rows.iterrows()):
            if j == 0:
                row_start = f"      \\multirow{{{len(model_rows)}}}{{*}}{{{model}}} & {row['Retriever']}"
            else:
                row_start = f"       & {row['Retriever']}"
            
            values = []
            for granularity in present_granularities:
                # Order: P, R, F1
                for metric in ['PRECISION', 'RECALL', 'F1']:
                    col = f"{granularity}_{metric}"
                    val = row.get(col)
                    
                    if pd.isna(val) or val is None:
                        values.append('--')
                    else:
                        # Format with bold if it's max value
                        formatted = f"{float(val):.2f}"
                        if round(val, 2) == max_vals.get(col, float('-inf')):
                            formatted = f"\\textbf{{{formatted}}}"
                        values.append(formatted)
            
            latex.append(f"{row_start} & " + " & ".join(values) + " \\\\")
        
        # Add a horizontal line between models except for the last one
        if i < len(models) - 1:
            latex.append("      \\midrule")
    
    latex.append("      \\bottomrule")
    latex.append("    \\end{tabular}")
    latex.append("  }")
    
    # Add caption and label
    caption = "DoB performance metrics at different granularity levels for "
    if entity_type.upper() == "QID":
        caption += "entity linking on known entities (QID)."
    else:
        caption += "NIL identification (entities not in the knowledge base)."
    
    latex.append(f"  \\caption{{{caption}}}")
    latex.append(f"  \\label{{tab:dob_{entity_type.lower()}_granularity}}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def main():
    """Main function to run the enhanced granular DoB evaluation scripts."""
    parser = argparse.ArgumentParser(description="Evaluate DoB property linking at different granularity levels with enhanced reporting.")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help=f"Input directory to search for DoB CSV files (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help=f"Output directory for evaluated files (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--report-output", default=DEFAULT_REPORT_DIR, help=f"Output directory for reports (default: {DEFAULT_REPORT_DIR})")
    parser.add_argument("--latex-output", default=DEFAULT_LATEX_DIR, help=f"Output directory for LaTeX tables (default: {DEFAULT_LATEX_DIR})")
    parser.add_argument("--skip-files", nargs="+", help="Skip files containing these strings")
    parser.add_argument("--include-files", nargs="+", help="Only include files containing these strings")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug output (even more verbose)")
    parser.add_argument("--test-date-normalization", action="store_true", help="Run tests for date normalization")
    parser.add_argument("--skip-voting", action="store_true", help="Skip voting algorithm results (they're included by default)")
    parser.add_argument("--include-voting", action="store_true", default=True, help="Include voting algorithm results from the /scripts/linking/output directory")
    
    args = parser.parse_args()
    
    # Set logging level based on verbose/debug flags
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    
    # Option to test date normalization
    if args.test_date_normalization:
        test_dates = [
            "1809-02-03T00:00:00Z",
            "1809-02-03",
            "1809/02/03",
            "February 3, 1809",
            "3 February 1809",
            "born on February 3, 1809",
            "born 3rd February 1809",
            "born in 1809",
            "1800s",
            "19th century"
        ]
        
        logger.info("Testing date normalization:")
        for date_str in test_dates:
            normalized = normalize_date(date_str)
            year = extract_year(date_str)
            decade = extract_decade(date_str)
            century = extract_century(date_str)
            
            logger.info(f"Original: {date_str}")
            logger.info(f"  Normalized: {normalized}")
            logger.info(f"  Year: {year}")
            logger.info(f"  Decade: {decade}")
            logger.info(f"  Century: {century}")
            logger.info("---")
        
        # Test exact date comparison
        test_pairs = [
            ("1809-02-03T00:00:00Z", "1809-02-03"),
            ("1809-02-03", "1809/02/03"),
            ("1809-02-03", "3 February 1809"),
            ("1809-02-03", "born on February 3, 1809"),
            ("1809-02-03", "1809"),  # This should not match for exact comparison
            ("1809", "1809-02-03")   # This should not match for exact comparison
        ]
        
        logger.info("Testing exact date comparison:")
        for date1, date2 in test_pairs:
            match = compare_dates_exact(date1, date2)
            logger.info(f"Comparing: '{date1}' and '{date2}'")
            logger.info(f"  Match: {match}")
            logger.info("---")
        
        return 0
    
    # Override output and report directories to use property-specific paths
    args.output = DEFAULT_OUTPUT_DIR
    args.report_output = DEFAULT_REPORT_DIR
    args.latex_output = DEFAULT_LATEX_DIR
    
    # Log the output paths
    logger.info(f"Using property-specific output paths:")
    logger.info(f"  - Output directory: {args.output}")
    logger.info(f"  - Report directory: {args.report_output}")
    logger.info(f"  - LaTeX directory: {args.latex_output}")
    
    # Find files to evaluate
    logger.info("Finding DoB files to evaluate...")
    
    # Determine input directories - always include the default directory
    input_dirs = [args.input]
    
    # Add voting algorithm directories by default unless --skip-voting is specified
    if args.include_voting and not args.skip_voting:
        logger.info(f"Including voting algorithm directories: {VOTING_DIRS}")
        input_dirs.extend(VOTING_DIRS)
    
    files = find_dob_files(input_dirs, args.include_files, args.skip_files)
    
    if not files:
        logger.error("No matching DoB files found. Check your input directory and filters.")
        return 1
    
    logger.info(f"Found {len(files)} files to evaluate.")
    
    # Filter out files that already have "_evaluated" in their name to avoid redundancy
    files = [f for f in files if "_evaluated_evaluated" not in f]
    logger.info(f"After filtering redundant files: {len(files)} files")
    
    # Count how many QID and NIL files were found for validation
    qid_files = [f for f in files if "qid" in f.lower()]
    nil_files = [f for f in files if "nil" in f.lower()]
    unknown_files = [f for f in files if "qid" not in f.lower() and "nil" not in f.lower()]

    logger.info(f"Files breakdown: {len(nil_files)} NIL files, {len(qid_files)} QID files, {len(unknown_files)} unknown")

    if len(qid_files) == 0:
        logger.warning("⚠️ NO QID FILES FOUND! QID report will be empty.")
    
    # Evaluate each file
    results = []
    for i, file in enumerate(files):
        logger.info(f"Evaluating file {i+1}/{len(files)}: {file}...")
        result = evaluate_file_granular(file, args.output, args.verbose)
        results.append(result)

        # Print entity type stats every 10 files
        if (i+1) % 10 == 0:
            entity_types = {r.entity_type: 0 for r in results}
            for r in results:
                entity_types[r.entity_type] += 1
            logger.info(f"Entity types processed so far: {entity_types}")

    # Check if there are any QID results before generating reports
    qid_results = [r for r in results if r.entity_type.upper() == "QID"]
    nil_results = [r for r in results if r.entity_type.upper() == "NIL"]

    logger.info(f"Results breakdown: {len(nil_results)} NIL results, {len(qid_results)} QID results")

    if len(qid_results) == 0:
        logger.warning("⚠️ NO QID RESULTS WERE PROCESSED! QID report will be empty.")
    
    # Generate structured reports for LaTeX tables
    logger.info("Generating enhanced reports...")
    
    for entity_type in ["nil", "qid"]:
        # Check if there are any results for this entity type
        entity_results = [r for r in results if r.entity_type.upper() == entity_type.upper()]
        
        if entity_results:
            logger.info(f"Generating detailed reports for {entity_type.upper()} entity type...")
            
            # Generate table reports (similar to main evaluation scripts)
            generate_granular_table_reports(results, args.report_output, entity_type)
            
            # Generate combined report
            combined_report = generate_combined_granular_report(args.report_output, entity_type, args.report_output)
            
            # Generate LaTeX table
            os.makedirs(args.latex_output, exist_ok=True)
            
            if combined_report and os.path.exists(combined_report):
                # Generate LaTeX table from combined report
                latex_table = generate_latex_table(combined_report, entity_type)
                
                # Save LaTeX table
                latex_path = os.path.join(args.latex_output, f'{entity_type.lower()}_granularity_table.tex')
                with open(latex_path, 'w') as f:
                    f.write(latex_table)
                
                logger.info(f"Generated LaTeX table for {entity_type.upper()}: {latex_path}")
            else:
                logger.warning(f"Could not generate LaTeX table for {entity_type.upper()} - combined report missing")
        else:
            logger.warning(f"No results found for {entity_type.upper()} entity type.")

    logger.info("All granular reports generated:")
    for entity_type in ["nil", "qid"]:
        for granularity in ['exact', 'year', 'decade', 'century']:
            for metric in ['precision', 'recall', 'f1']:
                report_path = os.path.join(args.report_output, f"table_report_{entity_type}_{granularity}_{metric}.csv")
                if os.path.exists(report_path):
                    logger.info(f"  - {os.path.basename(report_path)}")

    logger.info("Enhanced DoB granular evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())