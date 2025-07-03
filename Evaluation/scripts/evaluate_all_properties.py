#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation scripts for NIL Grounding property linking.

This scripts evaluates property linking results for any property type, including:
- CoC (Country of Citizenship)
- FamilyName
- GivenName
- DoB (Date of Birth)
- occupation
- sexGender

The scripts:
- Processes CSV files with linked entities
- Adds evaluation columns (TP, FP, FN)
- Calculates metrics (precision, recall, F1)
- Generates detailed reports in the format needed for LaTeX table generation

Special handling for Date of Birth (DoB) property:
- For DoB properties, the scripts extracts and compares years from dates
- This allows for more flexible matching when only the year matters
- Supported date formats include ISO dates, dates with time, and plain years
- The scripts will consider a match if at least one year is common between gold and linked values
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
from typing import List, Dict, Tuple, Any, Optional, Set
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
PROPERTY_TYPES = ["CoC", "FamilyName", "GivenName", "DoB", "occupation", "sexGender", "Gender"]
PROPERTY_DISPLAY = {'CoC': 'CoC', 'DoB': 'DoB', 'FamilyName': 'FName',
                    'GivenName': 'GName', 'occupation': 'Occ', 'sexGender': 'Gender'}

# Date format patterns
DATE_FORMATS = [
    r'(\d{4})-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z',  # ISO date with time: 1728-05-11T00:00:00Z
    r'(\d{4})-\d{2}-\d{2}',                    # ISO date: 1728-05-11
    r'(\d{4})\/\d{1,2}\/\d{1,2}',              # Slash date: 1728/05/11
    r'\b(1\d{3}|20\d{2}|21\d{2})\b'            # Year only: 1728, 2023, 2100
]
DEFAULT_INPUT_DIR = #
DEFAULT_OUTPUT_DIR = #
DEFAULT_REPORT_DIR = #

# Model mappings to standardize names (optimized for LaTeX tables)
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
    'majority_voting': 'Majority',  # Shortened to avoid overflow
    'boyer_moore_voting': 'Boyer-M',  # Fixed to proper display name
    'flexible_boyer_voting': 'Flexible-Boyer',
    'adaptive_threshold_voting': 'Adaptive',  # Shortened to avoid overflow
    'boyer-moore': 'Boyer-M',
    'adaptive_threshold': 'Adaptive',
    'boyer_moore': 'Boyer-M',
    'flexible_boyer': 'Flexible-B',
    'flexible_boyer_moore_voting': 'Flexible-Boyer',
    'boyer_moore_per_model': 'Boyer-M'
}

# Retriever formatting and ordering
RETRIEVER_MAPPINGS = {
    'none': '-',  # Zero-shot (changed from 'ZS' to '-' to avoid overflow issues)
    'bge-large': 'BGE',
    'bm25': 'BM25',
    'instructor-xl': 'Inst',   # Shortened from 'Inst-X'
    'contriever': 'Contr',
    'gtr-xl': 'GTR-X',
    'gtr-large': 'GTR-L',
    'adaptive_threshold': 'Adaptive',
    'boyer_moore': 'Boyer-M',
    'flexible_boyer': 'Flexible-B',
    'adaptive_threshold_voting': 'Adaptive',
    'boyer_moore_voting': 'Boyer-M',
    'flexible_boyer_moore_voting': 'Flexible-B',
    'boyer_moore_per_model': 'Boyer-M'
}

RETRIEVER_ORDERING = {
    'boyer_moore': -2,  # Boyer-Moore should come first
    'boyer-moore': -2,
    'boyer_moore_per_model': -2,
    'none': 0,  # Zero-shot (no retriever)
    'bge-large': 1,
    'bm25': 2,
    'instructor-xl': 3,
    'contriever': 4,
    'gtr-xl': 5,
    'gtr-large': 6,
}

def extract_year(date_str: str) -> Optional[int]:
    """
    Extract the year component from a date string in various formats.
    
    Args:
        date_str (str): A string containing a date in various possible formats
        
    Returns:
        Optional[int]: The extracted year as an integer, or None if no year could be extracted
    """
    if not date_str or not isinstance(date_str, str):
        return None
        
    # Try each regex pattern to extract year
    for pattern in DATE_FORMATS:
        match = re.search(pattern, date_str)
        if match:
            try:
                year = int(match.group(1))
                # Validate year (limit to reasonable range for people's birth years)
                if 1000 <= year <= 2023:  # Adjust upper bound as needed
                    return year
            except (ValueError, IndexError):
                continue
                
    # Additional handling for special cases
    # Try to extract 4-digit years anywhere in the string (supporting years up to 2100)
    years = re.findall(r'\b(1\d{3}|20\d{2}|21\d{2})\b', date_str)
    if years:
        try:
            year = int(years[0])
            # Using broader validation range to accommodate future dates
            if 1000 <= year <= 2100:
                return year
        except (ValueError, IndexError):
            pass
            
    return None

class EvaluationResult:
    """Class to store evaluation results for a single file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.property_type = self._extract_property_type()
        self.approach = self._extract_approach()
        self.retriever = self._extract_retriever()
        self.model = self._extract_model()
        self.entity_type = self._extract_entity_type()
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        
    def _extract_property_type(self) -> str:
        """Extract property type from filename."""
        for prop in PROPERTY_TYPES:
            if f"QA_{prop}_" in self.file_path or f"_{prop}_" in self.file_path:
                # Map "Gender" to "sexGender" for consistency
                if prop == "Gender":
                    return "sexGender"
                return prop
        return "unknown"
    
    def _extract_approach(self) -> str:
        """Extract approach (RAG or ZS) from path or filename."""
        if "RAG" in self.file_path:
            return "RAG"
        elif "ZS" in self.file_path:
            return "ZS"
        elif "answers" in self.file_path:
            return "RAG"
        elif "ZS_answers" in self.file_path:
            return "ZS"
        return "unknown"
    
    def _extract_retriever(self) -> str:
        """Extract retriever from filename."""
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
        """Extract model from filename."""
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
    
    def calculate_metrics(self) -> None:
        """Calculate precision, recall, and F1 score."""
        try:
            self.precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
            self.recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
            self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall) if (self.precision + self.recall) > 0 else 0.0
        except ZeroDivisionError:
            logger.warning(f"Division by zero when calculating metrics for {self.file_path}")
            self.precision = 0.0
            self.recall = 0.0
            self.f1 = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for reporting."""
        return {
            "file": os.path.basename(self.file_path),
            "property": self.property_type,
            "approach": self.approach,
            "retriever": self.retriever,
            "model": self.model,
            "entity_type": self.entity_type,
            "true_positives": self.tp,
            "false_positives": self.fp,
            "false_negatives": self.fn,
            "true_negatives": self.tn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1
        }


def find_files(folders: List[str], include_files: List[str], skip_files: List[str], property_type: Optional[str] = None) -> List[str]:
    """Find CSV files in the given folders matching the criteria."""
    all_files = []

    # Count files by entity type for debugging
    entity_type_counts = {"nil": 0, "qid": 0, "unknown": 0}

    for folder in folders:
        # Convert to Path object for easier manipulation
        folder_path = Path(folder)
        if not folder_path.exists():
            logger.warning(f"Folder does not exist: {folder}")
            continue

        # Find all CSV files recursively
        csv_files = glob.glob(f"{folder}/**/*.csv", recursive=True)
        logger.info(f"Found {len(csv_files)} total CSV files in {folder}")

        # Filter files based on property type
        if property_type:
            # Handle sexGender and Gender as equivalent
            if property_type == "sexGender":
                csv_files = [f for f in csv_files if "sexGender" in f or "Gender" in f]
            else:
                csv_files = [f for f in csv_files if property_type in f]
            logger.info(f"After property type filter ({property_type}): {len(csv_files)} files")

        # Filter files based on include_files and skip_files
        if include_files:
            filtered_files = []
            for file in csv_files:
                if all(pattern in file for pattern in include_files):
                    filtered_files.append(file)
            csv_files = filtered_files
            logger.info(f"After include_files filter: {len(csv_files)} files")

        if skip_files:
            filtered_files = []
            for file in csv_files:
                if not any(pattern in file for pattern in skip_files):
                    filtered_files.append(file)
            csv_files = filtered_files
            logger.info(f"After skip_files filter: {len(csv_files)} files")

        # Count files by entity type (debug)
        for file in csv_files:
            file_lower = file.lower()
            if "nil" in file_lower:
                entity_type_counts["nil"] += 1
            elif "qid" in file_lower:
                entity_type_counts["qid"] += 1
            else:
                entity_type_counts["unknown"] += 1

        all_files.extend(csv_files)

    # Log entity type counts
    logger.info(f"Files by entity type: {entity_type_counts}")

    # Print a sample of QID files for verification
    qid_files = [f for f in all_files if "qid" in f.lower()]
    if qid_files:
        logger.info(f"Sample of QID files found (up to 5): {qid_files[:5]}")
    else:
        logger.warning("No QID files found! This may explain empty QID report.")

    return all_files


def evaluate_file(file_path: str, out_dir: str, verbose: bool = False) -> EvaluationResult:
    """Evaluate a single CSV file."""
    result = EvaluationResult(file_path)

    try:
        # Read the CSV file
        df = pd.read_csv(file_path, delimiter=',', low_memory=False)

        # Add evaluation columns if they don't exist
        if 'TP' not in df.columns:
            df['TP'] = 0
        if 'FP' not in df.columns:
            df['FP'] = 0
        if 'FN' not in df.columns:
            df['FN'] = 0

        # Determine entity type using a hybrid approach that favors filename but considers content
        
        # Determine entity type from filename
        entity_type_from_filename = "unknown"
        if "qid" in file_path.lower():
            entity_type_from_filename = "QID"
        elif "nil" in file_path.lower():
            entity_type_from_filename = "NIL"
            
        # Check if entity_type exists in dataframe
        entity_type_from_data = None
        if 'entity_type' in df.columns and len(df) > 0:
            # Get the most common entity_type in the dataframe
            entity_counts = df['entity_type'].value_counts()
            if not entity_counts.empty:
                entity_type_from_data = entity_counts.index[0]
                
        # Make a decision on which entity_type to use
        final_entity_type = entity_type_from_filename
        
        # Log warnings if there's a mismatch
        if entity_type_from_data and entity_type_from_data != entity_type_from_filename:
            if verbose:
                logger.warning(f"Entity type mismatch in {file_path}! Dataframe has {entity_type_from_data} but filename indicates {entity_type_from_filename}")
                logger.warning(f"Using {final_entity_type} as the entity type for evaluation")
        
        # Update result's entity_type
        result.entity_type = final_entity_type
        
        # Set entity_type column in dataframe
        df['entity_type'] = final_entity_type

        logger.debug(f"Using entity_type={entity_type_from_filename} for {os.path.basename(file_path)}")
        
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
        
        # Perform the evaluation
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        # Add TN tracking if not exists
        if 'TN' not in df.columns:
            df['TN'] = 0

        for i, row in df.iterrows():
            linked_qid = str(row[linked_qid_col]) if pd.notna(row[linked_qid_col]) else ""
            gold_qid = str(row[gold_qid_col]) if pd.notna(row[gold_qid_col]) else ""

            # Different handling based on file type (NIL vs QID files)
            if result.entity_type.upper() == "NIL":
                # Case for NIL files:

                # If gold_qid is NIL
                if gold_qid.upper() == "NIL":
                    # If linked_qid is NIL or empty -> True Negative
                    if not linked_qid or linked_qid.upper() == "NIL":
                        tn += 1
                        df.at[i, 'TN'] = 1
                    # If linked_qid is not NIL or empty -> False Positive
                    else:
                        fp += 1
                        df.at[i, 'FP'] = 1
                # If gold_qid is empty string, skip
                elif not gold_qid:
                    continue
                # If gold_qid has a valid QID (non-NIL, non-empty)
                else:
                    # If linked_qid is empty or NIL, count as false negative
                    if not linked_qid or linked_qid.upper() == "NIL":
                        fn += 1
                        df.at[i, 'FN'] = 1
                    else:
                        # Process QIDs to find matches
                        # Normalize gold QIDs
                        normalized_gold = gold_qid.replace(';', ',').replace('|', ',')
                        gold_qids = set()
                        for qid in re.split(r'[,;\s]+', normalized_gold):
                            qid = qid.strip()
                            if qid and qid.upper() != 'NIL':
                                gold_qids.add(qid)

                        # Normalize linked QIDs
                        normalized_linked = linked_qid.replace(';', ',').replace('|', ',')
                        linked_qids = set()
                        for qid in re.split(r'[,;\s]+', normalized_linked):
                            qid = qid.strip()
                            if qid and qid.upper() != 'NIL':
                                linked_qids.add(qid)

                        # Check if there's at least one common QID between gold and linked
                        common_qids = gold_qids.intersection(linked_qids)

                        # Special handling for DoB (Date of Birth) property
                        if result.property_type == 'DoB':
                            # Extract years from both gold and linked QIDs
                            gold_years = set()
                            for qid in gold_qids:
                                year = extract_year(qid)
                                if year:
                                    gold_years.add(year)
                                elif verbose:
                                    logger.debug(f"Could not extract year from gold_qid: {qid}")
                                    
                            linked_years = set()
                            for qid in linked_qids:
                                year = extract_year(qid)
                                if year:
                                    linked_years.add(year)
                                elif verbose:
                                    logger.debug(f"Could not extract year from linked_qid: {qid}")
                            
                            # Add extracted years to dataframe for debugging (optional)
                            df.at[i, 'gold_years'] = ','.join(str(y) for y in gold_years) if gold_years else ''
                            df.at[i, 'linked_years'] = ','.join(str(y) for y in linked_years) if linked_years else ''
                            
                            # Count match if years match, even if full dates don't
                            if gold_years and linked_years and gold_years.intersection(linked_years):
                                tp += 1
                                df.at[i, 'TP'] = 1
                                # Store the common years for reference
                                common_years = gold_years.intersection(linked_years)
                                df.at[i, 'common_years'] = ','.join(str(y) for y in common_years)
                            else:
                                # If no year matches, it's a false negative (gold entity wasn't identified correctly)
                                fn += 1
                                df.at[i, 'FN'] = 1
                                df.at[i, 'common_years'] = ''
                                # Also mark as a false positive if linked_years is not empty (something was linked but wrongly)
                                if linked_years:
                                    fp += 1
                                    df.at[i, 'FP'] = 1
                                
                        # Standard exact string matching for other property types
                        else:
                            # If there's at least one match, count as true positive
                            if common_qids:
                                tp += 1
                                df.at[i, 'TP'] = 1
                            # If there are no common QIDs, it's a false negative (gold entity wasn't found)
                            else:
                                fn += 1
                                df.at[i, 'FN'] = 1
                                # Also mark as false positive if linked_qids is not empty (something was linked incorrectly)
                                if linked_qids:
                                    fp += 1
                                    df.at[i, 'FP'] = 1
            else:  # For QID files
                # If gold_qid is empty
                if not gold_qid:
                    # If linked_qid is empty as well -> True Negative
                    if not linked_qid or linked_qid.upper() == "NIL":
                        tn += 1
                        df.at[i, 'TN'] = 1
                    # If linked_qid is not empty -> False Positive
                    else:
                        fp += 1
                        df.at[i, 'FP'] = 1
                # For QID files, we should not have gold_qid = NIL as per requirements
                elif gold_qid.upper() == "NIL":
                    logger.warning(f"Unexpected NIL value in gold column for QID file: {file_path}, row {i}")
                    continue
                # If gold_qid has a valid QID
                else:
                    # If linked_qid is empty or NIL, count as false negative
                    if not linked_qid or linked_qid.upper() == "NIL":
                        fn += 1
                        df.at[i, 'FN'] = 1
                    else:
                        # Process QIDs to find matches
                        # Normalize gold QIDs
                        normalized_gold = gold_qid.replace(';', ',').replace('|', ',')
                        gold_qids = set()
                        for qid in re.split(r'[,;\s]+', normalized_gold):
                            qid = qid.strip()
                            if qid and qid.upper() != 'NIL':
                                gold_qids.add(qid)

                        # Normalize linked QIDs
                        normalized_linked = linked_qid.replace(';', ',').replace('|', ',')
                        linked_qids = set()
                        for qid in re.split(r'[,;\s]+', normalized_linked):
                            qid = qid.strip()
                            if qid and qid.upper() != 'NIL':
                                linked_qids.add(qid)

                        # Check if there's at least one common QID between gold and linked
                        common_qids = gold_qids.intersection(linked_qids)

                        # Special handling for DoB (Date of Birth) property
                        if result.property_type == 'DoB':
                            # Extract years from both gold and linked QIDs
                            gold_years = set()
                            for qid in gold_qids:
                                year = extract_year(qid)
                                if year:
                                    gold_years.add(year)
                                elif verbose:
                                    logger.debug(f"Could not extract year from gold_qid: {qid}")
                                    
                            linked_years = set()
                            for qid in linked_qids:
                                year = extract_year(qid)
                                if year:
                                    linked_years.add(year)
                                elif verbose:
                                    logger.debug(f"Could not extract year from linked_qid: {qid}")
                            
                            # Add extracted years to dataframe for debugging (optional)
                            df.at[i, 'gold_years'] = ','.join(str(y) for y in gold_years) if gold_years else ''
                            df.at[i, 'linked_years'] = ','.join(str(y) for y in linked_years) if linked_years else ''
                            
                            # Count match if years match, even if full dates don't
                            if gold_years and linked_years and gold_years.intersection(linked_years):
                                tp += 1
                                df.at[i, 'TP'] = 1
                                # Store the common years for reference
                                common_years = gold_years.intersection(linked_years)
                                df.at[i, 'common_years'] = ','.join(str(y) for y in common_years)
                            else:
                                # If no year matches, it's a false negative (gold entity wasn't identified correctly)
                                fn += 1
                                df.at[i, 'FN'] = 1
                                df.at[i, 'common_years'] = ''
                                # Also mark as a false positive if linked_years is not empty (something was linked but wrongly)
                                if linked_years:
                                    fp += 1
                                    df.at[i, 'FP'] = 1
                                
                        # Standard exact string matching for other property types
                        else:
                            # If there's at least one match, count as true positive
                            if common_qids:
                                tp += 1
                                df.at[i, 'TP'] = 1
                            # If there are no common QIDs, it's a false negative (gold entity wasn't found)
                            else:
                                fn += 1
                                df.at[i, 'FN'] = 1
                                # Also mark as false positive if linked_qids is not empty (something was linked incorrectly)
                                if linked_qids:
                                    fp += 1
                                    df.at[i, 'FP'] = 1

        # Update the result object
        result.tp = tp
        result.fp = fp
        result.fn = fn
        result.tn = tn
        result.calculate_metrics()

        # Add metrics to the dataframe
        df['precision'] = result.precision
        df['recall'] = result.recall
        df['f1'] = result.f1
        
        # Save the evaluated file
        out_path = get_output_path(file_path, out_dir)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        
        if verbose:
            logger.info(f"Evaluated: {file_path}")
            logger.info(f"  Entity Type: {result.entity_type}")
            logger.info(f"  Property Type: {result.property_type}")
            logger.info(f"  TP: {tp}, FP: {fp}, FN: {fn}")
            logger.info(f"  Precision: {result.precision:.4f}, Recall: {result.recall:.4f}, F1: {result.f1:.4f}")
            logger.info(f"  Saved to: {out_path}")
            
            # Additional logging for DoB property type
            if result.property_type == 'DoB':
                logger.info("  Using year comparison for DoB evaluation")
    
    except Exception as e:
        logger.error(f"Error evaluating {file_path}: {e}")
    
    return result


def get_output_path(input_path: str, output_dir: str) -> str:
    """Generate output path preserving original folder structure."""
    # Extract property type, approach type, entity type, and file name
    basename = os.path.basename(input_path)
    dirname = os.path.dirname(input_path)

    # Try to extract property from path or filename
    property_type = None
    for prop in PROPERTY_TYPES:
        if prop in dirname or prop in basename:
            property_type = prop
            # Map "Gender" to "sexGender" for consistency in output directories
            if prop == "Gender":
                property_type = "sexGender"
            break

    # Determine approach (RAG or ZS)
    approach = "unknown"
    if "RAG" in dirname or ("answers" in dirname and "ZS_answers" not in dirname):
        approach = "RAG"
    elif "ZS" in dirname or "ZS_answers" in dirname:
        approach = "ZS"

    # Determine entity type (NIL or QID)
    entity_type = "unknown"
    if "nil" in basename.lower():
        entity_type = "nil"
    elif "qid" in basename.lower():
        entity_type = "qid"

    # Create output filename
    name, ext = os.path.splitext(basename)
    eval_name = f"{name}_evaluated{ext}"

    # Create directory structure in output directory mirroring linking output structure
    if property_type:
        return os.path.join(output_dir, property_type, approach, entity_type, eval_name)
    else:
        # Fallback if property type not found
        return os.path.join(output_dir, eval_name)


def normalize_model_name(model_str):
    """Normalize model name using the MODEL_MAPPINGS."""
    for key, value in MODEL_MAPPINGS.items():
        if key in model_str:
            return value
    return model_str


def format_retriever_name(retriever):
    """Format retriever name for reporting."""
    if retriever in RETRIEVER_MAPPINGS:
        return RETRIEVER_MAPPINGS[retriever]
    return retriever


def generate_table_report(results: List[EvaluationResult], report_dir: str, entity_type: str):
    """Generate a structured report in the format needed for LaTeX tables."""
    # Create nested dictionaries for each metric (precision, recall, f1)
    precision_data = defaultdict(lambda: defaultdict(dict))
    recall_data = defaultdict(lambda: defaultdict(dict))
    f1_data = defaultdict(lambda: defaultdict(dict))

    # Add debugging info
    logger.info(f"Generating report for entity type: {entity_type}")
    entity_type_counts = {"NIL": 0, "QID": 0, "unknown": 0}
    property_counts = defaultdict(int)

    # Process results
    for result in results:
        # Count entity types before filtering
        entity_type_counts[result.entity_type] += 1
        property_counts[result.property_type] += 1

        # Skip results that don't match the requested entity type
        if result.entity_type.upper() != entity_type.upper():
            logger.debug(f"Skipping result with entity_type={result.entity_type}, expected={entity_type}")
            continue

        # Additional debug info
        logger.debug(f"Processing result: {os.path.basename(result.file_path)}, "
                    f"entity_type={result.entity_type}, property={result.property_type}, "
                    f"metrics: P={result.precision:.2f}, R={result.recall:.2f}, F1={result.f1:.2f}")

        # Normalize model and retriever names
        model = normalize_model_name(result.model)
        retriever = result.retriever
        property_type = result.property_type

        # Use property display name
        property_display = PROPERTY_DISPLAY.get(property_type, property_type)

        # Store metrics for each model/retriever/property combination
        # If there are duplicates, use the one with the best F1 score
        if property_display not in f1_data[model][retriever] or result.f1 > f1_data[model][retriever][property_display]:
            precision_data[model][retriever][property_display] = result.precision
            recall_data[model][retriever][property_display] = result.recall
            f1_data[model][retriever][property_display] = result.f1

    # Log summary of entity type counts
    logger.info(f"Entity type counts in results: {entity_type_counts}")
    logger.info(f"Property type counts in results: {dict(property_counts)}")

    # Get all properties used in results
    all_properties = set()
    for model_data in f1_data.values():
        for retriever_data in model_data.values():
            all_properties.update(retriever_data.keys())

    # Sort properties in the order defined in PROPERTY_DISPLAY
    sorted_properties = sorted(all_properties,
                              key=lambda p: list(PROPERTY_DISPLAY.values()).index(p) if p in PROPERTY_DISPLAY.values() else 999)

    # Create separate reports for precision, recall, and f1
    for metric_name, metric_data in [
        ("precision", precision_data),
        ("recall", recall_data),
        ("f1", f1_data)
    ]:
        # Convert to a list of rows for the CSV report
        rows = []

        # Create header row
        header = ["Model", "Retriever"] + sorted_properties

        # Sort models and add rows with Boyer-M first and rest alphabetically
        def model_sort_key(model):
            if 'Boyer-M' in model:
                return (0, model)
            else:
                return (1, model)
            
        sorted_models = sorted(metric_data.keys(), key=model_sort_key)
        for model in sorted_models:
            # Sort retrievers by predefined order
            sorted_retrievers = sorted(metric_data[model].keys(), key=lambda r: RETRIEVER_ORDERING.get(r, 999))

            for retriever in sorted_retrievers:
                retriever_display = format_retriever_name(retriever)
                row = [model, retriever_display]

                # Add metric values for each property
                for prop in sorted_properties:
                    value = metric_data[model][retriever].get(prop, None)
                    row.append(value)

                rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows, columns=header)

        # Save report
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"table_report_{entity_type.lower()}_{metric_name}.csv")
        df.to_csv(report_path, index=False)

        logger.info(f"Generated {metric_name} report: {report_path}")

    # Return the path to the F1 report for backward compatibility
    return os.path.join(report_dir, f"table_report_{entity_type.lower()}_f1.csv")


def main():
    """Main function to run the evaluation scripts."""
    parser = argparse.ArgumentParser(description="Evaluate property linking results.")
    parser.add_argument("--property", choices=PROPERTY_TYPES, help="Property type to evaluate (default: all)")
    parser.add_argument("--folders", nargs="+", default=[DEFAULT_INPUT_DIR], help=f"Input folders to search for CSV files (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help=f"Output directory for evaluated files (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--report-output", default=DEFAULT_REPORT_DIR, help=f"Output directory for reports (default: {DEFAULT_REPORT_DIR})")
    parser.add_argument("--skip-files", nargs="+", help="Skip files containing these strings")
    parser.add_argument("--include-files", nargs="+", help="Only include files containing these strings")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug output (even more verbose)")
    parser.add_argument("--include-voting", action="store_true", default=True, help="Include voting algorithm results from the /scripts/linking/output directory")

    args = parser.parse_args()

    # Set logging level based on verbose/debug flags
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    # Add voting algorithm directories if requested
    if args.include_voting:
        voting_dirs = [
            "",
            "",
            ""
        ]
        args.folders.extend(voting_dirs)
        logger.info(f"Including voting algorithm directories: {voting_dirs}")

    # Find files to evaluate
    logger.info("Finding files to evaluate...")
    files = find_files(args.folders, args.include_files, args.skip_files, args.property)

    if not files:
        logger.error("No matching files found. Check your input folders and filters.")
        return 1

    logger.info(f"Found {len(files)} files to evaluate.")

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
        result = evaluate_file(file, args.output, args.verbose)
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
        logger.warning("Try checking:")
        logger.warning("1. Are QID files present in the input directories?")
        logger.warning("2. Do QID files have 'qid' in their filenames?")
        logger.warning("3. Are you filtering out QID files with --skip-files or --include-files?")

    # Generate structured reports for LaTeX tables
    logger.info("Generating reports...")
    nil_f1_report = generate_table_report(results, args.report_output, "NIL")
    qid_f1_report = generate_table_report(results, args.report_output, "QID")

    logger.info("All reports generated:")
    for entity_type in ["nil", "qid"]:
        for metric in ["precision", "recall", "f1"]:
            report_path = os.path.join(args.report_output, f"table_report_{entity_type}_{metric}.csv")
            if os.path.exists(report_path):
                logger.info(f"  - {os.path.basename(report_path)}")

    # Generate combined reports with all metrics
    try:
        from generate_combined_reports import generate_combined_report, generate_latex_table
        from generate_latex_tables import regenerate_tables

        logger.info("Generating combined reports with F1, P, R metrics...")
        latex_dir = os.path.join(args.report_output, "../latex_tables")
        os.makedirs(latex_dir, exist_ok=True)

        for entity_type in ["nil", "qid"]:
            # Generate combined report
            combined_path = generate_combined_report(args.report_output, entity_type, args.report_output)

            if combined_path and os.path.exists(combined_path):
                # Generate LaTeX table from combined report
                latex_table = generate_latex_table(combined_path, entity_type)

                # Save LaTeX table
                latex_path = os.path.join(latex_dir, f'{entity_type}_combined_results_table.tex')
                with open(latex_path, 'w') as f:
                    f.write(latex_table)

                logger.info(f"Generated LaTeX table for {entity_type}: {latex_path}")
        
        # Regenerate all tables to ensure they use the updated retriever ordering
        logger.info("Regenerating LaTeX tables with Boyer-M first ordering...")
        regenerate_tables()
    except Exception as e:
        logger.warning(f"Failed to generate combined reports: {e}")
        logger.warning("You can run generate_combined_reports.py manually to create combined reports.")

    logger.info("Evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())