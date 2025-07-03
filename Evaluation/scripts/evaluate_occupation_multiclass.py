#!/usr/bin/env python3
"""
Script for evaluating occupation property as a multi-class classification problem.
This scripts calculates precision, recall, and F1 score for each sample by treating
QIDs as class labels. It generates statistics (mean, median, mode, standard deviation)
grouped by LLM and retriever combination.
"""

import os
import re
import csv
import glob
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Set, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Constants
METRICS = ['precision', 'recall', 'f1']
ENTITY_TYPES = ['NIL', 'QID']
APPROACH_TYPES = ['RAG', 'ZS', 'unknown']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Model and retriever name mappings for standardized display
MODEL_MAPPINGS = {
    'google_gemma-2-27b-it': 'Gemma',
    'microsoft_phi-3-medium-128k-instruct': 'Phi-3',
    'mistralai_mixtral-8x7b-instruct': 'Mixtral',
    'meta-llama_llama-3.3-70b-instruct': 'LLAMA',
    'openai_gpt-4o-mini': 'GPT-4o',
    'qwen_qwen-2.5-72b-instruct': 'Qwen',
    # Boyer-Moore per-model specific mappings (with corrected parsing)
    'meta-llama_llama.3.3-70b-instruct': 'LLAMA',
    'qwen_qwen.2.5-72b-instruct': 'Qwen',
    # Legacy partial name mappings for backward compatibility
    'google_gemma': 'Gemma',
    'microsoft_phi': 'Phi-3',
    'mistralai_mixtral': 'Mixtral',
    'meta-llama_llama': 'LLAMA',
    'openai_gpt': 'GPT-4o',
    'qwen_qwen': 'Qwen',
    'adaptive_threshold': 'Adaptive',
    'boyer_moore': 'Boyer-M',
    'flexible_boyer': 'Flexible-B',
    'adaptive_threshold_voting': 'Adaptive',
    'boyer_moore_voting': 'Boyer-M',
    'flexible_boyer_moore_voting': 'Flexible-B',
    'boyer_moore_per_model': 'Boyer-M',
    'unknown': 'Majority Voting'
}

RETRIEVER_MAPPINGS = {
    'bm25': 'BM25',
    'bge-large': 'BGE',
    'contriever': 'Contr',
    'gtr-large': 'GTR-L',
    'gtr-xl': 'GTR-X',
    'instructor-xl': 'Inst',
    'zero-shot': '-',
    'zs': '-',        # Alias for zero-shot
    'zero_shot': '-', # Another alias
    'none': '-',      # Often used for no retriever (zero-shot)
    'adaptive_threshold': 'Adaptive',
    'boyer_moore': 'Boyer-M',
    'flexible_boyer': 'Flexible-B',
    'adaptive_threshold_voting': 'Adaptive',
    'boyer_moore_voting': 'Boyer-M',
    'flexible_boyer_moore_voting': 'Flexible-B',
    'boyer_moore_per_model': 'Boyer-M',
    'unknown': 'Majority Voting'
}

# Preferred ordering for retrievers (Boyer-M first for consistency)
RETRIEVER_ORDERING = {
    'boyer_moore': -2,               # Boyer-Moore should come first
    'boyer_moore_voting': -2,        # Same position as non-voting version
    'boyer_moore_per_model': -2,     # Same position as other Boyer-Moore versions
    'zero-shot': 0,
    'zs': 0,        # Alias for zero-shot
    'zero_shot': 0, # Another alias
    'none': 0,      # Often used for no retriever (zero-shot)
    'bge-large': 1,
    'bm25': 2,
    'instructor-xl': 3,
    'contriever': 4,
    'gtr-xl': 5,
    'gtr-large': 6,
    'adaptive_threshold': 7,
    'flexible_boyer': 9,
    'adaptive_threshold_voting': 7,  # Same position as non-voting version
    'flexible_boyer_moore_voting': 9  # Same position as non-voting version
}

# Output directories
PROPERTY_NAME = "occupation"
OUTPUT_BASE_DIR = os.path.join(SCRIPT_DIR, "granular_analyses", PROPERTY_NAME)
PROCESSED_DIR = os.path.join(OUTPUT_BASE_DIR, "processed")
REPORTS_DIR = os.path.join(OUTPUT_BASE_DIR, "reports")
LATEX_DIR = os.path.join(OUTPUT_BASE_DIR, "latex_tables")
PLOTS_DIR = os.path.join(OUTPUT_BASE_DIR, "plots")

# Entity type specific directories for RAG
RAG_NIL_DIR = os.path.join(OUTPUT_BASE_DIR, "RAG", "nil")
RAG_QID_DIR = os.path.join(OUTPUT_BASE_DIR, "RAG", "qid")

# Entity type specific directories for ZS
ZS_NIL_DIR = os.path.join(OUTPUT_BASE_DIR, "ZS", "nil")
ZS_QID_DIR = os.path.join(OUTPUT_BASE_DIR, "ZS", "qid")

# Entity type specific directories for unknown
UNKNOWN_NIL_DIR = os.path.join(OUTPUT_BASE_DIR, "unknown", "nil")
UNKNOWN_QID_DIR = os.path.join(OUTPUT_BASE_DIR, "unknown", "qid")

# Create output directories
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(LATEX_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RAG_NIL_DIR, exist_ok=True)
os.makedirs(RAG_QID_DIR, exist_ok=True)
os.makedirs(ZS_NIL_DIR, exist_ok=True)
os.makedirs(ZS_QID_DIR, exist_ok=True)
os.makedirs(UNKNOWN_NIL_DIR, exist_ok=True)
os.makedirs(UNKNOWN_QID_DIR, exist_ok=True)

def parse_qids(qid_str: str) -> Set[str]:
    """
    Parse a string containing QIDs (semicolon or comma separated) into a set of QIDs.
    
    Args:
        qid_str: String containing QIDs (e.g., "Q123;Q456" or "Q123, Q456")
        
    Returns:
        Set of QID strings
    """
    if not qid_str or pd.isna(qid_str) or qid_str == "":
        return set()
    
    # Clean up the string and split by semicolon or comma
    qid_str = str(qid_str).strip()
    if ";" in qid_str:
        qids = [q.strip() for q in qid_str.split(";")]
    elif "," in qid_str:
        qids = [q.strip() for q in qid_str.split(",")]
    else:
        qids = [qid_str]
    
    # Filter out empty strings and return a set
    return {q for q in qids if q and q.strip() and not q.isspace()}

def calculate_metrics(gold_qids: Set[str], linked_qids: Set[str]) -> Dict[str, float]:
    """
    Calculate precision, recall, F1 score, TP, FP, and FN for a single sample.
    
    Args:
        gold_qids: Set of gold standard QIDs
        linked_qids: Set of linked QIDs
        
    Returns:
        Dictionary containing metrics including precision, recall, F1, TP, FP, and FN
    """
    # Calculate intersection and set differences
    # True positives are QIDs that appear in both sets
    true_positives = len(gold_qids.intersection(linked_qids))
    
    # False positives are QIDs in linked_qids that are not in gold_qids
    false_positives = len(linked_qids - gold_qids)
    
    # False negatives are QIDs in gold_qids that are not in linked_qids
    false_negatives = len(gold_qids - linked_qids)
    
    # Precision = TP / (TP + FP)
    precision = true_positives / max(len(linked_qids), 1) if linked_qids else 0.0
    
    # Recall = TP / (TP + FN)
    recall = true_positives / max(len(gold_qids), 1) if gold_qids else 0.0
    
    # F1 = 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': true_positives,
        'FP': false_positives,
        'FN': false_negatives
    }

def process_file(file_path: str) -> pd.DataFrame:
    """
    Process a CSV file containing occupation lookup_tables and calculate metrics for each sample.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with added metrics for each sample
    """
    # Create empty DataFrame as fallback
    df = pd.DataFrame()
    
    try:
        # Extract metadata from file name
        filename = os.path.basename(file_path)
        entity_type = "unknown"
        llm = "unknown"
        retriever = "unknown"
        is_zs = False
        is_unknown = False
        approach_reason = ""
        
        # Extract entity type
        if "_NIL_" in file_path:
            entity_type = "NIL"
        elif "_QID_" in file_path:
            entity_type = "QID"
        
        # Extract retriever and LLM
        # First try Boyer-Moore per-model pattern
        boyer_moore_pattern = r'QA_occupation_\w+_openrouter_([a-zA-Z0-9\-]+_[a-zA-Z0-9\.\-]+)_boyer_moore_per_model'
        boyer_moore_match = re.search(boyer_moore_pattern, filename)
        if boyer_moore_match:
            model_name = boyer_moore_match.group(1)
            # Convert hyphens back to dots for consistency with other model names  
            model_name = model_name.replace('-3-3-', '.3.3-').replace('-2-5-', '.2.5-')
            llm = model_name
            retriever = "boyer_moore_per_model"
        else:
            # Try RAG pattern
            rag_pattern = r'QA_occupation_\w+_output_(\w+[-\w]*)_openrouter_(\w+[-\w]*)_(\w+[-\w]*)'
            rag_match = re.search(rag_pattern, filename)
            if rag_match:
                retriever, llm_provider, llm_model = rag_match.groups()
                llm = f"{llm_provider}_{llm_model}"
            else:
                # Try zero-shot pattern - two possible formats
                zs_pattern = r'QA_occupation_\w+_noctx_openrouter_(\w+[-\w]*)_(\w+[-\w]*)'
                zs_match = re.search(zs_pattern, filename)
                if zs_match:
                    llm_provider, llm_model = zs_match.groups()
                    retriever = "zero-shot"
                    llm = f"{llm_provider}_{llm_model}"
                else:
                    # Alternative zero-shot pattern
                    alt_zs_pattern = r'QA_occupation_(\w+)_noctx_openrouter_(\w+[-\w]*)_(\w+[-\w]*)'
                    alt_zs_match = re.search(alt_zs_pattern, filename)
                    if alt_zs_match:
                        entity_type_match, llm_provider, llm_model = alt_zs_match.groups()
                        if entity_type == "unknown":
                            entity_type = entity_type_match.upper()
                        retriever = "zero-shot"
                        llm = f"{llm_provider}_{llm_model}"
        
        # Skip files with duplicate "_evaluated" suffix
        if "_evaluated_evaluated.csv" in filename:
            print(f"  Skipping duplicate evaluated file: {filename}")
            return df
            
        # Skip any files marked as granular results
        if filename.endswith('_granular.csv'):
            print(f"  Skipping granular file: {filename}")
            return df
            
        # Skip any files containing test indicators
        if "test" in filename.lower() or "_test_" in filename or "_TEST_" in filename:
            print(f"  Skipping test file: {filename}")
            return df
        
        # Create a processed version of the file with metrics
        # Determine approach (RAG, ZS, or unknown)
        approach = "RAG"  # Default
        
        # Special handling for Boyer-Moore per-model files - always treat as RAG
        if "boyer_moore_per_model" in filename:
            approach = "RAG"
            approach_reason = "Boyer-Moore per-model file (RAG approach)"
        # Check if file is from the "unknown" directory
        elif "/unknown/" in file_path:
            is_unknown = True
            approach = "unknown"
            approach_reason = "unknown in file path"
        # Check for combined algorithm patterns characteristic of unknown folder
        elif "_combined_" in file_path:
            is_unknown = True
            approach = "unknown"
            approach_reason = "combined algorithm file"
        # Check for zero-shot conditions
        elif "noctx" in file_path.lower():
            is_zs = True
            approach = "ZS"
            approach_reason = "noctx in filename"
        elif "zs" in file_path.lower():
            is_zs = True
            approach = "ZS"
            approach_reason = "zs in filename"
        elif "zero-shot" in retriever.lower():
            is_zs = True
            approach = "ZS"
            approach_reason = "zero-shot in retriever"
        elif "/ZS/" in file_path:
            is_zs = True
            approach = "ZS"
            approach_reason = "ZS in file path"
        else:
            approach = "RAG"  # Default to RAG if not identified as ZS or unknown
            approach_reason = "default to RAG"
        
        # Set appropriate retriever value based on approach
        if is_zs:
            # Force the retriever value to be "zero-shot" for consistency
            retriever = "zero-shot"
            print(f"  Setting file as zero-shot: {os.path.basename(file_path)} - Reason: {approach_reason}")
        elif is_unknown:
            # For unknown approach, extract algorithm or retriever information
            if "_combined_" in file_path:
                # Extract the algorithm name
                algorithm_match = re.search(r'_combined_(\w+[-\w]*)', os.path.basename(file_path))
                if algorithm_match:
                    retriever = algorithm_match.group(1)
                    print(f"  Setting retriever to algorithm: {retriever}")
            print(f"  Setting file as unknown: {os.path.basename(file_path)} - Reason: {approach_reason}")
        
        # Determine appropriate output directory based on entity type and approach
        if is_zs:
            # Zero-shot directories
            if entity_type.upper() == "NIL":
                entity_output_dir = ZS_NIL_DIR
            elif entity_type.upper() == "QID":
                entity_output_dir = ZS_QID_DIR
            else:
                entity_output_dir = PROCESSED_DIR
                
            # Make sure the ZS directory exists
            os.makedirs(entity_output_dir, exist_ok=True)
            print(f"  Saving ZS file to: {entity_output_dir}")
        elif is_unknown:
            # Unknown directories
            if entity_type.upper() == "NIL":
                entity_output_dir = UNKNOWN_NIL_DIR
            elif entity_type.upper() == "QID":
                entity_output_dir = UNKNOWN_QID_DIR
            else:
                entity_output_dir = PROCESSED_DIR
                
            # Make sure the unknown directory exists
            os.makedirs(entity_output_dir, exist_ok=True)
            print(f"  Saving unknown file to: {entity_output_dir}")
        else:
            # RAG directories
            if entity_type.upper() == "NIL":
                entity_output_dir = RAG_NIL_DIR
            elif entity_type.upper() == "QID":
                entity_output_dir = RAG_QID_DIR
            else:
                entity_output_dir = PROCESSED_DIR
        
        # Define the output path
        output_basename = os.path.basename(file_path).replace('.csv', '_granular.csv')
        output_path = os.path.join(entity_output_dir, output_basename)
        
        # Check if output already exists
        if os.path.exists(output_path):
            print(f"  Using existing processed file: {output_path}")
            return pd.read_csv(output_path)
        
        # QID files need special handling due to format issues
        if "_QID_" in file_path:
            try:
                # Create array to store sample lookup_tables
                sample_data = []
                
                # Open and read the source file 
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Read the first line to get headers
                    header = f.readline().strip().split(',')
                    
                    # Find indices for relevant columns
                    llm_answer_idx = 8  # Default positions
                    qid_gold_true_idx = 11
                    linked_qid_idx = 12
                    
                    # If columns exist in header, get their actual indices
                    if 'llm_answer' in header:
                        llm_answer_idx = header.index('llm_answer')
                    if 'qid_gold_true' in header:
                        qid_gold_true_idx = header.index('qid_gold_true')
                    if 'linked_qid' in header:
                        linked_qid_idx = header.index('linked_qid')
                    
                    # Process rest of file
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= max(llm_answer_idx, qid_gold_true_idx, linked_qid_idx) + 1:
                            # Get values
                            llm_answer = row[llm_answer_idx]
                            qid_gold_true = row[qid_gold_true_idx]
                            linked_qid = row[linked_qid_idx]
                            
                            # Skip rows where gold_qid is empty
                            if not qid_gold_true or pd.isna(qid_gold_true):
                                continue
                            
                            # Calculate metrics
                            gold_qids = parse_qids(qid_gold_true)
                            linked_qids = parse_qids(linked_qid)
                            metrics = calculate_metrics(gold_qids, linked_qids)
                            
                            # Add to sample lookup_tables
                            sample_data.append({
                                'retriever': retriever,
                                'llm': llm,
                                'entity_type': entity_type,
                                'llm_answer': llm_answer,
                                'qid_gold_true': qid_gold_true,
                                'linked_qid': linked_qid,
                                'sample_precision': metrics['precision'],
                                'sample_recall': metrics['recall'],
                                'sample_f1': metrics['f1'],
                                'TP': metrics['TP'],
                                'FP': metrics['FP'],
                                'FN': metrics['FN']
                            })
                
                # Create DataFrame
                df = pd.DataFrame(sample_data)
                
                # Add approach field to dataframe
                df['approach'] = approach
                
                # Map 'unknown' approach to 'Majority Voting' for better labeling
                if approach == 'unknown':
                    df['approach_display'] = 'Majority Voting'
                else:
                    df['approach_display'] = approach
                
                # Save processed file
                df.to_csv(output_path, index=False)
                
            except Exception as e:
                print(f"Error processing QID file {file_path}: {e}")
                return pd.DataFrame()
        else:
            # For regular files, try standard reading
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                try:
                    # If that fails, try with a more robust approach
                    # Read with low_memory=False and on_bad_lines='skip'
                    df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')
                except Exception as inner_e:
                    print(f"Error processing file {file_path}: {inner_e}")
                    return pd.DataFrame()
            
            # Ensure we have the necessary columns
            if 'retriever' not in df.columns:
                df['retriever'] = retriever
            else:
                # For Boyer-Moore per-model files, ensure retriever is set correctly
                if "boyer_moore_per_model" in filename and (df['retriever'].isna().all() or df['retriever'].fillna('').eq('').all()):
                    df['retriever'] = retriever
                    print(f"  Set retriever to '{retriever}' for Boyer-Moore per-model file")
            if 'llm' not in df.columns:
                df['llm'] = llm
            if 'entity_type' not in df.columns:
                df['entity_type'] = entity_type
            
            # Calculate metrics for each row
            metrics_data = {
                'sample_precision': [],
                'sample_recall': [],
                'sample_f1': [],
                'TP': [],
                'FP': [],
                'FN': []
            }
            
            for _, row in df.iterrows():
                gold_qids = parse_qids(row.get('qid_gold_true', ''))
                linked_qids = parse_qids(row.get('linked_qid', ''))
                
                # Skip rows without proper QID information for both entity types
                # Change: Apply empty gold_qids check to both QID and NIL entity types
                if not gold_qids:
                    # Add NaN values to keep row alignment
                    for metric in METRICS:
                        metrics_data[f'sample_{metric}'].append(np.nan)
                    metrics_data['TP'].append(0)
                    metrics_data['FP'].append(0)
                    metrics_data['FN'].append(0)
                    continue
                    
                # Calculate metrics
                metrics = calculate_metrics(gold_qids, linked_qids)
                
                # Add to metrics lookup_tables
                for metric in METRICS:
                    metrics_data[f'sample_{metric}'].append(metrics[metric])
                
                # Add TP, FP, FN counts
                metrics_data['TP'].append(metrics['TP'])
                metrics_data['FP'].append(metrics['FP'])
                metrics_data['FN'].append(metrics['FN'])
            
            # Add metrics to DataFrame
            for metric in METRICS:
                df[f'sample_{metric}'] = metrics_data[f'sample_{metric}']
            
            # Add TP, FP, FN columns
            df['TP'] = metrics_data['TP']
            df['FP'] = metrics_data['FP']
            df['FN'] = metrics_data['FN']
            
            # Add approach field to dataframe
            df['approach'] = approach
            
            # Map 'unknown' approach to 'Majority Voting' for better labeling
            if approach == 'unknown':
                df['approach_display'] = 'Majority Voting'
            else:
                df['approach_display'] = approach
            
            # Save the processed file
            df.to_csv(output_path, index=False)
        
        # Print key info before returning
        print(f"  Processed file: {os.path.basename(file_path)}")
        print(f"    Entity type: {entity_type}")
        print(f"    Approach: {approach}")
        print(f"    Retriever: {retriever}")
        print(f"    Model: {llm}")
        print(f"    Is zero-shot: {is_zs}")
        print(f"    Is unknown: {is_unknown}")
        print(f"    Output file: {output_path}")
        
        if not df.empty:
            print(f"    DataFrame size: {len(df)} rows")
            if 'sample_precision' in df.columns:
                avg_precision = df['sample_precision'].mean() if not df['sample_precision'].empty else 0
                avg_recall = df['sample_recall'].mean() if not df['sample_recall'].empty else 0
                avg_f1 = df['sample_f1'].mean() if not df['sample_f1'].empty else 0
                print(f"    Metrics - P: {avg_precision:.3f}, R: {avg_recall:.3f}, F1: {avg_f1:.3f}")
        
        return df
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return pd.DataFrame()

def process_zs_files(entity_type: str) -> pd.DataFrame:
    """
    Directly process zero-shot files from the ZS directory to ensure they're included.
    
    Args:
        entity_type: Entity type (NIL or QID)
        
    Returns:
        DataFrame with statistics for ZS experiments
    """
    # Path to ZS directory
    zs_dir = os.path.join(SCRIPT_DIR, 'granular_analyses', 'occupation', 'ZS', entity_type.lower())
    
    # Check if directory exists
    if not os.path.exists(zs_dir):
        print(f"ZS directory not found: {zs_dir}")
        return pd.DataFrame()
    
    # Find all granular files
    granular_files = glob.glob(os.path.join(zs_dir, "*_granular.csv"))
    
    if not granular_files:
        print(f"No granular files found in ZS directory: {zs_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(granular_files)} ZS granular files")
    
    # Process each file
    zs_dfs = []
    for file_path in granular_files:
        try:
            df = pd.read_csv(file_path)
            # Make sure retriever is set to zero-shot
            df['retriever'] = 'zero-shot'
            # Extract model name from filename
            filename = os.path.basename(file_path)
            model_match = re.search(r'openrouter_(\w+[-\w]*)_(\w+[-\w]*)', filename)
            if model_match:
                llm_provider, llm_model = model_match.groups()
                model_name = f"{llm_provider}_{llm_model}"
                # Normalize the model name
                df['llm'] = model_name
            
            zs_dfs.append(df)
        except Exception as e:
            print(f"Error processing ZS file {file_path}: {e}")
    
    if not zs_dfs:
        print(f"No valid ZS files processed")
        return pd.DataFrame()
    
    # Combine all ZS dataframes
    combined_zs_df = pd.concat(zs_dfs, ignore_index=True)
    return combined_zs_df

def calculate_statistics(df: pd.DataFrame, force_zs: bool = False, entity_type: str = "NIL") -> pd.DataFrame:
    """
    Calculate statistics (mean, median, mode, std) for each LLM and retriever combination.
    
    Args:
        df: DataFrame containing sample metrics
        force_zs: Whether to force inclusion of zero-shot results
        entity_type: Entity type (NIL or QID)
        
    Returns:
        DataFrame with statistics for each LLM and retriever combination
    """
    stats_data = []
    
    # Make sure all zero-shot retrievers are consistently named
    if 'retriever' in df.columns:
        # Normalize all zero-shot related retriever names
        zero_shot_patterns = ['zero-shot', 'zs', 'zero_shot', 'none']
        for pattern in zero_shot_patterns:
            if isinstance(df['retriever'], pd.Series):  # Check if it's a Series before using str accessor
                df.loc[df['retriever'].str.lower() == pattern.lower(), 'retriever'] = 'zero-shot'
    
    # Print unique retrievers found in the dataframe for debugging
    print(f"Found retrievers in dataset: {df['retriever'].unique()}")
    print(f"Found models in dataset: {df['llm'].unique()}")
    
    # Additional debug info about ZS experiments
    zs_df = df[df['retriever'] == 'zero-shot']
    if not zs_df.empty:
        print(f"Found {len(zs_df)} zero-shot rows with models: {zs_df['llm'].unique()}")
    elif force_zs:
        # If no ZS rows found and force_zs is True, process ZS files directly
        print("No ZS lookup_tables found in combined dataframe, attempting to process ZS files directly...")
        zs_df = process_zs_files(entity_type)
        if not zs_df.empty:
            # Add ZS lookup_tables to main dataframe
            df = pd.concat([df, zs_df], ignore_index=True)
            print(f"Added {len(zs_df)} ZS rows to dataframe")
    
    # Group by LLM and retriever
    grouped = df.groupby(['llm', 'retriever'])
    
    for (llm, retriever), group in grouped:
        row_data = {
            'llm': llm,
            'retriever': retriever
        }
        
        # Calculate statistics for each metric
        for metric in METRICS:
            values = group[f'sample_{metric}'].dropna().values
            
            if len(values) > 0:
                # Mean
                row_data[f'{metric}_mean'] = np.mean(values)
                
                # Median
                row_data[f'{metric}_median'] = np.median(values)
                
                # Mode (most common value)
                # Round to 3 decimal places to avoid too many unique values
                rounded_values = np.round(values, 3)
                try:
                    mode_result = stats.mode(rounded_values)
                    # Handle different scipy versions
                    if hasattr(mode_result, 'mode'):
                        if isinstance(mode_result.mode, np.ndarray):
                            row_data[f'{metric}_mode'] = mode_result.mode[0] if mode_result.mode.size > 0 else np.nan
                        else:
                            row_data[f'{metric}_mode'] = mode_result.mode
                    else:
                        # Older scipy version returns a tuple
                        row_data[f'{metric}_mode'] = mode_result[0][0] if len(mode_result[0]) > 0 else np.nan
                except Exception as e:
                    print(f"Warning: Error calculating mode for {metric}: {e}")
                    row_data[f'{metric}_mode'] = np.nan
                
                # Standard deviation
                row_data[f'{metric}_std'] = np.std(values)
            else:
                # No values available
                for stat in ['mean', 'median', 'mode', 'std']:
                    row_data[f'{metric}_{stat}'] = np.nan
        
        stats_data.append(row_data)
    
    # Create DataFrame from statistics
    stats_df = pd.DataFrame(stats_data)
    
    # ---- ADD MAJORITY VOTING ENTRIES EXPLICITLY ----
    # First, check for any missing algorithm entries from the 'unknown' directory
    try:
        # Check directory paths for unknown/Majority Voting entries
        unknown_data_dir = os.path.join(OUTPUT_BASE_DIR, 'unknown', entity_type.lower())
        if os.path.exists(unknown_data_dir):
            print(f"Found unknown directory: {unknown_data_dir}")
            
            # Known algorithm patterns to look for
            algorithms = ['adaptive_threshold', 'boyer_moore', 'flexible_boyer']
            
            # Find all CSV files in the directory
            unknown_files = glob.glob(os.path.join(unknown_data_dir, '*.csv'))
            print(f"Found {len(unknown_files)} files in unknown directory")
            
            # Process each algorithm
            for algorithm in algorithms:
                # Display names for algorithms
                display_name = algorithm
                if algorithm == 'adaptive_threshold':
                    display_name = 'Adaptive Threshold'
                elif algorithm == 'boyer_moore':
                    display_name = 'Boyer-Moore'
                elif algorithm == 'flexible_boyer':
                    display_name = 'Flexible Boyer-Moore'
                    
                # Explicitly check if this specific algorithm with this specific display name exists
                # We want to add the entry regardless of what exists if it's not already in the dataframe
                # with exactly the display name we want
                algorithm_exists = False
                if not stats_df.empty and 'retriever' in stats_df.columns:
                    algorithm_exists = ((stats_df['llm'] == 'Majority Voting') & 
                                       (stats_df['retriever'] == display_name)).any()
                
                if not algorithm_exists:
                    print(f"Looking for algorithm files for {algorithm}")
                    # Find matching files (improved pattern matching)
                    matching_files = [f for f in unknown_files if algorithm.lower() in os.path.basename(f).lower()]
                    print(f"Found {len(matching_files)} matching files for {algorithm}")
                    
                    # Process files if any found
                    for file_path in matching_files:
                        try:
                            print(f"Processing file: {os.path.basename(file_path)}")
                            # Try to load the file
                            df_algo = pd.read_csv(file_path)
                            
                            # Calculate metrics if the file has the required columns
                            if 'sample_precision' in df_algo.columns:
                                print(f"File {os.path.basename(file_path)} has required metrics columns")
                                row_data = {
                                    'llm': 'Majority Voting',
                                    'retriever': display_name  # Use display name, not the raw algorithm name
                                }
                                
                                # Add metrics
                                for metric in METRICS:
                                    values = df_algo[f'sample_{metric}'].dropna().values
                                    if len(values) > 0:
                                        print(f"  Processing {metric} with {len(values)} values")
                                        row_data[f'{metric}_mean'] = np.mean(values)
                                        row_data[f'{metric}_median'] = np.median(values)
                                        try:
                                            # Handle mode calculation safely
                                            rounded_values = np.round(values, 3)
                                            mode_result = stats.mode(rounded_values)
                                            
                                            # More robust handling of mode result based on scipy version
                                            if hasattr(mode_result, 'mode'):
                                                # New scipy version
                                                if isinstance(mode_result.mode, np.ndarray) and mode_result.mode.size > 0:
                                                    row_data[f'{metric}_mode'] = mode_result.mode[0]
                                                else:
                                                    row_data[f'{metric}_mode'] = mode_result.mode if not np.isscalar(mode_result.mode) or not np.isnan(mode_result.mode) else 0.0
                                            else:
                                                # Old scipy version
                                                if isinstance(mode_result[0], np.ndarray) and mode_result[0].size > 0:
                                                    row_data[f'{metric}_mode'] = mode_result[0][0]
                                                else:
                                                    row_data[f'{metric}_mode'] = mode_result[0] if hasattr(mode_result, '__getitem__') and len(mode_result) > 0 else 0.0
                                        except Exception as e:
                                            # If any error in mode calculation, set to 0.0 (not NaN) for Majority Voting
                                            print(f"  Error calculating mode for {metric}: {e}")
                                            row_data[f'{metric}_mode'] = 0.0
                                        
                                        # Calculate standard deviation
                                        row_data[f'{metric}_std'] = np.std(values)
                                    else:
                                        # No values - set to 0 not NaN for consistency
                                        print(f"  No values for {metric}, setting to 0.0")
                                        for stat in ['mean', 'median', 'mode', 'std']:
                                            row_data[f'{metric}_{stat}'] = 0.0
                                
                                # Print metrics for debugging
                                print(f"  Calculated metrics for {display_name}:")
                                for metric in METRICS:
                                    print(f"    {metric}: mean={row_data[f'{metric}_mean']:.4f}, "
                                         f"median={row_data[f'{metric}_median']:.4f}, "
                                         f"mode={row_data[f'{metric}_mode']}, "
                                         f"std={row_data[f'{metric}_std']:.4f}")
                                
                                # Add row to stats_df at the beginning (top)
                                new_row_df = pd.DataFrame([row_data])
                                # Concatenate with new row first to ensure it appears at the top
                                stats_df = pd.concat([new_row_df, stats_df], ignore_index=True)
                                print(f"Added Majority Voting entry for {display_name}")
                                
                                # One file per algorithm is sufficient
                                break
                        except Exception as e:
                            print(f"Error processing algorithm file {file_path}: {e}")
            
            # Also look for combined algorithm files in backup locations if no entries were added
            print("Checking for Majority Voting entries in the dataframe")
            if not ((stats_df['llm'] == 'Majority Voting') & 
                     (stats_df['retriever'].isin(['Adaptive Threshold', 'Boyer-Moore', 'Flexible Boyer-Moore']))).any():
                print("No Majority Voting entries found, checking backup locations")
                
                # Check in the main scripts directory
                backup_dirs = [
                    os.path.join(ROOT_DIR, 'scripts', 'linking', 'output'),
                    os.path.join(ROOT_DIR, 'scripts', 'boyer-moore_majority_voting', 'output'),
                    os.path.join(ROOT_DIR, 'lookup_tables', 'linked_evaluated_data', 'occupation', 'unknown', entity_type.lower())
                ]
                
                for backup_dir in backup_dirs:
                    if os.path.exists(backup_dir):
                        print(f"Checking backup directory: {backup_dir}")
                        # Look for all CSV files
                        backup_files = glob.glob(os.path.join(backup_dir, '**', '*.csv'), recursive=True)
                        print(f"Found {len(backup_files)} backup files")
                        
                        # Try to match algorithm patterns
                        for algorithm in algorithms:
                            display_name = RETRIEVER_MAPPINGS.get(algorithm, algorithm)
                            if not ((stats_df['llm'] == 'Majority Voting') & (stats_df['retriever'] == display_name)).any():
                                # Find matching files
                                matching_files = [f for f in backup_files if algorithm.lower() in os.path.basename(f).lower()]
                                if matching_files:
                                    print(f"Found {len(matching_files)} backup files for {algorithm}")
                                    # Process the first matching file
                                    try:
                                        df_algo = pd.read_csv(matching_files[0])
                                        if 'sample_precision' in df_algo.columns or 'precision' in df_algo.columns:
                                            # Create row lookup_tables with default values
                                            row_data = {
                                                'llm': 'Majority Voting',
                                                'retriever': display_name
                                            }
                                            
                                            # Add default metrics as fallback
                                            for metric in METRICS:
                                                # Try both column naming patterns
                                                column_name = f'sample_{metric}' if f'sample_{metric}' in df_algo.columns else metric
                                                
                                                if column_name in df_algo.columns:
                                                    values = df_algo[column_name].dropna().values
                                                    if len(values) > 0:
                                                        row_data[f'{metric}_mean'] = np.mean(values)
                                                        row_data[f'{metric}_median'] = np.median(values)
                                                        row_data[f'{metric}_mode'] = 0.0  # Default to 0 to avoid issues
                                                        row_data[f'{metric}_std'] = np.std(values) 
                                                    else:
                                                        # Set default values if no lookup_tables
                                                        for stat in ['mean', 'median', 'mode', 'std']:
                                                            row_data[f'{metric}_{stat}'] = 0.0
                                                else:
                                                    # If column doesn't exist, set default values
                                                    for stat in ['mean', 'median', 'mode', 'std']:
                                                        row_data[f'{metric}_{stat}'] = 0.0
                                            
                                            # Add row to stats_df at the beginning
                                            stats_df = pd.concat([pd.DataFrame([row_data]), stats_df], ignore_index=True)
                                            print(f"Added backup Majority Voting entry for {display_name}")
                                    except Exception as e:
                                        print(f"Error processing backup file {matching_files[0]}: {e}")
    except Exception as e:
        print(f"Error adding Majority Voting entries: {e}")
        import traceback
        traceback.print_exc()
    
    # Only ensure Boyer-Moore Majority Voting entry is present if it exists in the lookup_tables
    # Don't create default entries for adaptive_threshold and flexible_boyer as they're not wanted
    for algorithm, display_name in [('boyer_moore', 'Boyer-Moore')]:
        if not ((stats_df['llm'] == 'Majority Voting') & (stats_df['retriever'] == display_name)).any():
            print(f"Note: No {display_name} majority voting lookup_tables found - this is expected if no such files exist")
    
    # First, remove any duplicate entries
    stats_df = stats_df.drop_duplicates(subset=['llm', 'retriever'], keep='first')
    
    # Remove any unwanted algorithm entries that might have gotten through
    stats_df = stats_df[~stats_df['retriever'].isin(['Adaptive Threshold', 'Flexible Boyer-Moore'])]
    
    # Check for Majority Voting entries and ensure they're only present once
    majority_entries = stats_df[stats_df['llm'] == 'Majority Voting'].copy()
    
    # If Majority Voting entries exist, remove them from the main DataFrame and
    # we'll add them back at the top afterward
    if not majority_entries.empty:
        # Remove all Majority Voting entries from the main DataFrame
        stats_df = stats_df[stats_df['llm'] != 'Majority Voting'].copy()
        
        # Ensure the majority entries are unique
        majority_entries = majority_entries.drop_duplicates(subset=['retriever'], keep='first')
        
    # Move Majority Voting entries to the top of the DataFrame
    majority_df = majority_entries if not majority_entries.empty else pd.DataFrame()
    non_majority_df = stats_df.copy()
    
    # Sort non-majority entries
    non_majority_df = non_majority_df.sort_values(['llm', 'retriever'])
    
    # Sort majority entries by retriever (to ensure consistent ordering)
    majority_df = majority_df.sort_values(['retriever'])
    
    # Combine with majority entries first
    stats_df = pd.concat([majority_df, non_majority_df], ignore_index=True)
    
    return stats_df

def normalize_model_name(model_str):
    """Normalize model name using the MODEL_MAPPINGS."""
    if model_str is None or not isinstance(model_str, str):
        return "unknown"
        
    for key, value in MODEL_MAPPINGS.items():
        if key in model_str:
            return value
    return model_str


def format_retriever_name(retriever):
    """Format retriever name for reporting."""
    if retriever is None or not isinstance(retriever, str):
        return "unknown"
        
    # Special handling for zero-shot
    if retriever.lower() in ["zero-shot", "zs", "zero_shot", "none"]:
        return RETRIEVER_MAPPINGS["zero-shot"]
        
    if retriever in RETRIEVER_MAPPINGS:
        return RETRIEVER_MAPPINGS[retriever]
    return retriever


def generate_latex_table(stats_df: pd.DataFrame, entity_type: str):
    """
    Generate a LaTeX table from the statistics DataFrame.
    This function produces a LaTeX table with vertical separators between column groups,
    with ZS runs coming before other runs and the highest values in bold per column.
    
    Args:
        stats_df: DataFrame containing statistics
        entity_type: Entity type (NIL or QID)
    """
    # Filter out unwanted algorithm entries and remove duplicates
    filtered_df = stats_df.copy()
    filtered_df = filtered_df[~filtered_df['retriever'].isin(['Adaptive Threshold', 'Flexible Boyer-Moore'])]
    filtered_df = filtered_df.drop_duplicates(subset=['llm', 'retriever'], keep='first')
    
    # Use filtered dataframe for the rest of the function
    stats_df = filtered_df
    
    # Create LaTeX table content with maximum compactness
    latex_content = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\tiny", # Use tiny for maximum compactness
        "  \\setlength{\\tabcolsep}{1pt}", # Minimal column spacing
        "  \\renewcommand{\\arraystretch}{0.9}", # Reduce row height
        "  \\begin{tabular}{ll|ccc|ccc|ccc}"
    ]
    
    # Add header rows with more compact column names
    latex_content.extend([
        "    \\toprule",
        "     &  & \\multicolumn{3}{c|}{Precision} & \\multicolumn{3}{c|}{Recall} & \\multicolumn{3}{c}{F1} \\\\",
        "    Model & Retr. & M & Med & SD & M & Med & SD & M & Med & SD \\\\",
        "    \\midrule"
    ])
    
    # Group by model name after normalization to ensure proper grouping
    stats_df['normalized_model'] = stats_df['llm'].apply(normalize_model_name)
    
    # Get unique normalized model names
    model_names = stats_df['normalized_model'].unique()
    
    # Define standard retrievers in consistent order (Boyer-M first)
    standard_retrievers = ['boyer_moore', 'boyer_moore_voting', 'boyer_moore_per_model', 'bge-large', 'bm25', 'instructor-xl', 'contriever', 'gtr-xl', 'gtr-large']
    zero_shot_retrievers = ['zero-shot', 'zs', 'zero_shot', 'none']
    
    # Find maximum values for each column (mean, median, stddev) for each metric
    max_values = {}
    for metric in METRICS:
        # Find max for mean
        max_values[f'{metric}_mean'] = stats_df[f'{metric}_mean'].max()
        # Find max for median
        max_values[f'{metric}_median'] = stats_df[f'{metric}_median'].max()
        # Find max for standard deviation
        max_values[f'{metric}_std'] = stats_df[f'{metric}_std'].max()
    
    # Process each model group
    for i, model_name in enumerate(model_names):
        # Get all rows for this model
        model_rows = stats_df[stats_df['normalized_model'] == model_name].copy()
        
        # Keep track of seen retrievers to avoid duplicates (like multiple GTR-X entries)
        seen_retrievers = set()
        
        # Separate rows into ZS and non-ZS for ordering
        zs_mask = model_rows['retriever'].isin(zero_shot_retrievers)
        zs_rows = model_rows[zs_mask].copy()
        non_zs_rows = model_rows[~zs_mask].copy()
        
        # Deduplicate retrievers if needed
        if len(non_zs_rows) > len(standard_retrievers):
            # Keep track of unique retrievers
            unique_retrievers = []
            duplicate_indices = []
            
            for idx, retriever in enumerate(non_zs_rows['retriever'].values):
                normalized_retriever = retriever.lower()
                if normalized_retriever in seen_retrievers:
                    duplicate_indices.append(idx)
                else:
                    seen_retrievers.add(normalized_retriever)
                    unique_retrievers.append(retriever)
            
            # Filter out duplicates
            if duplicate_indices:
                print(f"Warning: Removing {len(duplicate_indices)} duplicate retrievers for model {model_name}")
                non_zs_rows = non_zs_rows.iloc[[i for i in range(len(non_zs_rows)) if i not in duplicate_indices]]
        
        # Sort non-zero-shot rows by standard retriever order
        non_zs_rows['retriever_order'] = non_zs_rows['retriever'].apply(
            lambda x: standard_retrievers.index(x) if x in standard_retrievers else 999
        )
        non_zs_rows = non_zs_rows.sort_values('retriever_order')
        
        # Calculate total rows for multirow command
        total_rows = len(non_zs_rows) + len(zs_rows)
        first_row = True
        
        # Build the rows for this model
        model_rows_latex = []
        
        # Process zero-shot rows FIRST (this is the key change)
        if not zs_rows.empty:
            # Just use one ZS row (in case there are duplicates)
            zs_row = zs_rows.iloc[0]
            
            retriever_name = "-"  # Use consistent name for zero-shot
            
            # Format model column with multirow for the first row of the model group
            if first_row:
                row_start = f"    \\multirow{{{total_rows}}}{{*}}{{{model_name}}} & {retriever_name}"
                first_row = False
            else:
                row_start = f"     & {retriever_name}"
            
            # Values to collect for the row
            values = []
            
            # Process each metric group (precision, recall, f1)
            for metric_group in ['precision', 'recall', 'f1']:
                mean_val = zs_row[f'{metric_group}_mean']
                median_val = zs_row[f'{metric_group}_median']
                std_val = zs_row[f'{metric_group}_std']
                
                # Format Mean value with bold if it's the maximum
                mean_formatted = f"{mean_val:.2f}"
                if pd.isna(mean_val):
                    mean_formatted = "--"
                elif abs(mean_val - max_values[f'{metric_group}_mean']) < 0.001:
                    mean_formatted = f"\\textbf{{{mean_formatted}}}"
                
                # Format Median value with bold if it's the maximum
                median_formatted = f"{median_val:.2f}"
                if pd.isna(median_val):
                    median_formatted = "--"
                elif abs(median_val - max_values[f'{metric_group}_median']) < 0.001 and median_val > 0:
                    median_formatted = f"\\textbf{{{median_formatted}}}"
                
                # Format StdDev value (no bold for standard deviation)
                std_formatted = f"{std_val:.2f}"
                if pd.isna(std_val):
                    std_formatted = "--"
                
                # Add formatted values to the row
                values.append(mean_formatted)
                values.append(median_formatted)
                values.append(std_formatted)
            
            # Add the row to the table
            model_rows_latex.append(f"{row_start} & " + " & ".join(values) + " \\\\")
        
        # Process non-zero-shot rows AFTER zero-shot rows (no midrule between them)
        if not non_zs_rows.empty:
            for _, row in non_zs_rows.iterrows():
                retriever_name = format_retriever_name(row['retriever'])
                
                # Format model column with multirow for the first row of the model group
                if first_row:
                    # Escape special characters in model_name
                    escaped_model_name = model_name.replace("_", "\\_").replace("&", "\\&").replace("$", "\\$").replace("#", "\\#")
                    row_start = f"    \\multirow{{{total_rows}}}{{*}}{{{escaped_model_name}}} & {retriever_name}"
                    first_row = False
                else:
                    row_start = f"     & {retriever_name}"
                
                # Values to collect for the row
                values = []
                
                # Process each metric group (precision, recall, f1)
                for metric_group in ['precision', 'recall', 'f1']:
                    mean_val = row[f'{metric_group}_mean']
                    median_val = row[f'{metric_group}_median']
                    std_val = row[f'{metric_group}_std']
                    
                    # Format Mean value with bold if it's the maximum
                    mean_formatted = f"{mean_val:.2f}"
                    if pd.isna(mean_val):
                        mean_formatted = "--"
                    elif abs(mean_val - max_values[f'{metric_group}_mean']) < 0.001:
                        mean_formatted = f"\\textbf{{{mean_formatted}}}"
                    
                    # Format Median value with bold if it's the maximum
                    median_formatted = f"{median_val:.2f}"
                    if pd.isna(median_val):
                        median_formatted = "--"
                    elif abs(median_val - max_values[f'{metric_group}_median']) < 0.001 and median_val > 0:
                        median_formatted = f"\\textbf{{{median_formatted}}}"
                    
                    # Format StdDev value (no bold for standard deviation)
                    std_formatted = f"{std_val:.2f}"
                    if pd.isna(std_val):
                        std_formatted = "--"
                    
                    # Add formatted values to the row
                    values.append(mean_formatted)
                    values.append(median_formatted)
                    values.append(std_formatted)
                
                # Add the row to the table
                model_rows_latex.append(f"{row_start} & " + " & ".join(values) + " \\\\")
        
        # Add all rows for this model to the latex content
        latex_content.extend(model_rows_latex)
        
        # Add separator between models except for the last one
        if i < len(model_names) - 1:
            latex_content.append("    \\midrule")
    
    # Complete the table
    latex_content.extend([
        "    \\bottomrule",
        "  \\end{tabular}"
    ])
    
    # Add caption and label
    caption = "Occupation performance metrics for "
    if entity_type.upper() == "QID":
        caption += "QID identification."
    else:
        caption += "NIL identification."
    
    # Escape special characters in the caption
    caption = caption.replace("_", "\\_").replace("&", "\\&").replace("$", "\\$").replace("#", "\\#")
        
    latex_content.append(f"  \\caption{{{caption}}}")
    latex_content.append(f"  \\label{{tab:occupation_{entity_type.lower()}_multiclass}}")
    latex_content.append("\\end{table}")
    
    # Convert to string and write to file
    latex_string = "\n".join(latex_content)
    output_file = os.path.join(LATEX_DIR, f"{entity_type.lower()}_occupation_multiclass_table.tex")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write the file with explicit UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_string)
    
    print(f"Generated LaTeX table: {output_file}")
    return latex_string

def generate_csv_report(stats_df: pd.DataFrame, entity_type: str):
    """
    Generate a CSV report from the statistics DataFrame.
    
    Args:
        stats_df: DataFrame containing statistics
        entity_type: Entity type (NIL or QID)
    """
    # Create a copy of the DataFrame for normalization
    csv_df = stats_df.copy()
    
    # Filter out unwanted algorithm entries (Adaptive Threshold and Flexible Boyer-Moore)
    csv_df = csv_df[~csv_df['retriever'].isin(['Adaptive Threshold', 'Flexible Boyer-Moore'])]
    
    # Remove duplicate entries (keep first occurrence)
    csv_df = csv_df.drop_duplicates(subset=['llm', 'retriever'], keep='first')
    
    # Apply model name normalization for CSV output
    csv_df['llm'] = csv_df['llm'].apply(normalize_model_name)
    
    # Apply retriever name formatting for CSV output
    csv_df['retriever'] = csv_df['retriever'].apply(format_retriever_name)
    
    # Create summary directory if it doesn't exist
    summary_dir = os.path.join(REPORTS_DIR, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Save to CSV
    output_file = os.path.join(summary_dir, f"{entity_type.lower()}_occupation_multiclass_report.csv")
    csv_df.to_csv(output_file, index=False)
    
    print(f"Generated CSV report: {output_file}")

def plot_metrics_heatmap(stats_df: pd.DataFrame, entity_type: str):
    """
    Generate heatmaps for precision, recall, and F1 score.
    
    Args:
        stats_df: DataFrame containing statistics
        entity_type: Entity type (NIL or QID)
    """
    # Define retriever order
    retriever_order = ['bm25', 'bge-large', 'contriever', 'gtr-large', 'gtr-xl', 'instructor-xl', 'zero-shot', 'adaptive_threshold', 'boyer_moore', 'flexible_boyer', 'adaptive_threshold_voting', 'boyer_moore_voting', 'flexible_boyer_moore_voting', 'unknown', 'Majority Voting']
    
    # Create a copy of the dataframe to avoid modifying the original
    plot_df = stats_df.copy()
    
    # Replace 'unknown' with 'Majority Voting' for display
    if 'retriever' in plot_df.columns:
        plot_df.loc[plot_df['retriever'] == 'unknown', 'retriever'] = 'Majority Voting'
    
    # Pivot lookup_tables for each metric
    for metric in METRICS:
        # Pivot table with llm as rows and retriever as columns
        pivot_data = plot_df.pivot_table(
            index='llm', 
            columns='retriever', 
            values=f'{metric}_mean',
            aggfunc='first'
        )
        
        # Reorder columns according to retriever_order
        columns = [col for col in retriever_order if col in pivot_data.columns]
        # If we have both 'unknown' and 'Majority Voting' in columns, keep only 'Majority Voting'
        if 'unknown' in columns and 'Majority Voting' in columns:
            columns.remove('unknown')
        pivot_data = pivot_data[columns]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        ax = sns.heatmap(
            pivot_data, 
            annot=True, 
            cmap='YlGnBu', 
            fmt='.3f',
            linewidths=0.5,
            vmin=0, 
            vmax=1
        )
        
        # Set title and labels
        plt.title(f'Occupation {entity_type} - Mean {metric.capitalize()} by LLM and Retriever')
        plt.xlabel('Retriever')
        plt.ylabel('LLM')
        
        # Save figure
        output_file = os.path.join(PLOTS_DIR, f"{entity_type.lower()}_occupation_{metric}_heatmap.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        print(f"Generated heatmap for {metric}: {output_file}")

def main():
    """Main function to evaluate occupation property as a multi-class classification problem."""
    parser = argparse.ArgumentParser(description='Evaluate occupation property as a multi-class classification problem.')
    parser.add_argument('--entity-type', choices=ENTITY_TYPES, default=None, help='Entity type to evaluate (NIL or QID)')
    parser.add_argument('--force-zs', action='store_true', help='Force inclusion of zero-shot results')
    parser.add_argument('--exclude-unknown', action='store_true', help='Exclude files from the unknown directory')
    parser.add_argument('--exclude-voting', action='store_true', help='Exclude voting algorithm results from the /scripts/linking/output directory (voting results are included by default)')
    
    # Print information about the command-line arguments
    print("Command-line arguments:")
    print("  --entity-type: Specify which entity type to evaluate (NIL or QID)")
    print("  --force-zs: Force inclusion of zero-shot results")
    print("  --exclude-unknown: Exclude files from the unknown directory")
    print("  --exclude-voting: Exclude voting algorithm results from the linking/output directory (included by default)")
    
    args = parser.parse_args()
    
    # Process each entity type or the specified one
    entity_types = [args.entity_type] if args.entity_type else ENTITY_TYPES
    force_zs = args.force_zs
    include_unknown = not args.exclude_unknown  # Default to including unknown files
    
    if include_unknown:
        print("Including files from the 'unknown' directory (default)")
    else:
        print("Not including files from the 'unknown' directory (--exclude-unknown was specified)")
    
    for entity_type in entity_types:
        # Find all CSV files for the current entity type
        
        # IMPORTANT: The occupation directory structure is organization:
        # scripts/evaluation/occupation/{RAG or ZS}/{nil or qid}/*.csv
        
        # Define the most common path pattern first - we know this exists from directory checking
        occupation_rag_dir = os.path.join(SCRIPT_DIR, 'occupation', 'RAG', entity_type.lower())
        occupation_zs_dir = os.path.join(SCRIPT_DIR, 'occupation', 'ZS', entity_type.lower())
        
        # Fallback patterns if the primary ones don't exist
        rag_dir = os.path.join(SCRIPT_DIR, entity_type.lower(), 'RAG', entity_type.lower())
        zs_dir = os.path.join(SCRIPT_DIR, entity_type.lower(), 'ZS', entity_type.lower())
        
        # Additional ZS directory patterns to check
        zs_dir_patterns = [
            os.path.join(SCRIPT_DIR, 'occupation', 'ZS', entity_type.lower()),
            os.path.join(SCRIPT_DIR, 'ZS', entity_type.lower()),
            os.path.join(SCRIPT_DIR, entity_type.lower(), 'ZS')
        ]
        
        # Additional unknown directory patterns to check
        unknown_dir = os.path.join(SCRIPT_DIR, entity_type.lower(), 'unknown', entity_type.lower())
        unknown_dir_patterns = [
            os.path.join(SCRIPT_DIR, 'occupation', 'unknown', entity_type.lower()),
            os.path.join(SCRIPT_DIR, 'unknown', entity_type.lower()),
            os.path.join(SCRIPT_DIR, entity_type.lower(), 'unknown')
        ]
        
        # Add voting algorithms directories (included by default unless excluded)
        voting_dirs = []
        include_voting = not args.exclude_voting  # Include by default, exclude only if flag is set
        if include_voting:
            voting_base = ""
            voting_dirs = [
                os.path.join(voting_base, "adaptive_threshold_voting", "occupation", entity_type.lower()),
                os.path.join(voting_base, "boyer_moore_voting", "occupation", entity_type.lower()),
                os.path.join(voting_base, "boyer_moore_voting_per_model", "occupation", entity_type.lower()),
                os.path.join(voting_base, "flexible_boyer_moore_voting", "occupation", entity_type.lower())
            ]
            print(f"Including voting algorithm directories: {voting_dirs}")
        else:
            print("Excluding voting algorithm directories (--exclude-voting was specified)")
        
        # Check if directories exist - PRIORITY ORDER IS IMPORTANT
        # First try occupation-specific paths which we know exist
        if os.path.exists(occupation_rag_dir):
            rag_pattern = os.path.join(occupation_rag_dir, '*.csv')
            print(f"  Using occupation-specific RAG directory: {occupation_rag_dir}")
        elif os.path.exists(rag_dir):
            rag_pattern = os.path.join(rag_dir, '*.csv')
            print(f"  Using standard RAG directory: {rag_dir}")
        else:
            # Default fallback
            rag_pattern = os.path.join(SCRIPT_DIR, 'occupation', 'RAG', entity_type.lower(), '*.csv')
            print(f"  Falling back to: {rag_pattern}")
            
        # ZS patterns need special handling for consistent discovery
        zs_pattern = None
        
        # Use absolute known directory first (verified with Bash command)
        occupation_zs_dir_path = os.path.join(SCRIPT_DIR, 'occupation', 'ZS', entity_type.lower())
        if os.path.exists(occupation_zs_dir_path):
            zs_pattern = os.path.join(occupation_zs_dir_path, '*.csv')
            print(f"  Using primary ZS directory: {occupation_zs_dir_path}")
        elif os.path.exists(zs_dir):
            zs_pattern = os.path.join(zs_dir, '*.csv')
            print(f"  Using alternative ZS directory: {zs_dir}")
        else:
            # Try additional patterns as last resort
            for pattern_dir in zs_dir_patterns:
                if os.path.exists(pattern_dir):
                    zs_pattern = os.path.join(pattern_dir, '*.csv')
                    print(f"  Using pattern ZS directory: {pattern_dir}")
                    break
        
        # If still no pattern found, use a default to avoid errors
        if not zs_pattern:
            print(f"  Warning: No ZS directory found for {entity_type}, using fallback pattern")
            # Explicit pattern that matches format from the files we found
            zs_pattern = os.path.join(SCRIPT_DIR, '**', f"QA_occupation_{entity_type.lower()}_noctx*.csv")
            
        # Unknown patterns need similar handling
        unknown_pattern = None
        
        # Use absolute known directory first
        occupation_unknown_dir_path = os.path.join(SCRIPT_DIR, 'occupation', 'unknown', entity_type.lower())
        if os.path.exists(occupation_unknown_dir_path):
            unknown_pattern = os.path.join(occupation_unknown_dir_path, '*.csv')
            print(f"  Using primary unknown directory: {occupation_unknown_dir_path}")
        elif os.path.exists(unknown_dir):
            unknown_pattern = os.path.join(unknown_dir, '*.csv')
            print(f"  Using alternative unknown directory: {unknown_dir}")
        else:
            # Try additional patterns as last resort
            for pattern_dir in unknown_dir_patterns:
                if os.path.exists(pattern_dir):
                    unknown_pattern = os.path.join(pattern_dir, '*.csv')
                    print(f"  Using pattern unknown directory: {pattern_dir}")
                    break
                    
        # If still no pattern found, use a default to avoid errors
        if not unknown_pattern:
            print(f"  Warning: No unknown directory found for {entity_type}, using fallback pattern")
            # Explicit pattern that matches format from the files we found
            unknown_pattern = os.path.join(SCRIPT_DIR, '**', f"QA_occupation_{entity_type.lower()}_combined*.csv")
        
        # Get files - using recursive=True for broader pattern matching
        # Only use recursive mode for patterns that need it
        rag_recursive = '**' in rag_pattern
        zs_recursive = '**' in zs_pattern
        unknown_recursive = '**' in unknown_pattern
        
        rag_files = glob.glob(rag_pattern, recursive=rag_recursive)
        zs_files = glob.glob(zs_pattern, recursive=zs_recursive)
        unknown_files = glob.glob(unknown_pattern, recursive=unknown_recursive)
        
        print(f"  Using recursive={zs_recursive} for ZS file search")
        print(f"  ZS pattern: {zs_pattern}")
        print(f"  Using recursive={unknown_recursive} for unknown file search")
        print(f"  Unknown pattern: {unknown_pattern}")
        
        # Print paths we're looking in
        print(f"Looking for {entity_type} files in:")
        print(f"  RAG path: {rag_pattern}")
        print(f"  ZS path: {zs_pattern}")
        print(f"  Unknown path: {unknown_pattern}")
        
        # Get voting files if not excluded
        voting_files = []
        if include_voting:
            for voting_dir in voting_dirs:
                if os.path.exists(voting_dir):
                    voting_pattern = os.path.join(voting_dir, "*.csv")
                    voting_dir_files = glob.glob(voting_pattern)
                    print(f"Found {len(voting_dir_files)} voting files in {voting_dir}")
                    voting_files.extend(voting_dir_files)
            
        # Combine all files based on command-line arguments
        if include_unknown:
            csv_files = rag_files + zs_files + unknown_files + voting_files
            print(f"Found {len(rag_files)} RAG files, {len(zs_files)} ZS files, {len(unknown_files)} unknown files, and {len(voting_files)} voting files for {entity_type}")
        else:
            # Exclude unknown files if --exclude-unknown is specified
            csv_files = rag_files + zs_files + voting_files
            print(f"Found {len(rag_files)} RAG files, {len(zs_files)} ZS files, and {len(voting_files)} voting files for {entity_type} (ignoring {len(unknown_files)} unknown files)")
        
        # If we found no ZS files, we need to search more aggressively using multiple patterns and recursive search
        if len(zs_files) == 0:
            print("No ZS files found through standard paths, searching more broadly...")
            
            # Try several different patterns to find ZS files
            broader_patterns = [
                # Most specific pattern first
                os.path.join(SCRIPT_DIR, 'occupation', 'ZS', '**', f'*{entity_type.lower()}*noctx*.csv'),
                # Then a more general pattern across all evaluation directories
                os.path.join(SCRIPT_DIR, '**', f'*{entity_type.lower()}*noctx*.csv'),
                # Even more general if needed
                os.path.join(SCRIPT_DIR, '**', f'*noctx*{entity_type.lower()}*.csv')
            ]
            
            for pattern in broader_patterns:
                broader_zs_files = glob.glob(pattern, recursive=True)
                if broader_zs_files:
                    print(f"Found {len(broader_zs_files)} ZS files with pattern: {pattern}")
                    # Filter out duplicates or _evaluated_evaluated.csv files
                    filtered_zs_files = [f for f in broader_zs_files if "_evaluated_evaluated.csv" not in f]
                    if filtered_zs_files:
                        print(f"  After filtering: {len(filtered_zs_files)} files")
                        csv_files.extend(filtered_zs_files)
                        zs_files = filtered_zs_files
                        break
        
        # Print ZS files for debugging
        if len(zs_files) > 0:
            print("Zero-shot files found:")
            for zs_file in zs_files:
                print(f"  {os.path.basename(zs_file)}")
        
        # Print unknown files for debugging
        if include_unknown and len(unknown_files) > 0:
            print("Unknown files found (will be processed):")
            for unknown_file in unknown_files:
                print(f"  {os.path.basename(unknown_file)}")
        
        # Filter out duplicates and already processed files
        unique_files = {}
        for file_path in csv_files:
            basename = os.path.basename(file_path)
            
            # Skip any files with double "_evaluated_evaluated" suffix
            if "_evaluated_evaluated" in basename:
                print(f"  Skipping duplicate evaluated file: {basename}")
                continue
                
            # Skip any files marked as granular results
            if basename.endswith('_granular.csv'):
                print(f"  Skipping granular file: {basename}")
                continue
                
            # Skip any test files
            if "test" in basename.lower() or "_test_" in basename or "_TEST_" in basename:
                print(f"  Skipping test file: {basename}")
                continue
                
            # Extract key components from filename to create a unique key
            # Try to match common patterns:
            
            # Extract model and retriever for deduplication
            model_match = re.search(r'openrouter_(\w+[-\w]*)_(\w+[-\w]*)', basename)
            retriever_match = re.search(r'output_(\w+[-\w]*?)_', basename)
            
            if model_match:
                # Get the model part (e.g., google_gemma)
                model_key = f"{model_match.group(1)}_{model_match.group(2)}"
                
                # Try to get the retriever part if it exists
                retriever_key = "unknown"
                if retriever_match:
                    retriever_key = retriever_match.group(1)
                elif "noctx" in basename or "/ZS/" in file_path:
                    retriever_key = "zero-shot"
                    print(f"  Identified as zero-shot: {basename}")
                    
                # Create a composite key that identifies model + retriever
                composite_key = f"{model_key}_{retriever_key}"
                
                # Print info for debugging
                print(f"  File: {basename} -> Key: {composite_key}")
                
                if composite_key not in unique_files:
                    unique_files[composite_key] = file_path
                else:
                    print(f"  Skipping duplicate for {composite_key}: {basename}")
            else:
                # Fallback to the old method if pattern matching fails
                base_key = basename.replace("_linked_evaluated", "").replace("_evaluated", "")
                if base_key not in unique_files:
                    unique_files[base_key] = file_path
                else:
                    print(f"  Skipping duplicate: {basename}")
                    
        
        csv_files = list(unique_files.values())
        
        if not csv_files:
            print(f"No CSV files found for entity type: {entity_type}")
            continue
        
        print(f"Processing {len(csv_files)} files for entity type: {entity_type}")
        
        # Process each file and collect dataframes
        dataframes = []
        for file_path in csv_files:
            df = process_file(file_path)
            if not df.empty:
                dataframes.append(df)
        
        if not dataframes:
            print(f"No valid lookup_tables processed for entity type: {entity_type}")
            continue
        
        # Concatenate dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Calculate statistics with force_zs option
        stats_df = calculate_statistics(combined_df, force_zs=force_zs, entity_type=entity_type)
        
        # Debug: Log the retrievers and models in the stats_df
        print("\nStats DataFrame contains:")
        print(f"  Number of rows: {len(stats_df)}")
        print(f"  Unique retrievers: {stats_df['retriever'].unique()}")
        print(f"  Unique models: {stats_df['llm'].unique()}")
        
        # Rename 'unknown' to 'Majority Voting' in the stats_df
        # Both for display in reports and to ensure consistent naming
        if 'retriever' in stats_df.columns:
            if isinstance(stats_df['retriever'], pd.Series):
                # For retrievers
                stats_df.loc[stats_df['retriever'].str.contains('adaptive_threshold', na=False), 'retriever'] = 'Adaptive Threshold'
                stats_df.loc[stats_df['retriever'].str.contains('boyer_moore', na=False), 'retriever'] = 'Boyer-Moore'
                stats_df.loc[stats_df['retriever'].str.contains('flexible_boyer', na=False), 'retriever'] = 'Flexible Boyer-Moore'
                stats_df.loc[stats_df['retriever'] == 'unknown', 'retriever'] = 'Majority Voting'
                
        # Also update the model name for 'unknown' to 'Majority Voting'
        if 'llm' in stats_df.columns:
            if isinstance(stats_df['llm'], pd.Series):
                stats_df.loc[stats_df['llm'] == 'unknown', 'llm'] = 'Majority Voting'
                
        # Add any missing Majority Voting entries from the granular files
        try:
            # Check directly for specific algorithm files to ensure inclusion
            unknown_files = [
                os.path.join(OUTPUT_BASE_DIR, 'unknown', entity_type.lower(), f'QA_occupation_{entity_type.upper()}_combined_adaptive_threshold_20250520_linked_evaluated_granular.csv'),
                os.path.join(OUTPUT_BASE_DIR, 'unknown', entity_type.lower(), f'QA_occupation_{entity_type.upper()}_combined_boyer_moore_20250520_linked_evaluated_granular.csv'),
                os.path.join(OUTPUT_BASE_DIR, 'unknown', entity_type.lower(), f'QA_occupation_{entity_type.upper()}_combined_flexible_boyer_20250520_linked_evaluated_granular.csv')
            ]
            # Filter to only existing files
            unknown_files = [f for f in unknown_files if os.path.exists(f)]
            
            # Also add any additional files matching pattern
            unknown_files += glob.glob(os.path.join(OUTPUT_BASE_DIR, 'unknown', entity_type.lower(), '*.csv'))
            unknown_files += glob.glob(os.path.join(OUTPUT_BASE_DIR, 'unknown', entity_type.lower(), '*_granular.csv'))
            unknown_dfs = []
            for file in unknown_files:
                if os.path.basename(file) != 'combined_test' and '_test_' not in file and 'test_' not in file:
                    try:
                        df = pd.read_csv(file)
                        if not df.empty and 'sample_precision' in df.columns:
                            unknown_dfs.append(df)
                    except Exception as e:
                        print(f"Error reading unknown file {file}: {e}")
            
            if unknown_dfs:
                majority_df = pd.concat(unknown_dfs, ignore_index=True)
                
                # Group by algorithm
                for algorithm, group in majority_df.groupby('retriever'):
                    if 'adaptive_threshold' in algorithm:
                        retriever = 'Adaptive Threshold'
                    elif 'boyer_moore' in algorithm:
                        retriever = 'Boyer-Moore'
                    elif 'flexible_boyer' in algorithm:
                        retriever = 'Flexible Boyer-Moore'
                    else:
                        retriever = 'Majority Voting'
                        
                    # Calculate metrics for the algorithm
                    metrics = {}
                    for metric in METRICS:
                        values = group[f'sample_{metric}'].dropna().values
                        if len(values) > 0:
                            metrics[f'{metric}_mean'] = np.mean(values)
                            metrics[f'{metric}_median'] = np.median(values)
                            # Handle different scipy versions and fix scalar indexing issues
                            mode_result = stats.mode(np.round(values, 3))
                            if hasattr(mode_result, 'mode'):
                                if isinstance(mode_result.mode, np.ndarray) and mode_result.mode.size > 0:
                                    metrics[f'{metric}_mode'] = mode_result.mode[0]
                                else:
                                    metrics[f'{metric}_mode'] = mode_result.mode
                            else:
                                if isinstance(mode_result[0], np.ndarray) and mode_result[0].size > 0:
                                    metrics[f'{metric}_mode'] = mode_result[0][0]
                                else:
                                    metrics[f'{metric}_mode'] = float('nan')
                            metrics[f'{metric}_std'] = np.std(values)
                        else:
                            metrics[f'{metric}_mean'] = np.nan
                            metrics[f'{metric}_median'] = np.nan
                            metrics[f'{metric}_mode'] = np.nan
                            metrics[f'{metric}_std'] = np.nan
                    
                    # Only add if not already in stats_df
                    if not ((stats_df['llm'] == 'Majority Voting') & (stats_df['retriever'] == retriever)).any():
                        row = {'llm': 'Majority Voting', 'retriever': retriever}
                        row.update(metrics)
                        stats_df = pd.concat([stats_df, pd.DataFrame([row])], ignore_index=True)
        except Exception as e:
            print(f"Warning: Could not add Majority Voting entries: {e}")
            import traceback
            traceback.print_exc()
        
        print("\nAfter renaming:")
        print(f"  Unique retrievers: {stats_df['retriever'].unique()}")
        
        # Generate reports
        generate_csv_report(stats_df, entity_type)
        generate_latex_table(stats_df, entity_type)
        plot_metrics_heatmap(stats_df, entity_type)

if __name__ == "__main__":
    main()