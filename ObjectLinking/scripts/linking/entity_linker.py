#!/usr/bin/env python3
"""
Unified Entity Linking Module for NIL Grounding Evaluation

This module provides a unified framework for entity linking tasks, replacing 
the separate scripts for different entity types with a configurable system.
It links text answers to Wikidata QIDs by looking up in local CSV files.
"""

import os
import re
import glob
import time
import json
import sys
import math
import pandas as pd
import logging
import traceback
import unicodedata
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Set, Any, Callable, Union
from datetime import datetime
from difflib import get_close_matches
from abc import ABC, abstractmethod

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("tqdm not available. Install with 'pip install tqdm' for progress bars.")
    print("Continuing without progress bars...")


class EntityConfig(ABC):
    """
    Abstract base class for entity-specific configurations.
    Each entity type should implement this class with its specific settings.
    """
    
    @property
    @abstractmethod
    def entity_name(self) -> str:
        """Human-readable name of the entity type"""
        pass
    
    @property
    @abstractmethod
    def entity_type(self) -> str:
        """Short identifier for the entity type"""
        pass
    
    @property
    @abstractmethod
    def csv_file_path(self) -> str:
        """Path to the CSV file containing entity lookup_tables"""
        pass
    
    @property
    @abstractmethod
    def instance_qids(self) -> List[str]:
        """List of Wikidata QIDs that represent instances of this entity type"""
        pass
    
    @property
    def cache_file(self) -> str:
        """Path to the cache file for this entity type"""
        return os.path.join(os.path.dirname(self.csv_file_path), f"{self.entity_type}_qid_cache.json")
    
    @property
    def report_name(self) -> str:
        """Name for the report file"""
        return f"{self.entity_type}_linking_report.csv"
    
    @property
    def extraction_patterns(self) -> List[Tuple[str, int]]:
        """
        Extraction patterns have been deactivated.
        Returns an empty list.
        """
        # Empty list - no extraction patterns
        return []
    
    @property
    def normalization_patterns(self) -> List[Tuple[str, str]]:
        """
        Normalization patterns have been deactivated.
        Returns an empty list.
        """
        # Empty list - no normalization patterns
        return []
    
    def sanitize_input(self, text: str) -> str:
        """
        Basic input sanitization that:
        1. Trims whitespace
        2. Removes punctuation
        3. Handles diacritics

        Entity-specific sanitization has been deactivated.
        """
        if not text:
            return ""

        # Trim whitespace
        cleaned_text = text.strip()

        # Handle diacritics
        import unicodedata
        normalized_text = unicodedata.normalize('NFKD', cleaned_text)
        cleaned_text = ''.join([c for c in normalized_text if not unicodedata.combining(c)])

        # Remove punctuation
        import string
        cleaned_text = ''.join([c for c in cleaned_text if c not in string.punctuation])
        cleaned_text = cleaned_text.strip()  # Remove any whitespace created by punctuation removal

        return cleaned_text
    
    def normalize_entity(self, text: str) -> str:
        """
        Entity-specific normalization has been deactivated.
        Performs minimal normalization:
        1. Lowercasing
        2. Whitespace trimming
        3. Punctuation removal
        4. Diacritic handling
        """
        if not text:
            return ""

        # Trim whitespace and lowercase
        cleaned_text = text.lower().strip()

        # Handle diacritics
        import unicodedata
        normalized_text = unicodedata.normalize('NFKD', cleaned_text)
        cleaned_text = ''.join([c for c in normalized_text if not unicodedata.combining(c)])

        # Remove punctuation
        import string
        cleaned_text = ''.join([c for c in cleaned_text if c not in string.punctuation])
        cleaned_text = cleaned_text.strip()

        return cleaned_text


class EntityDatabase:
    """
    Class to load and manage entities from a local CSV file.
    Provides lookup by name and fuzzy matching capabilities.
    """
    
    def __init__(self, config: EntityConfig):
        self.config = config
        self.csv_file_path = config.csv_file_path
        self.entities_by_label = {}  # Label (lowercase) -> QID
        self.entities_by_alias = {}  # Alias (lowercase) -> QID
        self.qid_to_label = {}       # QID -> original label
        self.qid_to_aliases = {}     # QID -> list of aliases
        self.loaded = False
    
    def load(self):
        """Load the entities from the CSV file"""
        logging.info(f"Loading {self.config.entity_name} lookup_tables from {self.csv_file_path}")
        
        try:
            # Check if the CSV file exists
            if not os.path.exists(self.csv_file_path):
                logging.warning(f"{self.config.entity_name} CSV file not found: {self.csv_file_path}")
                logging.info("Will use cache only")
                self.loaded = True
                return
            
            df = pd.read_csv(self.csv_file_path)
            total_rows = len(df)
            
            # Display loading progress
            if TQDM_AVAILABLE:
                iterator = tqdm(df.iterrows(), total=total_rows, desc=f"Loading {self.config.entity_name}")
            else:
                iterator = df.iterrows()
                logging.info(f"Loading {total_rows} {self.config.entity_name} entries...")
            
            for _, row in iterator:
                qid = row['QID']
                label = row['Label']
                
                # Skip rows with missing QID or label
                if pd.isna(qid) or pd.isna(label):
                    continue
                
                # Store QID -> label mapping
                self.qid_to_label[qid] = label
                
                # Store label -> QID mapping (case insensitive)
                self.entities_by_label[label.lower()] = qid
                
                # Process aliases if available
                aliases = []
                if 'Aliases' in row and not pd.isna(row['Aliases']) and row['Aliases'] != '':
                    # Split aliases by pipe character
                    aliases = row['Aliases'].split('|')
                    
                    # Store each alias -> QID mapping (case insensitive)
                    for alias in aliases:
                        if alias and not pd.isna(alias):
                            self.entities_by_alias[alias.lower()] = qid
                
                # Store QID -> aliases mapping
                self.qid_to_aliases[qid] = aliases
            
            self.loaded = True
            logging.info(f"Loaded {len(self.entities_by_label)} labels and {len(self.entities_by_alias)} aliases")
            
        except Exception as e:
            logging.error(f"Error loading {self.config.entity_name} lookup_tables: {str(e)}")
            logging.error(traceback.format_exc())
            logging.warning("Will use cache only")
            self.loaded = True
    
    def lookup_by_name(self, name):
        """
        Look up a QID by entity name (label or alias)
        Applies minimal normalization (lowercase and trimming)
        """
        if not self.loaded:
            self.load()

        if not name:
            return None

        # Basic normalization - lowercase and trim whitespace
        name = name.strip().lower()

        # Check if name is in labels
        if name in self.entities_by_label:
            return self.entities_by_label[name]

        # Check if name is in aliases
        if name in self.entities_by_alias:
            return self.entities_by_alias[name]

        return None
    
    def get_close_name_matches(self, name, cutoff=0.85):
        """Find close matches for an entity name in the database"""
        if not self.loaded:
            self.load()
        
        if not self.entities_by_label and not self.entities_by_alias:
            return None
        
        name = name.strip().lower()
        
        # Try to find matches in labels
        label_matches = get_close_matches(name, self.entities_by_label.keys(), n=1, cutoff=cutoff)
        if label_matches:
            matched_label = label_matches[0]
            return self.entities_by_label[matched_label]
        
        # If no match in labels, try aliases
        alias_matches = get_close_matches(name, self.entities_by_alias.keys(), n=1, cutoff=cutoff)
        if alias_matches:
            matched_alias = alias_matches[0]
            return self.entities_by_alias[matched_alias]
        
        return None


class CacheManager:
    """
    Manages different types of caches for the application.
    Provides caching of string-to-QID mappings and fuzzy matching.
    """
    
    def __init__(self, cache_file: str, save_interval: int = 25):
        self.cache_file = cache_file
        self.save_interval = save_interval
        self.string_to_qid_cache = self.load_cache()  # Persistent string-to-QID mappings
        self.last_save_time = time.time()
        self.items_since_save = 0
        self.fuzzy_match_dict = {}  # For fuzzy matching
        self.retry_queue = set()  # Set of items that failed and should be retried
        self.cache_hits = 0  # Track cache performance
        self.cache_misses = 0
    
    def load_cache(self) -> Dict[str, str]:
        """Load the cache from disk or initialize if not present"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    logging.info(f"Loaded {len(cache)} entries from cache file")
                    return cache
            else:
                logging.info("No cache file found, starting with empty cache")
                return {}
        except Exception as e:
            logging.error(f"Error loading cache: {str(e)}\n{traceback.format_exc()}")
            return {}
    
    def save_cache(self, force=False):
        """Save the cache to disk, respecting the save interval"""
        current_time = time.time()
        self.items_since_save += 1
        
        # Only save if forced or we've processed enough items since last save
        if force or self.items_since_save >= self.save_interval or (current_time - self.last_save_time) > 300:
            try:
                # Ensure cache directory exists
                cache_dir = os.path.dirname(self.cache_file)
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                
                # Use a temporary file to avoid corruption if interrupted
                temp_file = f"{self.cache_file}.tmp"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.string_to_qid_cache, f, ensure_ascii=False, indent=2)
                
                # Atomic rename
                if os.path.exists(self.cache_file):
                    os.replace(temp_file, self.cache_file)
                else:
                    os.rename(temp_file, self.cache_file)
                
                self.last_save_time = current_time
                self.items_since_save = 0
                logging.info(f"Saved {len(self.string_to_qid_cache)} entries to cache file")
            
            except Exception as e:
                logging.error(f"Error saving cache: {str(e)}\n{traceback.format_exc()}")
    
    def add_to_cache(self, string: str, qid: str):
        """Add a string-to-QID mapping to the cache"""
        if not string or not qid:
            return
        
        string = string.lower().strip()
        self.string_to_qid_cache[string] = qid
        
        # Update fuzzy match dictionary
        self.fuzzy_match_dict[string] = qid
    
    def get_from_cache(self, string: str, fuzzy_threshold: float = 0.85) -> Optional[str]:
        """Get a QID from the cache using exact or fuzzy matching"""
        if not string:
            return None
        
        string = string.lower().strip()
        
        # Try exact match first
        if string in self.string_to_qid_cache:
            self.cache_hits += 1
            return self.string_to_qid_cache[string]
        
        # Try fuzzy matching if we have enough entries
        if len(self.fuzzy_match_dict) > 5:
            close_matches = get_close_matches(string, self.fuzzy_match_dict.keys(), n=1, cutoff=fuzzy_threshold)
            if close_matches:
                matched_string = close_matches[0]
                qid = self.fuzzy_match_dict[matched_string]
                logging.info(f"Fuzzy matched '{string}' to '{matched_string}' -> {qid}")
                
                # Add the new form to the cache
                self.add_to_cache(string, qid)
                self.cache_hits += 1
                return qid
        
        self.cache_misses += 1
        return None
    
    def add_to_retry_queue(self, item: str):
        """Add an item to the retry queue"""
        self.retry_queue.add(item)
    
    def get_retry_queue(self) -> Set[str]:
        """Get the current retry queue"""
        return self.retry_queue.copy()
    
    def clear_retry_queue(self):
        """Clear the retry queue"""
        self.retry_queue.clear()


class EntityLinker:
    """
    Main entity linking class that processes files and links
    entity mentions to Wikidata QIDs.
    """
    
    def __init__(self, config: EntityConfig, output_dir: str = None, 
                 max_workers: int = 2, batch_size: int = 10,
                 fuzzy_match_threshold: float = 0.85,
                 checkpoint_interval: int = 5,
                 test_mode: bool = False):
        self.config = config
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), "output")
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.checkpoint_interval = checkpoint_interval
        self.test_mode = test_mode
        
        # Configure paths
        self.cache_dir = os.path.join(self.output_dir, "cache")
        self.cache_file = os.path.join(self.cache_dir, f"{config.entity_type}_qid_cache.json")
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.log_file = os.path.join(self.output_dir, f"{config.entity_type}_linking.log")
        self.error_log_file = os.path.join(self.output_dir, f"{config.entity_type}_errors.log")
        
        # Initialize components
        self.entity_db = EntityDatabase(config)
        self.cache_manager = CacheManager(self.cache_file)
        
        # Set up logging for this instance
        self.setup_logging()
        
        # Ensure necessary directories exist
        self.ensure_directories()
    
    def setup_logging(self):
        """Set up logging with both main log and separate error log"""
        if not os.path.exists(os.path.dirname(self.log_file)):
            os.makedirs(os.path.dirname(self.log_file))
        
        if not os.path.exists(os.path.dirname(self.error_log_file)):
            os.makedirs(os.path.dirname(self.error_log_file))
        
        # Configure main logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create formatters
        main_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        error_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s\n%(pathname)s:%(lineno)d\n")
        
        # Set up main log file
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(main_formatter)
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        
        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(main_formatter)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        
        # Set up error log file (only errors and criticals)
        error_handler = logging.FileHandler(self.error_log_file)
        error_handler.setFormatter(error_formatter)
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
        
        logging.info(f"Logging initialized. Main log: {self.log_file}, Error log: {self.error_log_file}")
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.output_dir,
            self.cache_dir,
            self.checkpoint_dir
        ]
        
        if self.test_mode:
            test_folder = os.path.join(self.output_dir, "test")
            directories.extend([
                test_folder,
                os.path.join(test_folder, "RAG"),
                os.path.join(test_folder, "ZS")
            ])
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logging.info(f"Created directory: {directory}")
    
    def split_answer_into_entities(self, answer: str) -> List[str]:
        """
        Split an answer into multiple entities if it contains delimiters like semicolons.
        Handles various delimiter formats and removes any leading/trailing whitespace.

        Examples:
            "composer; conductor; harpsichordist" -> ["composer", "conductor", "harpsichordist"]
            "actor:director" -> ["actor", "director"]
            "teacher,writer" -> ["teacher", "writer"]
        """
        if not answer:
            return []

        # Define common separators
        separators = [';', ':', ',', '/', '|']

        # Check if the answer contains any of the separators
        if any(sep in answer for sep in separators):
            # Use regex to split by any of the separators while handling whitespace
            import re
            pattern = r'[;:,/|]+\s*'
            entities = re.split(pattern, answer)
            # Also handle the case where there might be a separator at the beginning
            entities = [e.strip() for e in entities if e.strip()]
            return entities

        # Return the single entity if no separators found
        return [answer.strip()]

    def process_answer(self, answer: str) -> Union[Optional[str], List[str]]:
        """
        Process a single entity mention to find its Wikidata QID(s).
        Uses only exact matching with minimal cleaning.
        All normalization and fuzzy matching has been deactivated.

        Performs minimal input sanitization:
        1. Whitespace trimming
        2. Punctuation removal
        3. Diacritic removal

        If the answer contains multiple entities separated by delimiters,
        attempts to link each entity and returns a semicolon-separated list of all QIDs.
        """
        if not answer or pd.isna(answer) or answer.strip() == '':
            return None

        # Check for multiple entities in the answer
        entities = self.split_answer_into_entities(answer)

        # If multiple entities are found, try to process each one
        if len(entities) > 1:
            logging.info(f"Found multiple entities in answer: {answer} -> {entities}")

            # Try to link each entity
            successful_qids = []
            for entity in entities:
                qid = self._process_single_entity(entity)
                if qid:
                    logging.info(f"Successfully linked entity '{entity}' from multi-entity answer '{answer}' to {qid}")
                    successful_qids.append(qid)

            # If any entities were linked, return the semicolon-joined list of QIDs
            if successful_qids:
                combined_qids = ";".join(successful_qids)
                logging.info(f"Combined QIDs for multi-entity answer '{answer}': {combined_qids}")
                return combined_qids

            # If no entity could be linked, return None
            logging.warning(f"Could not link any entity from multi-entity answer: {answer}")
            return None

        # Process single entity
        return self._process_single_entity(answer)

    def _process_single_entity(self, entity: str) -> Optional[str]:
        """
        Process a single entity to find its Wikidata QID.
        Helper method extracted from process_answer for clarity.
        """
        # Skip if answer is too long
        if len(str(entity)) > 2000:  # Increase from 100 to 2000 to handle longer responses
            logging.warning(f"Skipping entity with length {len(str(entity))} > 2000 characters")
            return None

        # Minimal text processing - trim whitespace
        original_answer = entity.strip()

        # Basic cleaning - handle diacritics
        normalized_answer = unicodedata.normalize('NFKD', original_answer)
        cleaned_answer = ''.join([c for c in normalized_answer if not unicodedata.combining(c)])

        # Remove punctuation (keeping minimal approach)
        import string
        no_punct_answer = ''.join([c for c in cleaned_answer if c not in string.punctuation])
        no_punct_answer = no_punct_answer.strip()  # Remove any whitespace created by punctuation removal

        # Log the processing at debug level to avoid log spam
        logging.debug(f"Processing entity: '{original_answer}'")
        if cleaned_answer != original_answer:
            logging.debug(f"Diacritic cleaning: '{original_answer}' → '{cleaned_answer}'")
        if no_punct_answer != cleaned_answer:
            logging.debug(f"Punctuation removal: '{cleaned_answer}' → '{no_punct_answer}'")

        # Step 1: Check in cache (exact match only)
        qid = None
        lower_original = original_answer.lower()
        lower_cleaned = cleaned_answer.lower()
        lower_no_punct = no_punct_answer.lower()

        # Try to get from cache with exact match only
        if lower_original in self.cache_manager.string_to_qid_cache:
            qid = self.cache_manager.string_to_qid_cache[lower_original]
            logging.info(f"Found QID for '{original_answer}' in cache: {qid}")
            return qid

        if lower_cleaned != lower_original and lower_cleaned in self.cache_manager.string_to_qid_cache:
            qid = self.cache_manager.string_to_qid_cache[lower_cleaned]
            logging.info(f"Found QID for cleaned '{cleaned_answer}' in cache: {qid}")
            return qid

        if lower_no_punct != lower_cleaned and lower_no_punct in self.cache_manager.string_to_qid_cache:
            qid = self.cache_manager.string_to_qid_cache[lower_no_punct]
            logging.info(f"Found QID for no-punctuation '{no_punct_answer}' in cache: {qid}")
            return qid

        # Step 2: Look up in the entity database directly - exact matches only
        logging.info(f"Looking up '{original_answer}' in {self.config.entity_name} database")

        # Try exact lookup with original form
        qid = self.entity_db.lookup_by_name(original_answer)

        # If not found, try with lowercase original
        if not qid and lower_original != original_answer:
            qid = self.entity_db.lookup_by_name(lower_original)

        # If not found, try with cleaned form (diacritics removed)
        if not qid and cleaned_answer != original_answer:
            qid = self.entity_db.lookup_by_name(cleaned_answer)

            # If not found, try with lowercase cleaned
            if not qid and lower_cleaned != cleaned_answer:
                qid = self.entity_db.lookup_by_name(lower_cleaned)

        # If not found, try with no punctuation form
        if not qid and no_punct_answer != cleaned_answer:
            qid = self.entity_db.lookup_by_name(no_punct_answer)

            # If not found, try with lowercase no punctuation
            if not qid and lower_no_punct != no_punct_answer:
                qid = self.entity_db.lookup_by_name(lower_no_punct)

        # Step 3: Process results
        if not qid:
            logging.warning(f"No exact match found for '{original_answer}'")
            return None

        # Found a QID
        if qid:
            # Add to cache with all processed forms
            self.cache_manager.add_to_cache(lower_original, qid)
            if lower_cleaned != lower_original:
                self.cache_manager.add_to_cache(lower_cleaned, qid)
            if lower_no_punct != lower_cleaned:
                self.cache_manager.add_to_cache(lower_no_punct, qid)

            # Periodically save cache
            self.cache_manager.save_cache()

            logging.info(f"Found QID for '{original_answer}': {qid}")

        return qid
    
    def process_batch(self, answers_batch: List[str]) -> Dict[str, Optional[str]]:
        """Process a batch of unique answers using parallel execution"""
        result_dict = {}
        
        # Use ThreadPoolExecutor to process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks and keep track of futures
            futures = {executor.submit(self.process_answer, answer): answer for answer in answers_batch}
            
            # Process results as they complete
            for future in futures:
                answer = futures[future]
                try:
                    # Get the result from the future
                    qid = future.result()
                    result_dict[answer] = qid
                except Exception as e:
                    logging.error(f"Error processing '{answer}': {str(e)}")
                    logging.error(traceback.format_exc())
                    result_dict[answer] = None
                    # Add to retry queue
                    self.cache_manager.add_to_retry_queue(answer)
        
        return result_dict
    
    def save_checkpoint(self, file_path: str, processed_data: Dict[str, Any]) -> None:
        """Save a checkpoint to allow resuming after errors"""
        try:
            checkpoint_file = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_{os.path.basename(file_path).replace('.csv', '')}.json"
            )
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"Checkpoint saved to {checkpoint_file}")
        except Exception as e:
            logging.error(f"Error saving checkpoint: {str(e)}")
            logging.error(traceback.format_exc())
    
    def load_checkpoint(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint if it exists"""
        try:
            checkpoint_file = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_{os.path.basename(file_path).replace('.csv', '')}.json"
            )
            
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    logging.info(f"Checkpoint loaded from {checkpoint_file}")
                    return checkpoint_data
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
            logging.error(traceback.format_exc())
        
        return None
    
    def process_file(self, file_path: str) -> pd.DataFrame:
        """
        Process a CSV file in batches with checkpointing to avoid lookup_tables loss
        and enable resuming after errors
        """
        filename = os.path.basename(file_path)
        logging.info(f"Processing file: {filename}")

        # Check for existing checkpoint
        checkpoint_data = self.load_checkpoint(file_path)

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Create a copy for processing
            df_processed = df.copy()

            # Add required columns if they don't exist
            if 'linked_qid' not in df_processed.columns:
                df_processed['linked_qid'] = None
            if 'match' not in df_processed.columns:
                df_processed['match'] = None

            # Filter for rows with non-empty qid_gold_true
            mask = ~df_processed['qid_gold_true'].isna() & (df_processed['qid_gold_true'] != '')

            # Skip rows with llm_answer longer than 100 characters
            try:
                # Convert any non-string answers to strings and handle missing values
                df_processed['llm_answer'] = df_processed['llm_answer'].fillna('')
                df_processed['llm_answer'] = df_processed['llm_answer'].astype(str)

                long_answer_mask = df_processed['llm_answer'].str.len() > 100
                if any(long_answer_mask):
                    logging.info(f"Skipping {long_answer_mask.sum()} rows with llm_answer longer than 100 characters")
                    mask = mask & ~long_answer_mask
            except Exception as e:
                logging.error(f"Error filtering long answers: {str(e)}")
                logging.error(traceback.format_exc())

            df_filtered = df_processed[mask]

            if df_filtered.empty:
                logging.warning(f"No rows with non-empty qid_gold_true in {filename}")
                return df_processed

            # Get unique answers to reduce redundant lookups
            unique_answers = df_filtered['llm_answer'].dropna().drop_duplicates().tolist()
            unique_answers = [a for a in unique_answers if a.strip() != '']

            if not unique_answers:
                logging.warning(f"No non-empty answers to process in {filename}")
                return df_processed

            # Pre-process answers to identify multi-entity answers
            multi_entity_map = {}
            for answer in unique_answers:
                entities = self.split_answer_into_entities(answer)
                if len(entities) > 1:
                    multi_entity_map[answer] = entities
                    logging.info(f"Identified multi-entity answer: {answer} -> {entities}")

            # Check if we have cached results from a checkpoint
            answer_to_qid_map = {}
            if checkpoint_data and 'answer_to_qid_map' in checkpoint_data:
                answer_to_qid_map = checkpoint_data['answer_to_qid_map']
                logging.info(f"Loaded {len(answer_to_qid_map)} cached answers from checkpoint")

                # Filter out answers that we already processed
                unique_answers = [a for a in unique_answers if a not in answer_to_qid_map]

            # Log the total number of unique answers to process
            num_batches = math.ceil(len(unique_answers)/self.batch_size)
            logging.info(f"Processing {len(unique_answers)} unique answers in {num_batches} batches")
            logging.info(f"Found {len(multi_entity_map)} answers containing multiple entities")

            # Process unique answers in batches
            batch_count = 0

            # Set up tqdm progress bar if available
            batch_iterator = range(0, len(unique_answers), self.batch_size)
            if TQDM_AVAILABLE:
                batch_iterator = tqdm(batch_iterator, total=num_batches, desc=f"Processing {filename}",
                                   unit="batch", leave=True)

            for i in batch_iterator:
                batch_count += 1
                batch = unique_answers[i:i+self.batch_size]

                if not TQDM_AVAILABLE:
                    logging.info(f"Processing batch {batch_count}/{num_batches} with {len(batch)} answers")

                # Process this batch
                batch_results = self.process_batch(batch)

                # Update the mapping
                answer_to_qid_map.update(batch_results)

                # Save checkpoint after each batch
                if batch_count % 2 == 0 or i+self.batch_size >= len(unique_answers):
                    self.save_checkpoint(file_path, {'answer_to_qid_map': answer_to_qid_map})

                # Save cache periodically
                self.cache_manager.save_cache()

                # Process any accumulated items in the retry queue if we've completed batches
                retry_queue = self.cache_manager.get_retry_queue()
                if retry_queue and (batch_count % 3 == 0 or i+self.batch_size >= len(unique_answers)):
                    logging.info(f"Processing {len(retry_queue)} items from retry queue")
                    retry_batch = list(retry_queue)[:self.batch_size]  # Take up to batch_size items
                    retry_results = self.process_batch(retry_batch)
                    answer_to_qid_map.update(retry_results)
                    self.cache_manager.clear_retry_queue()
                    self.save_checkpoint(file_path, {'answer_to_qid_map': answer_to_qid_map})

            # Apply the mapping to all rows
            matched_count = 0
            linked_count = 0
            for idx, row in df_filtered.iterrows():
                if pd.isna(row['llm_answer']) or row['llm_answer'].strip() == '':
                    continue

                answer = row['llm_answer']

                # Check if we already have a QID for this answer
                qid = answer_to_qid_map.get(answer)

                # If not found in map and is a multi-entity answer, try individual entities
                if qid is None and answer in multi_entity_map:
                    # Get all successful QIDs for the multiple entities
                    successful_qids = []
                    for entity in multi_entity_map[answer]:
                        entity_qid = self._process_single_entity(entity)
                        if entity_qid:
                            successful_qids.append(entity_qid)
                            logging.info(f"Found QID {entity_qid} for entity '{entity}' in multi-entity answer '{answer}'")

                    # Combine QIDs with semicolons if we found any
                    if successful_qids:
                        qid = ";".join(successful_qids)
                        logging.info(f"Combined QIDs for multi-entity answer '{answer}': {qid}")
                        # Add this mapping to our answer_to_qid_map for future reference
                        answer_to_qid_map[answer] = qid

                df_processed.at[idx, 'linked_qid'] = qid

                # Update counts - track if linked and matched
                if qid:
                    linked_count += 1

                    # For multi-QID answers, check if gold QID is in our list of QIDs
                    if ";" in str(qid):
                        gold_qid = row['qid_gold_true']
                        if str(gold_qid) in str(qid).split(";"):
                            matched_count += 1
                            df_processed.at[idx, 'match'] = 1
                        else:
                            df_processed.at[idx, 'match'] = 0
                    elif qid == row['qid_gold_true']:
                        matched_count += 1
                        df_processed.at[idx, 'match'] = 1
                    else:
                        df_processed.at[idx, 'match'] = 0
                else:
                    df_processed.at[idx, 'match'] = 0

            # Add match column for rows that weren't processed above
            match_mask = df_processed['match'].isna()
            if match_mask.any():
                df_processed.loc[match_mask, 'match'] = (df_processed.loc[match_mask, 'linked_qid'] == df_processed.loc[match_mask, 'qid_gold_true']).astype(int)

            # Calculate and log stats for this file
            valid_count = (~df_processed['qid_gold_true'].isna() & (df_processed['qid_gold_true'] != '')).sum()
            linked_count = (~df_processed['linked_qid'].isna() & (df_processed['linked_qid'] != '')).sum()
            matched_count = df_processed['match'].sum()

            if valid_count > 0:
                linking_rate = linked_count / valid_count
                match_rate = matched_count / valid_count
                logging.info(f"File stats - Valid: {valid_count}, Linked: {linked_count} ({linking_rate:.2%}), "
                             f"Matched: {matched_count} ({match_rate:.2%})")

            # Save the cache to disk
            self.cache_manager.save_cache(force=True)

            return df_processed

        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
            logging.error(traceback.format_exc())

            # If we have partial results from a checkpoint, try to create a partial result
            if checkpoint_data and 'answer_to_qid_map' in checkpoint_data:
                try:
                    logging.info(f"Attempting to create partial results from checkpoint")
                    df = pd.read_csv(file_path)
                    df_processed = df.copy()

                    # Add required columns
                    if 'linked_qid' not in df_processed.columns:
                        df_processed['linked_qid'] = None
                    if 'match' not in df_processed.columns:
                        df_processed['match'] = None

                    # Apply the mapping we have so far
                    answer_to_qid_map = checkpoint_data['answer_to_qid_map']
                    for idx, row in df_processed.iterrows():
                        if pd.isna(row['llm_answer']) or row['llm_answer'].strip() == '':
                            continue

                        answer = row['llm_answer']
                        if answer in answer_to_qid_map:
                            df_processed.at[idx, 'linked_qid'] = answer_to_qid_map[answer]

                    # Add match column
                    df_processed['match'] = (df_processed['linked_qid'] == df_processed['qid_gold_true']).astype(int)

                    logging.info(f"Created partial results from checkpoint for {filename}")
                    return df_processed
                except Exception as recovery_e:
                    logging.error(f"Failed to create partial results: {str(recovery_e)}")
                    logging.error(traceback.format_exc())

            return None
    
    def process_folder(self, folder_path: str) -> List[Tuple[str, pd.DataFrame]]:
        """Process all matching files (both NIL and QID) in a folder"""
        # Get the file pattern to use based on test mode
        file_pattern = "*.csv" if not self.test_mode else "*.csv"
        
        # Find all CSV files with 'nil' or 'qid' in their name (case insensitive)
        nil_pattern = os.path.join(folder_path, f"*[nN][iI][lL]*{file_pattern}")
        qid_pattern = os.path.join(folder_path, f"*[qQ][iI][dD]*{file_pattern}")
        
        nil_files = glob.glob(nil_pattern)
        qid_files = glob.glob(qid_pattern)
        all_files = nil_files + qid_files
        
        # Filter out files with '_linked' in the name as they are processed files
        files = [f for f in all_files if '_linked' not in f]
        
        if not files:
            logging.warning(f"No matching files found in {folder_path}")
            return []
        
        logging.info(f"Found {len(files)} files to process in {folder_path}")
        
        results = []
        folder_name = os.path.basename(folder_path)
        file_count = 0
        
        # Set up tqdm progress bar for files if available
        file_iterator = files
        if TQDM_AVAILABLE:
            file_iterator = tqdm(files, desc=f"Folder: {folder_name}",
                               unit="file", leave=True)
        
        for file_path in file_iterator:
            file_count += 1
            filename = os.path.basename(file_path)
            logging.info(f"Processing file {file_count}/{len(files)}: {filename}")
            
            # Process the file
            df_processed = self.process_file(file_path)
            
            if df_processed is None:
                logging.warning(f"Skipping {filename} due to processing errors")
                continue
            
            # Save the processed file
            output_filename = filename.replace('.csv', '_linked.csv')
            
            # Extract approach (RAG or ZS) from the folder path
            folder_parts = file_path.split(os.sep)
            approach_folder = 'Unknown'
            if any('ZS' == part for part in folder_parts):
                approach_folder = 'ZS'
            elif any('RAG' == part for part in folder_parts):
                approach_folder = 'RAG'
            
            # Determine entity type (nil or qid)
            entity_type = 'unknown'
            if any(s in filename.lower() for s in ['nil']):
                entity_type = 'nil'
            elif any(s in filename.lower() for s in ['qid']):
                entity_type = 'qid'
            
            # Create the output path with approach and entity type subfolder
            output_path = os.path.join(self.output_dir, folder_name, approach_folder, entity_type, output_filename)
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the result
            df_processed.to_csv(output_path, index=False)
            logging.info(f"Saved linked file to {output_path}")
            
            # Add to results
            results.append((file_path, df_processed))
            
            # Save cache after each file
            self.cache_manager.save_cache(force=True)
            
            # Save a checkpoint if we're at the interval point
            if file_count % self.checkpoint_interval == 0:
                checkpoint_info = {
                    'folder': folder_path,
                    'completed_files': file_count,
                    'total_files': len(files),
                    'last_file': file_path,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save folder checkpoint
                checkpoint_path = os.path.join(self.checkpoint_dir, f"folder_{folder_name}_checkpoint.json")
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_info, f, ensure_ascii=False, indent=2)
                
                logging.info(f"Folder checkpoint saved: {file_count}/{len(files)} files processed")
        
        return results
    
    def generate_report(self, results: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
        """Generate a report summarizing the results of linking"""
        report_data = []
        
        for file_path, df in results:
            # Extract metrics
            folder_name = os.path.basename(os.path.dirname(file_path))
            filename = os.path.basename(file_path)
            
            # Extract upper folder (RAG or ZS)
            folder_parts = file_path.split(os.sep)
            upper_folder = 'Unknown'
            if 'ZS' in folder_parts:
                upper_folder = 'ZS'
            elif 'RAG' in folder_parts:
                upper_folder = 'RAG'
            
            # Determine entity type (nil or qid)
            entity_type = 'unknown'
            if 'nil' in filename.lower():
                entity_type = 'nil'
            elif 'qid' in filename.lower():
                entity_type = 'qid'
            
            # Calculate metrics
            total_samples = len(df)
            valid_samples = (~df['qid_gold_true'].isna() & (df['qid_gold_true'] != '')).sum()
            linked_samples = (~df['linked_qid'].isna() & (df['linked_qid'] != '')).sum()
            matched_samples = df['match'].sum() if 'match' in df.columns else 0
            
            if valid_samples > 0:
                linking_rate = linked_samples / valid_samples
                match_rate = matched_samples / valid_samples
            else:
                linking_rate = 0
                match_rate = 0
            
            # Extract metadata from filename
            # Example: QA_FamilyName_NIL_output_bge-large_openrouter_google_gemma-2-27b-it_20250505.csv
            # We want to extract retriever (bge-large), model (google_gemma-2-27b-it)
            
            retriever = 'unknown'
            model = 'unknown'
            
            parts = filename.split('_')
            if len(parts) >= 5:
                # Try to extract retriever
                if 'output' in parts:
                    output_idx = parts.index('output')
                    if output_idx + 1 < len(parts):
                        retriever = parts[output_idx + 1]
                
                # Try to extract model name (usually after "openrouter")
                if 'openrouter' in parts:
                    openrouter_idx = parts.index('openrouter')
                    if openrouter_idx + 1 < len(parts):
                        model_parts = []
                        for i in range(openrouter_idx + 1, len(parts)):
                            if parts[i].endswith('.csv'):
                                model_parts.append(parts[i][:-4])  # Remove .csv
                                break
                            else:
                                model_parts.append(parts[i])
                        
                        model = '_'.join(model_parts)
            
            # Add to report
            report_data.append({
                'entity_type': entity_type,
                'upper_folder': upper_folder,
                'folder_name': folder_name,
                'filename': filename,
                'retriever': retriever,
                'model': model,
                'total_samples': total_samples,
                'valid_samples': valid_samples,
                'linked_samples': linked_samples,
                'matched_samples': matched_samples,
                'linking_rate': linking_rate,
                'match_rate': match_rate
            })
        
        # Convert to DataFrame
        report_df = pd.DataFrame(report_data)
        
        # Add additional columns for analysis
        if 'retriever' in report_df.columns and 'model' in report_df.columns:
            report_df['retriever_model'] = report_df['retriever'] + '_' + report_df['model']
        
        return report_df
    
    def generate_comparison_report(self, report_df: pd.DataFrame) -> str:
        """Generate a detailed comparison report with various aggregations"""
        if 'upper_folder' not in report_df.columns or 'entity_type' not in report_df.columns:
            return "Required columns missing in the report. Cannot generate comparison."
        
        # Initialize the report text
        report_text = f"===== {self.config.entity_name.upper()} LINKING COMPARISON REPORT (LOCAL CSV VERSION) =====\n"
        report_text += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 1. Overall statistics
        report_text += "=== OVERALL STATISTICS ===\n"
        total_files = len(report_df)
        total_samples = report_df['total_samples'].sum()
        valid_samples = report_df['valid_samples'].sum()
        linked_samples = report_df['linked_samples'].sum()
        matched_samples = report_df['matched_samples'].sum()
        
        report_text += f"Total files processed: {total_files}\n"
        report_text += f"Total samples: {total_samples}\n"
        report_text += f"Valid samples: {valid_samples}\n"
        report_text += f"Linked samples: {linked_samples}\n"
        report_text += f"Matched samples: {matched_samples}\n"
        
        if valid_samples > 0:
            overall_linking_rate = linked_samples / valid_samples
            overall_match_rate = matched_samples / valid_samples
            report_text += f"Overall linking rate: {overall_linking_rate:.4f}\n"
            report_text += f"Overall match rate: {overall_match_rate:.4f}\n"
        
        # 2. Comparison by entity type (NIL vs QID)
        entity_types = report_df['entity_type'].unique()
        if len(entity_types) > 1:
            report_text += "\n=== COMPARISON BY ENTITY TYPE ===\n"
            
            # Calculate aggregates by entity type
            entity_stats = report_df.groupby('entity_type').agg({
                'total_samples': 'sum',
                'valid_samples': 'sum',
                'linked_samples': 'sum',
                'matched_samples': 'sum'
            }).reset_index()
            
            # Calculate rates
            entity_stats['linking_rate'] = entity_stats['linked_samples'] / entity_stats['valid_samples']
            entity_stats['match_rate'] = entity_stats['matched_samples'] / entity_stats['valid_samples']
            
            # Add to report
            for _, row in entity_stats.iterrows():
                report_text += f"\n{row['entity_type']}:\n"
                report_text += f"  Files: {len(report_df[report_df['entity_type'] == row['entity_type']])}\n"
                report_text += f"  Total samples: {row['total_samples']}\n"
                report_text += f"  Valid samples: {row['valid_samples']}\n"
                report_text += f"  Linked samples: {row['linked_samples']} ({row['linking_rate']:.4f})\n"
                report_text += f"  Matched samples: {row['matched_samples']} ({row['match_rate']:.4f})\n"
        
        # 3. Comparison by approach (ZS vs RAG)
        upper_folders = report_df['upper_folder'].unique()
        if len(upper_folders) > 1:
            report_text += "\n=== COMPARISON BY APPROACH ===\n"
            
            # Calculate aggregates by upper folder
            approach_stats = report_df.groupby('upper_folder').agg({
                'total_samples': 'sum',
                'valid_samples': 'sum',
                'linked_samples': 'sum',
                'matched_samples': 'sum'
            }).reset_index()
            
            # Calculate rates
            approach_stats['linking_rate'] = approach_stats['linked_samples'] / approach_stats['valid_samples']
            approach_stats['match_rate'] = approach_stats['matched_samples'] / approach_stats['valid_samples']
            
            # Add to report
            for _, row in approach_stats.iterrows():
                report_text += f"\n{row['upper_folder']}:\n"
                report_text += f"  Files: {len(report_df[report_df['upper_folder'] == row['upper_folder']])}\n"
                report_text += f"  Total samples: {row['total_samples']}\n"
                report_text += f"  Valid samples: {row['valid_samples']}\n"
                report_text += f"  Linked samples: {row['linked_samples']} ({row['linking_rate']:.4f})\n"
                report_text += f"  Matched samples: {row['matched_samples']} ({row['match_rate']:.4f})\n"
            
            # Find best approach
            best_approach = approach_stats.loc[approach_stats['match_rate'].idxmax()]
            report_text += f"\nBest approach: {best_approach['upper_folder']} with Match Rate = {best_approach['match_rate']:.4f}\n"
            
            # Calculate improvement if we have exactly 2 approaches
            if len(upper_folders) == 2:
                sorted_approaches = approach_stats.sort_values('match_rate')
                baseline = sorted_approaches.iloc[0]
                improved = sorted_approaches.iloc[1]
                
                if baseline['match_rate'] > 0:
                    improvement_pct = (improved['match_rate'] - baseline['match_rate']) / baseline['match_rate'] * 100
                    report_text += f"Improvement of {improved['upper_folder']} over {baseline['upper_folder']}: {improvement_pct:.2f}%\n"
        
        # 4. Approach x Entity Type comparison
        if len(upper_folders) > 1 and len(entity_types) > 1:
            report_text += "\n=== COMPARISON BY APPROACH AND ENTITY TYPE ===\n"
            
            # Calculate aggregates by approach and entity type
            approach_entity_stats = report_df.groupby(['upper_folder', 'entity_type']).agg({
                'total_samples': 'sum',
                'valid_samples': 'sum',
                'linked_samples': 'sum',
                'matched_samples': 'sum'
            }).reset_index()
            
            # Calculate rates
            approach_entity_stats['linking_rate'] = approach_entity_stats['linked_samples'] / approach_entity_stats['valid_samples']
            approach_entity_stats['match_rate'] = approach_entity_stats['matched_samples'] / approach_entity_stats['valid_samples']
            
            # Sort by match rate
            approach_entity_stats = approach_entity_stats.sort_values('match_rate', ascending=False)
            
            # Add to report
            for _, row in approach_entity_stats.iterrows():
                report_text += f"{row['upper_folder']} + {row['entity_type']}:\n"
                report_text += f"  Matching Rate: {row['match_rate']:.4f}\n"
                report_text += f"  Linking Rate: {row['linking_rate']:.4f}\n"
        
        # 5. Comparison by retriever (if applicable)
        if 'retriever' in report_df.columns and len(report_df['retriever'].unique()) > 1:
            report_text += "\n=== COMPARISON BY RETRIEVER ===\n"
            
            retriever_stats = report_df.groupby('retriever').agg({
                'valid_samples': 'sum',
                'linked_samples': 'sum',
                'matched_samples': 'sum'
            }).reset_index()
            
            retriever_stats['linking_rate'] = retriever_stats['linked_samples'] / retriever_stats['valid_samples']
            retriever_stats['match_rate'] = retriever_stats['matched_samples'] / retriever_stats['valid_samples']
            
            # Sort by match rate
            retriever_stats = retriever_stats.sort_values('match_rate', ascending=False)
            
            for _, row in retriever_stats.iterrows():
                report_text += f"{row['retriever']}:\n"
                report_text += f"  Linking Rate: {row['linking_rate']:.4f}\n"
                report_text += f"  Match Rate: {row['match_rate']:.4f}\n\n"
        
        # 6. Comparison by model (if applicable)
        if 'model' in report_df.columns and len(report_df['model'].unique()) > 1:
            report_text += "\n=== COMPARISON BY MODEL ===\n"
            
            model_stats = report_df.groupby('model').agg({
                'valid_samples': 'sum',
                'linked_samples': 'sum',
                'matched_samples': 'sum'
            }).reset_index()
            
            model_stats['linking_rate'] = model_stats['linked_samples'] / model_stats['valid_samples']
            model_stats['match_rate'] = model_stats['matched_samples'] / model_stats['valid_samples']
            
            # Sort by match rate
            model_stats = model_stats.sort_values('match_rate', ascending=False)
            
            for _, row in model_stats.iterrows():
                report_text += f"{row['model']}:\n"
                report_text += f"  Linking Rate: {row['linking_rate']:.4f}\n"
                report_text += f"  Match Rate: {row['match_rate']:.4f}\n\n"
        
        # 7. Combined approach + retriever (if applicable)
        if 'upper_folder' in report_df.columns and 'retriever' in report_df.columns:
            report_text += "\n=== COMPARISON BY APPROACH + RETRIEVER ===\n"
            
            combo_stats = report_df.groupby(['upper_folder', 'retriever']).agg({
                'valid_samples': 'sum',
                'linked_samples': 'sum',
                'matched_samples': 'sum'
            }).reset_index()
            
            combo_stats['linking_rate'] = combo_stats['linked_samples'] / combo_stats['valid_samples']
            combo_stats['match_rate'] = combo_stats['matched_samples'] / combo_stats['valid_samples']
            
            # Sort by match rate
            combo_stats = combo_stats.sort_values('match_rate', ascending=False)
            
            for _, row in combo_stats.iterrows():
                report_text += f"{row['upper_folder']} + {row['retriever']}:\n"
                report_text += f"  Linking Rate: {row['linking_rate']:.4f}\n"
                report_text += f"  Match Rate: {row['match_rate']:.4f}\n\n"
            
            # Highlight best combination
            best_combo = combo_stats.iloc[0]
            report_text += f"Best combination: {best_combo['upper_folder']} + {best_combo['retriever']} with Match Rate = {best_combo['match_rate']:.4f}\n"
        
        # 8. Cache statistics
        report_text += "\n=== CACHE STATISTICS ===\n"
        report_text += f"Total cached mappings: {len(self.cache_manager.string_to_qid_cache)}\n"
        report_text += f"Cache hits: {self.cache_manager.cache_hits}, Cache misses: {self.cache_manager.cache_misses}\n"
        if self.cache_manager.cache_hits + self.cache_manager.cache_misses > 0:
            report_text += f"Cache hit rate: {self.cache_manager.cache_hits / (self.cache_manager.cache_hits + self.cache_manager.cache_misses):.2%}\n"
        report_text += f"Cache file: {self.cache_file}\n"
        
        # 9. Database statistics
        report_text += f"\n=== {self.config.entity_name.upper()} DATABASE STATISTICS ===\n"
        report_text += f"CSV file: {self.entity_db.csv_file_path}\n"
        if hasattr(self.entity_db, 'loaded') and self.entity_db.loaded:
            report_text += f"Total labels loaded: {len(self.entity_db.entities_by_label)}\n"
            report_text += f"Total aliases loaded: {len(self.entity_db.entities_by_alias)}\n"
            report_text += f"Total unique QIDs: {len(self.entity_db.qid_to_label)}\n"
        
        return report_text
    
    def run(self, folders_to_process: List[str]):
        """Run the entity linking process on multiple folders"""
        start_time = time.time()
        
        logging.info("=" * 80)
        logging.info(f"Starting {self.config.entity_name} linking process (LOCAL CSV VERSION)")
        logging.info("=" * 80)
        
        # Log configuration details
        logging.info(f"Config: {self.config.entity_name}")
        logging.info(f"CSV file: {self.config.csv_file_path}")
        logging.info(f"Worker threads: {self.max_workers}, Batch size: {self.batch_size}")
        
        # Load the entity database
        self.entity_db.load()
        
        try:
            # Process each folder
            all_results = []
            
            # Set up tqdm progress bar for folders if available
            folder_iterator = enumerate(folders_to_process)
            if TQDM_AVAILABLE:
                folder_iterator = tqdm(list(enumerate(folders_to_process)),
                                     desc="Overall progress", unit="folder", position=0)
            
            for folder_idx, folder_path in folder_iterator:
                if not os.path.exists(folder_path):
                    logging.warning(f"Folder does not exist: {folder_path}")
                    continue
                
                logging.info(f"Processing folder {folder_idx+1}/{len(folders_to_process)}: {folder_path}")
                folder_results = self.process_folder(folder_path)
                all_results.extend(folder_results)
                
                # Save cache after each folder
                self.cache_manager.save_cache(force=True)
            
            # Generate and save the report
            if all_results:
                # Create report DataFrame
                report_df = self.generate_report(all_results)
                report_path = os.path.join(self.output_dir, self.config.report_name.replace('.csv', '_local.csv'))
                report_df.to_csv(report_path, index=False)
                logging.info(f"Report saved to {report_path}")
                
                # Generate comparison report
                comparison_text = self.generate_comparison_report(report_df)
                comparison_path = os.path.join(self.output_dir, f"{self.config.entity_type}_comparison_local.txt")
                with open(comparison_path, 'w') as f:
                    f.write(comparison_text)
                logging.info(f"Comparison report saved to {comparison_path}")
                
                # Calculate and log overall statistics
                if report_df['valid_samples'].sum() > 0:
                    overall_linking_rate = report_df['linked_samples'].sum() / report_df['valid_samples'].sum()
                    overall_match_rate = report_df['matched_samples'].sum() / report_df['valid_samples'].sum()
                    
                    logging.info("\n" + "=" * 40)
                    logging.info("SUMMARY STATISTICS")
                    logging.info("=" * 40)
                    logging.info(f"Total files processed: {len(all_results)}")
                    logging.info(f"Total samples: {report_df['total_samples'].sum()}")
                    logging.info(f"Valid samples: {report_df['valid_samples'].sum()}")
                    logging.info(f"Linked samples: {report_df['linked_samples'].sum()}")
                    logging.info(f"Matched samples: {report_df['matched_samples'].sum()}")
                    logging.info(f"Overall linking rate: {overall_linking_rate:.4f}")
                    logging.info(f"Overall match rate: {overall_match_rate:.4f}")
            else:
                logging.warning("No files were processed successfully")
        
        except Exception as e:
            logging.error(f"Unexpected error in main process: {str(e)}")
            logging.error(traceback.format_exc())
        
        finally:
            # Save the cache to disk for future runs
            self.cache_manager.save_cache(force=True)
            
            # Calculate total runtime
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            logging.info("=" * 80)
            logging.info(f"Processing completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
            logging.info(f"Saved {len(self.cache_manager.string_to_qid_cache)} entries to cache at {self.cache_file}")
            logging.info("=" * 80)


# Main function to run the scripts from command line
def main():
    """Command-line entry point for the unified entity linking tool"""
    parser = argparse.ArgumentParser(description='Unified Entity Linking Tool')
    
    parser.add_argument('--entity-type', type=str, required=True,
                        choices=['country', 'family_name', 'given_name', 'occupation', 'sex_gender'],
                        help='The type of entity to link')
    
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for linked files and reports')
    
    parser.add_argument('--folders', type=str, nargs='+', required=True,
                        help='Folders to process (space separated)')
    
    parser.add_argument('--max-workers', type=int, default=2,
                        help='Maximum number of worker threads')
    
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for processing')
    
    parser.add_argument('--test-mode', action='store_true',
                        help='Run in test mode with limited files')
    
    parser.add_argument('--fuzzy-threshold', type=float, default=0.85,
                        help='Threshold for fuzzy matching (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Import entity configs based on the selected entity type
    from entity_configs import get_entity_config
    
    # Get the appropriate config
    config = get_entity_config(args.entity_type)
    
    # Create the entity linker
    linker = EntityLinker(
        config=config,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        fuzzy_match_threshold=args.fuzzy_threshold,
        test_mode=args.test_mode
    )
    
    # Run the linking process
    linker.run(args.folders)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Saving cache and exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)