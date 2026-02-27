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
import string as string_mod
import pandas as pd
import logging
import traceback
import unicodedata
import argparse
from pathlib import Path
from collections import Counter
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

# ---------------------------------------------------------------------------
# Relink-mode path constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_EVAL_DATA = _PROJECT_ROOT / "Evaluation" / "data"
_LOOKUP_DIR = _PROJECT_ROOT / "ObjectLinking" / "lookup_tables"

PROPERTY_TO_LOOKUP: Dict[str, Path] = {
    "CoC":        _LOOKUP_DIR / "extracted_country_of_citizenship.csv",
    "FamilyName": _LOOKUP_DIR / "extracted_family_names.csv",
    "GivenName":  _LOOKUP_DIR / "extracted_given_names.csv",
    "occupation":  _LOOKUP_DIR / "extracted_occupations.csv",
    "sexGender":  _LOOKUP_DIR / "extracted_gender.csv",
}


def _is_gold_empty(gold_value) -> bool:
    """Return True when the gold QID is absent (None, NaN, or empty string)."""
    if gold_value is None:
        return True
    if isinstance(gold_value, float) and math.isnan(gold_value):
        return True
    if str(gold_value).strip() == '':
        return True
    return False


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

    @classmethod
    def from_csv_path(cls, csv_path) -> "EntityDatabase":
        """Create an EntityDatabase directly from a CSV path (no EntityConfig needed)."""
        instance = cls.__new__(cls)
        instance.config = None
        instance.csv_file_path = str(csv_path)
        instance.entities_by_label = {}
        instance.entities_by_alias = {}
        instance.qid_to_label = {}
        instance.qid_to_aliases = {}
        instance.loaded = False
        return instance

    def load(self):
        """Load the entities from the CSV file"""
        entity_name = self.config.entity_name if self.config else Path(self.csv_file_path).stem
        logging.info(f"Loading {entity_name} lookup_tables from {self.csv_file_path}")
        
        try:
            # Check if the CSV file exists
            if not os.path.exists(self.csv_file_path):
                logging.warning(f"{entity_name} CSV file not found: {self.csv_file_path}")
                logging.info("Will use cache only")
                self.loaded = True
                return

            df = pd.read_csv(self.csv_file_path)
            total_rows = len(df)

            # Display loading progress
            if TQDM_AVAILABLE:
                iterator = tqdm(df.iterrows(), total=total_rows, desc=f"Loading {entity_name}")
            else:
                iterator = df.iterrows()
                logging.info(f"Loading {total_rows} {entity_name} entries...")
            
            for _, row in iterator:
                qid = row['QID']
                label = row['Label']
                
                # Skip rows with missing QID or label
                if pd.isna(qid) or pd.isna(label):
                    continue
                
                # Store QID -> label mapping
                self.qid_to_label[qid] = label
                
                # Store label -> QID mapping (case insensitive, first match wins)
                label_lower = label.lower()
                if label_lower not in self.entities_by_label:
                    self.entities_by_label[label_lower] = qid

                # Process aliases if available
                aliases = []
                if 'Aliases' in row and not pd.isna(row['Aliases']) and row['Aliases'] != '':
                    # Split aliases by pipe character
                    aliases = row['Aliases'].split('|')

                    # Store each alias -> QID mapping (case insensitive, first match wins)
                    for alias in aliases:
                        if alias and not pd.isna(alias):
                            alias_lower = alias.lower()
                            if alias_lower not in self.entities_by_alias:
                                self.entities_by_alias[alias_lower] = qid
                
                # Store QID -> aliases mapping
                self.qid_to_aliases[qid] = aliases
            
            self.loaded = True
            logging.info(f"Loaded {len(self.entities_by_label)} labels and {len(self.entities_by_alias)} aliases")
            
        except Exception as e:
            logging.error(f"Error loading {entity_name} lookup_tables: {str(e)}")
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

    def lookup_by_label(self, name):
        """Look up a QID by label only (no alias fallback)."""
        if not self.loaded:
            self.load()
        if not name:
            return None
        name = name.strip().lower()
        return self.entities_by_label.get(name)

    def lookup_by_alias(self, name):
        """Look up a QID by alias only (no label fallback)."""
        if not self.loaded:
            self.load()
        if not name:
            return None
        name = name.strip().lower()
        return self.entities_by_alias.get(name)
    
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
        # Phase 1: Check labels only (all cleaning variants)
        logging.info(f"Looking up '{original_answer}' in {self.config.entity_name} database")

        variants = [original_answer, lower_original]
        if cleaned_answer != original_answer:
            variants.extend([cleaned_answer, lower_cleaned])
        if no_punct_answer != cleaned_answer:
            variants.extend([no_punct_answer, lower_no_punct])

        for variant in variants:
            qid = self.entity_db.lookup_by_label(variant)
            if qid:
                break

        # Phase 2: If no label match, check aliases only (all cleaning variants)
        if not qid:
            for variant in variants:
                qid = self.entity_db.lookup_by_alias(variant)
                if qid:
                    break

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

            # Process ALL rows with a non-empty llm_answer (not just those with gold).
            # Rows with empty gold that get a linked_qid are false positives and
            # should be penalised by the evaluation, not silently skipped.

            # Convert any non-string answers to strings and handle missing values
            try:
                df_processed['llm_answer'] = df_processed['llm_answer'].fillna('')
                df_processed['llm_answer'] = df_processed['llm_answer'].astype(str)

                # Build mask: non-empty llm_answer, not too long
                mask = df_processed['llm_answer'].str.strip() != ''
                long_answer_mask = df_processed['llm_answer'].str.len() > 100
                if any(long_answer_mask):
                    logging.info(f"Skipping {long_answer_mask.sum()} rows with llm_answer longer than 100 characters")
                    mask = mask & ~long_answer_mask

                    # Explicitly clear linked_qid for long-answer rows
                    df_processed.loc[long_answer_mask, 'linked_qid'] = ''

                    # Rows with long answer AND no gold → true negative (nothing
                    # to link and we correctly abstained).  Mark match = 1 so they
                    # are not penalised.  Rows with long answer but valid gold →
                    # match = 0 (we failed to link).
                    no_gold = df_processed['qid_gold_true'].isna() | (df_processed['qid_gold_true'].astype(str).str.strip() == '')
                    df_processed.loc[long_answer_mask & no_gold, 'match'] = 1
                    df_processed.loc[long_answer_mask & ~no_gold, 'match'] = 0
            except Exception as e:
                logging.error(f"Error filtering long answers: {str(e)}")
                logging.error(traceback.format_exc())

            df_filtered = df_processed[mask]

            if df_filtered.empty:
                logging.warning(f"No rows with non-empty llm_answer in {filename}")
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
                    # No QID found — true negative if gold is also empty
                    if _is_gold_empty(row['qid_gold_true']):
                        df_processed.at[idx, 'match'] = 1
                    else:
                        df_processed.at[idx, 'match'] = 0

            # Add match column for rows that weren't processed above
            match_mask = df_processed['match'].isna()
            if match_mask.any():
                for idx in df_processed.index[match_mask]:
                    lq = df_processed.at[idx, 'linked_qid']
                    gq = df_processed.at[idx, 'qid_gold_true']
                    lq_empty = _is_gold_empty(lq)
                    gq_empty = _is_gold_empty(gq)
                    if lq_empty and gq_empty:
                        df_processed.at[idx, 'match'] = 1  # true negative
                    elif not lq_empty and not gq_empty and str(lq) == str(gq):
                        df_processed.at[idx, 'match'] = 1  # true positive
                    else:
                        df_processed.at[idx, 'match'] = 0

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

                    # Add match column (true-negative aware)
                    def _compute_match(row):
                        lq = row['linked_qid']
                        gq = row['qid_gold_true']
                        lq_empty = _is_gold_empty(lq)
                        gq_empty = _is_gold_empty(gq)
                        if lq_empty and gq_empty:
                            return 1  # true negative
                        if not lq_empty and not gq_empty and str(lq) == str(gq):
                            return 1  # true positive
                        return 0
                    df_processed['match'] = df_processed.apply(_compute_match, axis=1)

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


# ===========================================================================
# Relink helpers — ported from relink_evaluated.py
# ===========================================================================

def _cleaning_variants(text: str) -> List[str]:
    """Return cleaning variants: original, lowercase, diacritics-removed, no-punct."""
    original = text.strip()
    lower_original = original.lower()
    normalized = unicodedata.normalize("NFKD", original)
    cleaned = "".join(c for c in normalized if not unicodedata.combining(c))
    lower_cleaned = cleaned.lower()
    no_punct = "".join(c for c in cleaned if c not in string_mod.punctuation).strip()
    lower_no_punct = no_punct.lower()

    variants = [original, lower_original]
    if cleaned != original:
        variants.extend([cleaned, lower_cleaned])
    if no_punct != cleaned:
        variants.extend([no_punct, lower_no_punct])
    return variants


def relink_single(db: "EntityDatabase", entity: str) -> Optional[str]:
    """Link a single entity string to a QID: labels first, then aliases."""
    if not entity or len(entity) > 2000:
        return None
    variants = _cleaning_variants(entity)
    for v in variants:
        qid = db.lookup_by_label(v)
        if qid:
            return qid
    for v in variants:
        qid = db.lookup_by_alias(v)
        if qid:
            return qid
    return None


def relink_split_answer(answer: str) -> List[str]:
    """Split a multi-entity answer on common delimiters."""
    if not answer:
        return []
    if any(sep in answer for sep in [";", ":", ",", "/", "|"]):
        parts = re.split(r"[;:,/|]+\s*", answer)
        return [p.strip() for p in parts if p.strip()]
    return [answer.strip()]


def relink_answer(db: "EntityDatabase", answer: str) -> Optional[str]:
    """Link an LLM answer (possibly multi-entity) to QID(s)."""
    if not answer or pd.isna(answer):
        return None
    answer = str(answer).strip()
    if answer == "":
        return None
    entities = relink_split_answer(answer)
    if len(entities) > 1:
        qids = []
        for ent in entities:
            qid = relink_single(db, ent)
            if qid:
                qids.append(qid)
        return ";".join(qids) if qids else None
    return relink_single(db, answer)


def detect_property(file_path: str) -> Optional[str]:
    """Auto-detect property from directory structure."""
    parts = Path(file_path).parts
    for prop in PROPERTY_TO_LOOKUP:
        if prop in parts:
            return prop
    return None


def detect_entity_type(file_path: str) -> Optional[str]:
    """Return 'qid' or 'nil' from the directory structure."""
    parts = Path(file_path).parts
    for p in parts:
        if p in ("qid", "nil"):
            return p
    return None


def _normalise_model(name: str) -> str:
    """Normalise a model string for matching (dots -> hyphens, lowercase)."""
    return name.replace(".", "-").lower()


def _extract_model_from_bm(filename: str) -> Optional[str]:
    """Extract model identifier from a Boyer-Moore per-model filename."""
    m = re.search(r"openrouter_(.+?)_boyer_moore", filename)
    return m.group(1) if m else None


def _extract_model_from_individual(filename: str) -> Optional[str]:
    """Extract model identifier from an individual (non-BM) filename."""
    m = re.search(r"openrouter_(.+?)_\d{8}_linked", filename)
    return m.group(1) if m else None


def find_individual_files(bm_path: Path) -> List[Path]:
    """Find the individual files that feed into a Boyer-Moore voting file.

    For *per_model* files: return all individual files for that model+property+entity_type.
    For *combined* files: return all individual files for that property+entity_type.
    """
    prop = detect_property(str(bm_path))
    etype = detect_entity_type(str(bm_path))
    if not prop or not etype:
        return []

    prop_dir = _EVAL_DATA / prop
    raw: List[Path] = []
    for subdir in ("ZS", "RAG"):
        d = prop_dir / subdir / etype
        if d.exists():
            raw.extend(sorted(d.glob("*_linked_evaluated*.csv")))
    raw = [f for f in raw if "boyer_moore" not in f.name]

    # Deduplicate: keep only shortest name per base stem
    by_base: Dict[str, Path] = {}
    for f in raw:
        base = f.name.split("_linked_evaluated")[0]
        if base not in by_base or len(f.name) < len(by_base[base].name):
            by_base[base] = f
    candidates = sorted(by_base.values())

    if "combined" in bm_path.name:
        return candidates

    bm_model = _extract_model_from_bm(bm_path.name)
    if not bm_model:
        return candidates
    bm_model_norm = _normalise_model(bm_model)

    matched = []
    for f in candidates:
        ind_model = _extract_model_from_individual(f.name)
        if ind_model and _normalise_model(ind_model) == bm_model_norm:
            matched.append(f)
    return matched


def revote_boyer_moore(bm_path: Path, individual_files: List[Path],
                       threshold: float = 0.5) -> bool:
    """Re-derive linked_qid for a Boyer-Moore file from re-linked individual files.

    For each row (matched by 'span'), collect all linked_qids from the individual
    files, count QID frequencies, and keep only QIDs meeting the threshold.

    Returns True if the file was updated.
    """
    bm = pd.read_csv(bm_path, low_memory=False)
    if "span" not in bm.columns:
        logging.warning(f"  No 'span' column in {bm_path.name}, skipping")
        return False

    span_qids: Dict[str, List[str]] = {}
    for fpath in individual_files:
        df = pd.read_csv(fpath, low_memory=False)
        if "span" not in df.columns or "linked_qid" not in df.columns:
            continue
        for _, row in df.iterrows():
            span = row["span"]
            lq = str(row.get("linked_qid", "")).strip()
            if lq and lq != "nan":
                span_qids.setdefault(span, []).extend(lq.split(";"))

    for idx, row in bm.iterrows():
        span = row["span"]
        qids = span_qids.get(span, [])
        total = len(qids)

        if total == 0:
            bm.at[idx, "linked_qid"] = None
            bm.at[idx, "all_qids_voted"] = None
            bm.at[idx, "vote_count"] = 0
            bm.at[idx, "unique_qids"] = 0
            continue

        counts = Counter(qids)
        passing = [qid for qid, cnt in counts.items() if cnt / total >= threshold]
        passing.sort(key=lambda q: int(q[1:]) if q.startswith("Q") and q[1:].isdigit() else float("inf"))

        bm.at[idx, "linked_qid"] = ";".join(passing) if passing else None
        bm.at[idx, "all_qids_voted"] = ";".join(qids)
        bm.at[idx, "vote_count"] = total
        bm.at[idx, "unique_qids"] = len(counts)

    bm.to_csv(bm_path, index=False)
    return True


def cleanup_cascaded_files(data_dir: Path) -> int:
    """Delete cascaded _evaluated_evaluated*.csv files from previous runs."""
    cascaded = list(data_dir.rglob("*_evaluated_evaluated*.csv"))
    for f in cascaded:
        f.unlink()
    return len(cascaded)


def run_relink(eval_data_dir: Optional[Path] = None) -> int:
    """Re-link all _linked_evaluated.csv files using corrected EntityDatabase.

    Phase 1: re-link regular (non-BM) files.
    Phase 2: re-vote Boyer-Moore files from re-linked individual files.
    """
    eval_data = eval_data_dir or _EVAL_DATA

    logging.info("=" * 60)
    logging.info("Re-linking evaluated files with corrected EntityDatabase")
    logging.info("=" * 60)

    # Clean up cascaded files
    deleted = cleanup_cascaded_files(eval_data)
    if deleted:
        logging.info(f"Cleaned up {deleted} cascaded files from previous runs")

    # Load databases via from_csv_path (no EntityConfig needed)
    databases: Dict[str, EntityDatabase] = {}
    for prop, csv_path in PROPERTY_TO_LOOKUP.items():
        if not csv_path.exists():
            logging.error(f"Lookup table not found: {csv_path}")
            return 1
        db = EntityDatabase.from_csv_path(csv_path)
        db.load()
        databases[prop] = db

    # Discover files
    all_files = sorted(eval_data.rglob("*_linked_evaluated.csv"))
    bm_files = [f for f in all_files if "boyer_moore" in f.name]
    regular_files = [f for f in all_files if "boyer_moore" not in f.name]
    logging.info(f"Found {len(all_files)} files: {len(regular_files)} regular, {len(bm_files)} Boyer-Moore")

    # Phase 1
    relinked_count = 0
    skipped_count = 0

    file_iter = tqdm(regular_files, desc="Re-linking files", unit="file") if TQDM_AVAILABLE else regular_files
    for fpath in file_iter:
        prop = detect_property(str(fpath))
        if prop == "DoB" or prop is None:
            skipped_count += 1
            continue

        db = databases[prop]
        df = pd.read_csv(fpath, low_memory=False)

        if "llm_answer" not in df.columns:
            logging.warning(f"  No llm_answer column: {fpath.name}")
            skipped_count += 1
            continue

        df["llm_answer"] = df["llm_answer"].fillna("").astype(str)
        new_linked = []
        for ans in df["llm_answer"]:
            if ans.strip() == "":
                new_linked.append(None)
            elif len(ans) > 100:
                new_linked.append(None)
            else:
                new_linked.append(relink_answer(db, ans))
        df["linked_qid"] = new_linked

        # Reset stale evaluation columns — they were computed with the
        # previous linked_qid and are now invalid.  The evaluation script
        # will recompute them.
        for col in ("TP", "FP", "FN", "TN", "match"):
            if col in df.columns:
                df[col] = 0

        df.to_csv(fpath, index=False)
        relinked_count += 1

    logging.info(f"Phase 1 done. Re-linked: {relinked_count}, Skipped: {skipped_count}")

    # Phase 2
    revoted_count = 0
    bm_skipped = 0

    bm_iter = tqdm(bm_files, desc="Re-voting BM files", unit="file") if TQDM_AVAILABLE else bm_files
    for bm_path in bm_iter:
        prop = detect_property(str(bm_path))
        if prop == "DoB":
            bm_skipped += 1
            continue

        indiv = find_individual_files(bm_path)
        if not indiv:
            logging.warning(f"  No individual files found for {bm_path.name}")
            bm_skipped += 1
            continue

        if revote_boyer_moore(bm_path, indiv):
            revoted_count += 1
        else:
            bm_skipped += 1

    logging.info("=" * 60)
    logging.info(f"Done. Re-linked: {relinked_count}, Re-voted: {revoted_count}, "
                 f"Skipped: {skipped_count + bm_skipped}")
    logging.info("=" * 60)
    return 0


# Main function to run the scripts from command line
def main():
    """Command-line entry point with subcommands: link (original) and relink (new)."""
    parser = argparse.ArgumentParser(
        description='Unified Entity Linking Tool',
    )
    subparsers = parser.add_subparsers(dest='command')

    # ---- "link" subcommand (original behaviour) ----------------------------
    link_parser = subparsers.add_parser('link', help='Link entities in raw CSV files')
    link_parser.add_argument('--entity-type', type=str, required=True,
                             choices=['country', 'family_name', 'given_name', 'occupation', 'sex_gender'],
                             help='The type of entity to link')
    link_parser.add_argument('--output-dir', type=str, default=None,
                             help='Output directory for linked files and reports')
    link_parser.add_argument('--folders', type=str, nargs='+', required=True,
                             help='Folders to process (space separated)')
    link_parser.add_argument('--max-workers', type=int, default=2,
                             help='Maximum number of worker threads')
    link_parser.add_argument('--batch-size', type=int, default=10,
                             help='Batch size for processing')
    link_parser.add_argument('--test-mode', action='store_true',
                             help='Run in test mode with limited files')
    link_parser.add_argument('--fuzzy-threshold', type=float, default=0.85,
                             help='Threshold for fuzzy matching (0.0-1.0)')

    # ---- "relink" subcommand (replaces relink_evaluated.py) ----------------
    relink_parser = subparsers.add_parser(
        'relink',
        help='Re-link all _linked_evaluated.csv files in Evaluation/data/',
    )
    relink_parser.add_argument('--eval-data-dir', type=str, default=None,
                               help='Override default Evaluation/data/ directory')

    # ---- Backward compatibility: no subcommand but --entity-type present ----
    # Also add top-level arguments so the old CLI still works.
    parser.add_argument('--entity-type', type=str,
                        choices=['country', 'family_name', 'given_name', 'occupation', 'sex_gender'],
                        help='(Legacy) The type of entity to link')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='(Legacy) Output directory')
    parser.add_argument('--folders', type=str, nargs='+',
                        help='(Legacy) Folders to process')
    parser.add_argument('--max-workers', type=int, default=2,
                        help='(Legacy) Maximum worker threads')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='(Legacy) Batch size')
    parser.add_argument('--test-mode', action='store_true',
                        help='(Legacy) Test mode')
    parser.add_argument('--fuzzy-threshold', type=float, default=0.85,
                        help='(Legacy) Fuzzy matching threshold')

    args = parser.parse_args()

    # Determine which mode to run
    if args.command == 'relink':
        eval_dir = Path(args.eval_data_dir) if args.eval_data_dir else None
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        sys.exit(run_relink(eval_dir))

    # "link" subcommand or legacy (no subcommand but --entity-type present)
    entity_type = getattr(args, 'entity_type', None)
    folders = getattr(args, 'folders', None)
    if entity_type and folders:
        from entity_configs import get_entity_config
        config = get_entity_config(entity_type)
        linker = EntityLinker(
            config=config,
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            fuzzy_match_threshold=args.fuzzy_threshold,
            test_mode=args.test_mode,
        )
        linker.run(folders)
    else:
        parser.print_help()
        sys.exit(1)


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