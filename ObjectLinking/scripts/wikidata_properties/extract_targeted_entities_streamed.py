#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract entities from a Wikidata JSON dump based on P31 (instance of) values using streaming.

This scripts uses an extremely memory-efficient approach with ijson to process
the huge Wikidata dump without loading the entire file into memory. It extracts
entities that are instances of specified QIDs and saves the results with QID, label, and aliases.

Installation:
    pip install ijson  # For streaming JSON parsing
"""

import csv
import gzip
import os
import time
import argparse
import logging
import ijson
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Target P31 values we're interested in
TARGET_P31_VALUES = [
    "Q101352",    # Family name
    "Q18972245",  # Given name
    "Q18972207",  # Name used for males
    "Q11455398",  # Name used for females
    "Q98775491",  # Sex-neutral name
    "Q829026",    # Unisex name
    "Q27951364",  # Human name
]

def save_progress(output_file, processed_count, matched_count, elapsed_time):
    """Save progress statistics to a file."""
    stats = {
        'processed_count': processed_count,
        'matched_count': matched_count,
        'elapsed_time': elapsed_time,
        'timestamp': datetime.now().isoformat(),
        'lines_per_second': processed_count / elapsed_time if elapsed_time > 0 else 0,
        'matches_per_second': matched_count / elapsed_time if elapsed_time > 0 else 0
    }
    
    progress_file = f"{output_file}.progress"
    with open(progress_file, 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Progress saved: {processed_count} entities processed, {matched_count} matches found")

def get_english_string(values_dict, key="value"):
    """Extract English value from a dictionary of language-keyed values."""
    if not values_dict:
        return ""
    
    # Try English first
    if "en" in values_dict:
        return values_dict["en"][key]
    
    # Try other English variations
    for lang in ["en-gb", "en-us", "en-ca", "en-au"]:
        if lang in values_dict:
            return values_dict[lang][key]
    
    # Otherwise, take the first one
    for lang, value in values_dict.items():
        return value[key]
    
    return ""

def get_english_aliases(aliases_dict):
    """Extract English aliases from a dictionary of language-keyed aliases."""
    if not aliases_dict or "en" not in aliases_dict:
        return []
    
    return [alias["value"] for alias in aliases_dict["en"] if "value" in alias]

def process_wikidata_dump_streamed(input_file, output_file, progress_interval=10000, max_entities=None):
    """
    Process the Wikidata dump file using streaming JSON parser (ijson).
    This minimizes memory usage for handling extremely large dumps.
    """
    processed_count = 0
    matched_count = 0
    
    # Check if output exists and determine mode
    file_exists = os.path.exists(output_file)
    mode = 'a' if file_exists else 'w'
    
    # Open files and setup CSV writer
    with gzip.open(input_file, 'rb') as dump_file, \
         open(output_file, mode, newline='', encoding='utf-8') as csv_file:
        
        csv_writer = csv.writer(csv_file)
        
        # Write header if new file
        if mode == 'w':
            csv_writer.writerow(["QID", "Label", "Aliases"])
        
        start_time = time.time()
        last_progress_time = start_time
        
        try:
            # Use ijson to parse JSON objects one at a time
            parser = ijson.items(dump_file, 'item')
            
            for entity in parser:
                processed_count += 1
                
                # Progress reporting
                if processed_count % progress_interval == 0:
                    elapsed = time.time() - start_time
                    entities_per_sec = processed_count / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {processed_count} entities, "
                                f"matched {matched_count}, "
                                f"speed: {entities_per_sec:.2f} entities/sec")
                
                # Periodic progress saves (every 10 minutes)
                if time.time() - last_progress_time > 600:
                    elapsed = time.time() - start_time
                    save_progress(output_file, processed_count, matched_count, elapsed)
                    last_progress_time = time.time()
                    # Also flush CSV to ensure lookup_tables is written
                    csv_file.flush()
                
                # Check if we've reached the maximum
                if max_entities and matched_count >= max_entities:
                    logger.info(f"Reached maximum entities limit: {max_entities}")
                    break
                
                # Skip if not a QID entity
                if "id" not in entity or not entity["id"].startswith("Q"):
                    continue
                
                # Check for P31 (instance of) claims
                if "claims" not in entity or "P31" not in entity["claims"]:
                    continue
                
                # Check for matching P31 values
                p31_claims = entity["claims"]["P31"]
                for claim in p31_claims:
                    if "mainsnak" not in claim or "datavalue" not in claim["mainsnak"]:
                        continue
                    
                    datavalue = claim["mainsnak"]["datavalue"]
                    if "value" not in datavalue or "id" not in datavalue["value"]:
                        continue
                    
                    # If P31 value matches one of our targets
                    if datavalue["value"]["id"] in TARGET_P31_VALUES:
                        # Extract entity lookup_tables
                        qid = entity["id"]
                        
                        # Get English label
                        label = ""
                        if "labels" in entity and "en" in entity["labels"]:
                            label = entity["labels"]["en"]["value"]
                        
                        # Get English aliases
                        aliases = []
                        if "aliases" in entity and "en" in entity["aliases"]:
                            aliases = [alias["value"] for alias in entity["aliases"]["en"] 
                                      if "value" in alias]
                        
                        # Join aliases with pipe character
                        aliases_str = "|".join(aliases)
                        
                        # Write to CSV
                        csv_writer.writerow([qid, label, aliases_str])
                        matched_count += 1
                        
                        # Additional progress report for matches
                        if matched_count % 1000 == 0:
                            logger.info(f"Found {matched_count} matching entities so far")
                        
                        # We've found a match, no need to check other P31 claims
                        break
        
        except KeyboardInterrupt:
            logger.info("Process interrupted by user")
        except Exception as e:
            logger.error(f"Error processing dump: {e}")
        finally:
            # Save final progress
            elapsed = time.time() - start_time
            save_progress(output_file, processed_count, matched_count, elapsed)
            
            # Final statistics
            logger.info(f"Processing complete: {processed_count} entities processed")
            logger.info(f"Found {matched_count} entities matching the criteria")
            logger.info(f"Total time: {elapsed:.2f} seconds")
            if elapsed > 0:
                logger.info(f"Average processing speed: {processed_count / elapsed:.2f} entities/sec")

def main():
    """Main function to parse arguments and run processing."""
    parser = argparse.ArgumentParser(
        description="Extract entities from Wikidata dump based on P31 values using streaming")
    parser.add_argument("--input", default="latest-all.json.gz",
                      help="Path to Wikidata JSON dump (compressed with gzip)")
    parser.add_argument("--output", default="extracted_entities_streamed.csv",
                      help="Path to output CSV file")
    parser.add_argument("--progress-interval", type=int, default=10000,
                      help="Number of entities to process before reporting progress (default: 10000)")
    parser.add_argument("--max-entities", type=int, default=None,
                      help="Maximum number of entities to extract (default: no limit)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Check file size and warn if very large
    input_size_gb = os.path.getsize(args.input) / (1024**3)
    logger.info(f"Input file size: {input_size_gb:.2f} GB")
    if input_size_gb > 50:
        logger.warning("The input file is very large. Processing may take several hours.")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process the dump
    logger.info(f"Starting streamed extraction of entities from: {args.input}")
    logger.info(f"Looking for entities with P31 values: {', '.join(TARGET_P31_VALUES)}")
    
    process_wikidata_dump_streamed(
        args.input,
        args.output,
        progress_interval=args.progress_interval,
        max_entities=args.max_entities
    )
    
    logger.info(f"Extraction complete. Results saved to: {args.output}")
    return 0

if __name__ == "__main__":
    main()