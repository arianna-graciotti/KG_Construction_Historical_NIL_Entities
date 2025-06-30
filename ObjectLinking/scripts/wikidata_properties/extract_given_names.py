#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract given names from a Wikidata JSON dump based on specific P31 (instance of) values.

This scripts efficiently processes the large Wikidata JSON dump and extracts
entities that are instances of given name related QIDs. It saves the results in a CSV file
with QID, label, and aliases.

Processes the dump line by line, using low memory and supporting resumable processing.
"""

import json
import csv
import gzip
import os
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Target P31 values for given names
TARGET_P31_VALUES = [
    "Q98775491",  # Sex-neutral name
    "Q12308941",  # Male given name
    "Q108709",    # Female given name
    "Q3409032",   # Unisex personal name
    "Q11879590",  # Given name
]

def create_checkpoint_file(checkpoint_path, line_number, processed_count, matched_count):
    """Create a checkpoint file to allow resuming processing."""
    with open(checkpoint_path, 'w') as f:
        checkpoint = {
            'line_number': line_number,
            'processed_count': processed_count,
            'matched_count': matched_count,
            'timestamp': datetime.now().isoformat()
        }
        json.dump(checkpoint, f)
    logger.info(f"Checkpoint saved: {line_number} lines processed, {matched_count} matches found")

def load_checkpoint(checkpoint_path):
    """Load processing checkpoint if it exists."""
    if not os.path.exists(checkpoint_path):
        return 0, 0, 0
    
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
            logger.info(f"Resuming from checkpoint: {checkpoint['line_number']} lines, {checkpoint['matched_count']} matches")
            return checkpoint['line_number'], checkpoint['processed_count'], checkpoint['matched_count']
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Checkpoint file is corrupted, starting from beginning: {e}")
        return 0, 0, 0

def get_english_label(labels):
    """Extract English label from labels dictionary, with fallbacks."""
    if not labels:
        return ""
    
    # Try English label first
    if "en" in labels:
        return labels["en"]["value"]
    
    # Try other common languages
    for lang in ["en-gb", "en-us", "en-ca", "en-au"]:
        if lang in labels:
            return labels[lang]["value"]
    
    # Otherwise, take the first label
    for lang, label_data in labels.items():
        return label_data["value"]  # Return the first one we find
    
    return ""

def get_english_aliases(aliases):
    """Extract English aliases from aliases dictionary."""
    if not aliases or "en" not in aliases:
        return []
    
    alias_list = []
    for alias_item in aliases["en"]:
        if "value" in alias_item:
            alias_list.append(alias_item["value"])
    
    return alias_list

def process_entity(entity, target_p31_values):
    """Process a single entity and check if it matches our criteria."""
    if not entity or "id" not in entity or not entity["id"].startswith("Q"):
        return None
    
    # Check if entity has claims and P31 property
    if "claims" not in entity or "P31" not in entity["claims"]:
        return None
    
    # Check if any P31 value matches our target values
    p31_claims = entity["claims"]["P31"]
    for claim in p31_claims:
        if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
            datavalue = claim["mainsnak"]["datavalue"]
            if "value" in datavalue and "id" in datavalue["value"]:
                if datavalue["value"]["id"] in target_p31_values:
                    # Entity matches, extract required lookup_tables
                    qid = entity["id"]
                    label = get_english_label(entity.get("labels", {}))
                    aliases = get_english_aliases(entity.get("aliases", {}))
                    aliases_str = "|".join(aliases)
                    return [qid, label, aliases_str]
    
    return None

def process_wikidata_dump(input_file, output_file, checkpoint_file, 
                          batch_size=10000, max_entities=None, skip_to_line=0):
    """Process the Wikidata dump file and extract matching entities."""
    line_number = 0
    processed_count = 0
    matched_count = 0
    
    # Check if we should resume from checkpoint
    if os.path.exists(checkpoint_file):
        line_number, processed_count, matched_count = load_checkpoint(checkpoint_file)
    
    # Create/append to output file
    mode = 'a' if line_number > 0 else 'w'
    with gzip.open(input_file, 'rt', encoding='utf-8') as dump_file, \
         open(output_file, mode, newline='', encoding='utf-8') as csv_file:
        
        # Setup CSV writer
        csv_writer = csv.writer(csv_file)
        
        # Write header if new file
        if mode == 'w':
            csv_writer.writerow(["QID", "Label", "Aliases"])
        
        # Skip to the checkpoint position
        if line_number > 0:
            logger.info(f"Skipping to line {line_number}...")
            for _ in range(line_number):
                next(dump_file, None)
        
        # Process the dump line by line
        start_time = time.time()
        last_checkpoint_time = start_time
        
        try:
            # Skip the first line (opening bracket)
            if line_number == 0:
                dump_file.readline()
                line_number = 1
            
            for line in dump_file:
                line_number += 1
                
                # Progress reporting
                if line_number % 10000 == 0:
                    elapsed = time.time() - start_time
                    entities_per_sec = line_number / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {line_number} lines, "
                                f"matched {matched_count} entities, "
                                f"speed: {entities_per_sec:.2f} lines/sec")
                
                # Checkpoint every 10 minutes
                if time.time() - last_checkpoint_time > 600:  # 10 minutes
                    create_checkpoint_file(checkpoint_file, line_number, processed_count, matched_count)
                    last_checkpoint_time = time.time()
                
                # Check if we've reached the maximum
                if max_entities and matched_count >= max_entities:
                    logger.info(f"Reached maximum entities limit: {max_entities}")
                    break
                
                # Clean and parse the line
                if line.strip() in ['[', ']']:
                    continue
                
                line = line.strip().rstrip(',')
                if not line:
                    continue
                
                # Parse entity JSON
                try:
                    entity = json.loads(line)
                    processed_count += 1
                    
                    # Process the entity
                    result = process_entity(entity, TARGET_P31_VALUES)
                    if result:
                        csv_writer.writerow(result)
                        matched_count += 1
                        
                        # Additional progress report for matches
                        if matched_count % 1000 == 0:
                            logger.info(f"Found {matched_count} matching entities so far")
                            csv_file.flush()  # Ensure lookup_tables is written to disk
                    
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON at line {line_number}")
                except Exception as e:
                    logger.error(f"Error processing entity at line {line_number}: {e}")
        
        except KeyboardInterrupt:
            logger.info("Process interrupted by user")
        finally:
            # Save final checkpoint
            create_checkpoint_file(checkpoint_file, line_number, processed_count, matched_count)
            
            # Final statistics
            elapsed = time.time() - start_time
            logger.info(f"Processing complete: {line_number} lines processed")
            logger.info(f"Found {matched_count} entities matching the criteria")
            logger.info(f"Total time: {elapsed:.2f} seconds")
            if elapsed > 0:
                logger.info(f"Average processing speed: {line_number / elapsed:.2f} lines/sec")

def main():
    """Main function to parse arguments and run processing."""
    parser = argparse.ArgumentParser(description="Extract given names from Wikidata dump based on P31 values")
    parser.add_argument("--input", default="/media/arianna/Volume/latest-all.json.gz",
                      help="Path to Wikidata JSON dump (compressed with gzip)")
    parser.add_argument("--output", default="lookup_tables/extracted_given_names.csv",
                      help="Path to output CSV file")
    parser.add_argument("--checkpoint", default="wikidata_given_names_checkpoint.json",
                      help="Path to checkpoint file for resumable processing")
    parser.add_argument("--batch-size", type=int, default=10000,
                      help="Number of entities to process before checkpointing (default: 10000)")
    parser.add_argument("--max-entities", type=int, default=None,
                      help="Maximum number of entities to extract (default: no limit)")
    parser.add_argument("--language", default="en",
                      help="Language code for labels and aliases (default: en)")
    parser.add_argument("--reset", action="store_true",
                      help="Ignore checkpoint and start from beginning")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Check file size and warn if very large
    input_size_gb = os.path.getsize(args.input) / (1024**3)
    logger.info(f"Input file size: {input_size_gb:.2f} GB")
    if input_size_gb > 50:
        logger.warning("The input file is very large. Processing may take a long time.")
    
    # Delete checkpoint if reset requested
    if args.reset and os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
        logger.info("Deleted existing checkpoint file.")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process the dump
    logger.info(f"Starting extraction of given names from: {args.input}")
    logger.info(f"Looking for entities with P31 values: {', '.join(TARGET_P31_VALUES)}")
    
    process_wikidata_dump(
        args.input,
        args.output,
        args.checkpoint,
        batch_size=args.batch_size,
        max_entities=args.max_entities
    )
    
    logger.info(f"Extraction complete. Results saved to: {args.output}")
    return 0

if __name__ == "__main__":
    main()