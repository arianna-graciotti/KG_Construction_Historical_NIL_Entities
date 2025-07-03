#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Wikidata Entity Extraction Tool

This scripts provides a flexible way to extract entities from a Wikidata JSON dump
based on P31 (instance of) claims. You can specify:
1. Path to the Wikidata dump
2. Output CSV file path
3. List of QIDs that you want to extract (entities that are instances of these QIDs)

The scripts automatically chooses the most efficient extraction method based on
your system's available memory and outputs a CSV file with QID, label, and aliases.

Usage examples:
  # Extract family names
  python extract_wikidata_entities.py --input /path/to/latest-all.json.gz \
    --output lookup_tables/extracted_family_names.csv \
    --qids Q101352 Q18972245 Q18972207 Q11455398 Q98775491 Q829026 Q27951364
    
  # Extract given names
  python extract_wikidata_entities.py --input /path/to/latest-all.json.gz \
    --output lookup_tables/extracted_given_names.csv \
    --qids Q12308941 Q108709 Q11879590 Q3409032 Q12787828
    
  # Extract countries
  python extract_wikidata_entities.py --input /path/to/latest-all.json.gz \
    --output lookup_tables/extracted_countries.csv \
    --qids Q6256 Q3624078 Q3336843
    
  # Extract occupations
  python extract_wikidata_entities.py --input /path/to/latest-all.json.gz \
    --output lookup_tables/extracted_occupations.csv \
    --qids Q12737077 Q28640 Q12371902 Q40348 Q170790
    
  # Extract sex/gender
  python extract_wikidata_entities.py --input /path/to/latest-all.json.gz \
    --output lookup_tables/extracted_sex_gender.csv \
    --qids Q48277 Q290 Q6581097 Q6581072 Q1097630 Q1052281 Q2449503
"""

import os
import sys
import csv
import gzip
import json
import time
import argparse
import logging
import psutil
import ijson
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import deque

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def get_available_memory_gb():
    """Get available system memory in GB."""
    return psutil.virtual_memory().available / (1024**3)

def get_script_directory():
    """Get directory where this scripts is located."""
    return os.path.dirname(os.path.abspath(__file__))

def save_progress(output_file: str, processed_count: int, matched_count: int, elapsed_time: float):
    """Save progress statistics to a file."""
    stats = {
        'processed_count': processed_count,
        'matched_count': matched_count,
        'elapsed_time': elapsed_time,
        'timestamp': datetime.now().isoformat(),
        'lines_per_second': processed_count / elapsed_time if elapsed_time > 0 else 0,
    }
    
    checkpoint_file = f"{os.path.splitext(output_file)[0]}_checkpoint.json"
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Progress saved to {checkpoint_file}")

def extract_with_streaming(input_file: str, output_file: str, target_p31_values: List[str], 
                         max_entities: Optional[int] = None, checkpoint_interval: int = 100000,
                         memory_limit: Optional[float] = None):
    """
    Extract entities from a Wikidata dump using streaming approach (low memory usage).
    
    Args:
        input_file: Path to Wikidata JSON dump (compressed with gzip)
        output_file: Path to output CSV file
        target_p31_values: List of QIDs to extract (entities that are instances of these QIDs)
        max_entities: Maximum number of entities to extract (default: no limit)
        checkpoint_interval: How often to save progress checkpoint (in entities processed)
    """
    logger.info(f"Starting extraction with streaming parser")
    logger.info(f"Target P31 values: {target_p31_values}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Create CSV file and write header
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['QID', 'Label', 'Aliases'])
        
        start_time = time.time()
        processed_count = 0
        matched_count = 0
        target_p31_set = set(target_p31_values)
        
        # Use a completely different approach that's more reliable
        # Instead of using ijson directly on the gzipped file, use a preprocessing step
        
        # Check if the input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        # Set up external process to properly prepare the JSON stream
        logger.info(f"Using preprocessing pipeline to handle JSON format")
        # Use our dedicated Python helper scripts for more reliable streaming
        helper_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stream_helper.py")
        cmd = f"zcat '{input_file}' | python {helper_script}"
        
        # Use a more memory-efficient approach - directly decode from compressed file
        # This avoids loading the entire JSON content into memory at once
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1)
        
        # Create a parser for JSON objects in the processed stream with a memory-efficient backend
        # Use the 'yajl2_c' backend if available, as it's more memory-efficient
        try:
            parser = ijson.parse(process.stdout, use_float=True)
            logger.info("Using standard ijson parser")
        except Exception as e:
            logger.warning(f"Error creating parser: {e}. Falling back to basic parser.")
            parser = ijson.basic_parse(process.stdout)
        
        # Use a try-finally to ensure we clean up the process
        try:
            
            # Variables to track the current entity
            current_id = None
            current_p31_values = set()
            current_label = None
            current_aliases = []
            
            # Track state for JSON parsing
            in_claims = False
            in_p31 = False
            in_p31_value = False
            in_labels = False
            in_en_label = False
            in_aliases = False
            in_en_aliases = False
            in_alias_value = False
            
            # Process the JSON stream
            for prefix, event, value in parser:
                # Process each entity in the stream
                if prefix == "item" and event == "start_map":
                    # Start of a new entity
                    current_id = None
                    current_p31_values = set()
                    current_label = None
                    current_aliases = []
                    in_claims = False
                    in_p31 = False
                    in_p31_value = False
                    in_labels = False
                    in_en_label = False
                    in_aliases = False
                    in_en_aliases = False
                    in_alias_value = False
                
                elif prefix == "item" and event == "end_map":
                    # End of the current entity
                    processed_count += 1
                    
                    # Check if this entity matches our criteria
                    if current_p31_values & target_p31_set and current_id and current_label:
                        matched_count += 1
                        # Write the entity to the CSV file
                        writer.writerow([
                            current_id,
                            current_label,
                            "|".join(current_aliases) if current_aliases else ""
                        ])
                        
                        # Flush occasionally to save progress
                        if matched_count % 1000 == 0:
                            csvfile.flush()
                    
                    # Save progress checkpoint periodically
                    if processed_count % checkpoint_interval == 0:
                        elapsed_time = time.time() - start_time
                        logger.info(f"Processed {processed_count} entities, matched {matched_count}, "
                                   f"elapsed time: {elapsed_time:.2f}s, "
                                   f"rate: {processed_count/elapsed_time:.2f} entities/s")
                        save_progress(output_file, processed_count, matched_count, elapsed_time)
                        
                        # Periodically force garbage collection to prevent memory issues
                        import gc
                        gc.collect()
                        
                        # Check memory usage if a limit is set
                        if memory_limit:
                            current_memory = get_available_memory_gb()
                            memory_used = psutil.Process().memory_info().rss / (1024**3)
                            logger.info(f"Current memory usage: {memory_used:.2f} GB, Available: {current_memory:.2f} GB")
                            
                            # If memory usage is getting high, take more aggressive measures
                            if memory_used > memory_limit * 0.8:
                                logger.warning(f"Memory usage ({memory_used:.2f} GB) approaching limit, forcing additional garbage collection")
                                # Run a more thorough garbage collection
                                for _ in range(3):
                                    gc.collect()
                        
                    # Stop if we've reached the maximum number of entities
                    if max_entities and matched_count >= max_entities:
                        logger.info(f"Reached maximum of {max_entities} matched entities, stopping")
                        break
                
                # Track entity ID
                elif prefix == "item.id" and event == "string":
                    current_id = value
                
                # Track P31 values (instance of)
                elif prefix == "item.claims" and event == "start_map":
                    in_claims = True
                elif in_claims and prefix == "item.claims.P31" and event == "start_array":
                    in_p31 = True
                elif in_p31 and prefix.endswith(".mainsnak.datavalue.value.id") and event == "string":
                    current_p31_values.add(value)
                elif in_claims and prefix == "item.claims.P31" and event == "end_array":
                    in_p31 = False
                elif prefix == "item.claims" and event == "end_map":
                    in_claims = False
                
                # Track English label
                elif prefix == "item.labels" and event == "start_map":
                    in_labels = True
                elif in_labels and prefix == "item.labels.en" and event == "start_map":
                    in_en_label = True
                elif in_en_label and prefix == "item.labels.en.value" and event == "string":
                    current_label = value
                elif in_labels and prefix == "item.labels.en" and event == "end_map":
                    in_en_label = False
                elif prefix == "item.labels" and event == "end_map":
                    in_labels = False
                
                # Track English aliases
                elif prefix == "item.aliases" and event == "start_map":
                    in_aliases = True
                elif in_aliases and prefix == "item.aliases.en" and event == "start_array":
                    in_en_aliases = True
                elif in_en_aliases and prefix.endswith(".value") and event == "string":
                    current_aliases.append(value)
                elif in_aliases and prefix == "item.aliases.en" and event == "end_array":
                    in_en_aliases = False
                elif prefix == "item.aliases" and event == "end_map":
                    in_aliases = False
            
            # Final progress update
            elapsed_time = time.time() - start_time
            logger.info(f"Finished processing! Processed {processed_count} entities, matched {matched_count}, "
                       f"elapsed time: {elapsed_time:.2f}s, "
                       f"rate: {processed_count/elapsed_time:.2f} entities/s")
            save_progress(output_file, processed_count, matched_count, elapsed_time)
        
        except Exception as e:
            logger.error(f"Error during extraction: {str(e)}")
            raise
            
        finally:
            # Always clean up the subprocess if it exists and is still running
            if 'process' in locals() and process.poll() is None:
                process.terminate()

def has_matching_p31(entity: Dict[str, Any], target_p31_set: set) -> bool:
    """Check if an entity has any matching P31 values."""
    if 'claims' not in entity or 'P31' not in entity['claims']:
        return False
    
    for claim in entity['claims']['P31']:
        if 'mainsnak' in claim and 'datavalue' in claim['mainsnak'] and \
           'value' in claim['mainsnak']['datavalue'] and \
           'id' in claim['mainsnak']['datavalue']['value'] and \
           claim['mainsnak']['datavalue']['value']['id'] in target_p31_set:
            return True
    
    return False

def process_chunk(chunk: List[Dict[str, Any]], writer: csv.writer, target_p31_set: set) -> int:
    """Process a chunk of entities and write matching ones to CSV.
    
    Args:
        chunk: List of entity dictionaries to process
        writer: CSV writer to write matching entities
        target_p31_set: Set of target P31 QIDs to match against
        
    Returns:
        int: Number of matching entities written to CSV
    """
    matched_count = 0
    for entity in chunk:
        try:
            # Validate entity structure
            if 'id' not in entity or not isinstance(entity.get('claims'), dict):
                continue
                
            if not has_matching_p31(entity, target_p31_set):
                continue
            
            entity_id = entity.get('id')
            
            # Extract English label
            label = None
            if 'labels' in entity and 'en' in entity['labels']:
                label = entity['labels']['en'].get('value')
            
            # Skip if no ID or label
            if not entity_id or not label:
                continue
            
            # Extract English aliases
            aliases = []
            if 'aliases' in entity and 'en' in entity['aliases']:
                for alias in entity['aliases']['en']:
                    if 'value' in alias:
                        aliases.append(alias['value'])
            
            # Write to CSV
            writer.writerow([
                entity_id,
                label,
                '|'.join(aliases)
            ])
            
            matched_count += 1
        except Exception as e:
            logger.error(f"Error processing entity: {e}")
            continue
            
    return matched_count

def extract_with_full_loading(input_file: str, output_file: str, target_p31_values: List[str],
                            max_entities: Optional[int] = None):
    """
    Extract entities from a Wikidata dump by loading chunks into memory (higher memory usage but faster).
    
    Args:
        input_file: Path to Wikidata JSON dump (compressed with gzip)
        output_file: Path to output CSV file
        target_p31_values: List of QIDs to extract (entities that are instances of these QIDs)
        max_entities: Maximum number of entities to extract (default: no limit)
    """
    logger.info(f"Starting extraction with partial loading parser")
    logger.info(f"Target P31 values: {target_p31_values}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Set for faster lookups
    target_p31_set = set(target_p31_values)
    
    # Create CSV file and write header
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['QID', 'Label', 'Aliases'])
        
        start_time = time.time()
        processed_count = 0
        matched_count = 0
        
        try:
            # Process dump file
            with gzip.open(input_file, 'rt', encoding='utf-8') as f:
                # Skip to the opening bracket, accounting for potential whitespace
                found_bracket = False
                while not found_bracket:
                    char = f.read(1)
                    if not char:
                        logger.error("Reached end of file without finding array start '['")
                        return
                    if char == '[':
                        found_bracket = True
                
                # Read and process the file in chunks
                chunk_size = 10000  # Process 10,000 entities per chunk
                chunk = deque(maxlen=chunk_size)
                entity_buffer = ""
                depth = 0
                in_entity = False
                
                # Use a more robust way to parse JSON entities with string awareness
                in_string = False
                escape_next = False

                for line in f:
                    # Process each line
                    for char in line:
                        if not in_entity and char == '{':
                            # Start of a new entity
                            in_entity = True
                            entity_buffer = '{'
                            depth = 1
                            in_string = False
                            escape_next = False
                        elif in_entity:
                            entity_buffer += char
                            
                            # Handle string literals and escaped characters to avoid misinterpreting brackets in strings
                            if escape_next:
                                escape_next = False
                            elif char == '\\':
                                escape_next = True
                            elif char == '"':
                                in_string = not in_string
                            elif not in_string:
                                if char == '{':
                                    depth += 1
                                elif char == '}':
                                    depth -= 1
                                
                                if depth == 0:
                                    # End of the entity
                                    in_entity = False
                                    # Remove trailing comma if present
                                    if entity_buffer.endswith(','):
                                        entity_buffer = entity_buffer[:-1]
                                    
                                    try:
                                        # Parse the entity
                                        entity = json.loads(entity_buffer)
                                        chunk.append(entity)
                                        
                                        # Process chunk if it reaches chunk_size
                                        if len(chunk) >= chunk_size:
                                            matched_count_chunk = process_chunk(list(chunk), writer, target_p31_set)
                                            processed_count += len(chunk)
                                            matched_count += matched_count_chunk
                                            
                                            # Clear chunk for next batch
                                            chunk.clear()
                                            
                                            # Save progress
                                            elapsed_time = time.time() - start_time
                                            logger.info(f"Processed {processed_count} entities, matched {matched_count}, "
                                                       f"elapsed time: {elapsed_time:.2f}s, "
                                                       f"rate: {processed_count/elapsed_time:.2f} entities/s")
                                            save_progress(output_file, processed_count, matched_count, elapsed_time)
                                            
                                            # Force garbage collection to free up memory
                                            import gc
                                            gc.collect()
                                            
                                            # Check if we've reached the limit
                                            if max_entities and matched_count >= max_entities:
                                                logger.info(f"Reached maximum of {max_entities} matched entities, stopping")
                                                break
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"Failed to parse entity: {e}")
                                        continue
                    
                    # Check if we've reached the limit
                    if max_entities and matched_count >= max_entities:
                        break
                
                # Process any remaining entities
                if chunk:
                    matched_count_chunk = process_chunk(list(chunk), writer, target_p31_set)
                    processed_count += len(chunk)
                    matched_count += matched_count_chunk
                
                # Final progress update
                elapsed_time = time.time() - start_time
                logger.info(f"Finished processing! Processed {processed_count} entities, matched {matched_count}, "
                           f"elapsed time: {elapsed_time:.2f}s, "
                           f"rate: {processed_count/elapsed_time:.2f} entities/s")
                save_progress(output_file, processed_count, matched_count, elapsed_time)
        
        except Exception as e:
            logger.error(f"Error during extraction: {str(e)}")
            raise

def test_extraction():
    """Test the extraction functionality with a small sample."""
    import tempfile
    import io
    
    # Simplified test lookup_tables for both methods
    # Note: Fixed JSON format that works with both parsing methods
    test_data = '''[
{"id":"Q42","type":"item","labels":{"en":{"language":"en","value":"Douglas Adams"}},"aliases":{"en":[{"language":"en","value":"Author of Hitchhiker's Guide"}]},"claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q5"}}}}]}},
{"id":"Q5","type":"item","labels":{"en":{"language":"en","value":"human"}},"claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q12345"}}}}]}}
]'''
    
    logger.info("Creating test files...")
    
    # Use direct CSV processing for verification
    with tempfile.NamedTemporaryFile(suffix='.json.gz', delete=False) as temp_file:
        # Write compressed test lookup_tables
        with gzip.open(temp_file.name, 'wt', encoding='utf-8') as f:
            f.write(test_data)
        
        temp_output1 = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
        temp_output2 = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
        
        try:
            # Test direct CSV writing
            logger.info("Testing direct CSV writing...")
            with open(temp_output1, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['QID', 'Label', 'Aliases'])
                
                # Parse test lookup_tables directly
                with gzip.open(temp_file.name, 'rt', encoding='utf-8') as f:
                    test_entities = json.load(f)
                    for entity in test_entities:
                        if ('claims' in entity and 'P31' in entity['claims'] and
                            any(claim.get('mainsnak', {}).get('datavalue', {}).get('value', {}).get('id') == 'Q5'
                                for claim in entity['claims']['P31'])):
                            
                            writer.writerow([
                                entity['id'],
                                entity['labels']['en']['value'],
                                '|'.join(alias['value'] for alias in entity['aliases'].get('en', []))
                            ])
                            logger.info(f"Direct CSV writer matched: {entity['id']}")
            
            # Test the batch loading extraction
            logger.info("Testing batch loading extraction...")
            extract_with_full_loading(temp_file.name, temp_output2, ["Q5"], max_entities=10)
            
            # Compare results
            with open(temp_output1, 'r', encoding='utf-8') as f1, open(temp_output2, 'r', encoding='utf-8') as f2:
                direct_output = f1.readlines()[1:]  # Skip header
                batch_output = f2.readlines()[1:]   # Skip header
                
                logger.info(f"Direct CSV output: {direct_output}")
                logger.info(f"Batch processor output: {batch_output}")
                
                if set(direct_output) == set(batch_output):
                    logger.info("🎉 Test passed! Both methods produced the same results.")
                else:
                    logger.warning("⚠️ Test failed: Results differ between methods.")
            
        finally:
            # Clean up
            os.unlink(temp_file.name)
            if os.path.exists(temp_output1):
                os.unlink(temp_output1)
            if os.path.exists(temp_output2):
                os.unlink(temp_output2)
    
    logger.info("Test completed!")

def main():
    """Main function to parse arguments and run processing."""
    parser = argparse.ArgumentParser(
        description="Extract entities from Wikidata dump based on P31 values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--input", default="latest-all.json.gz",
                      help="Path to Wikidata JSON dump (compressed with gzip)")
    parser.add_argument("--output", default="lookup_tables/extracted_entities.csv",
                      help="Path to output CSV file")
    parser.add_argument("--qids", nargs='+',
                      help="List of QIDs to extract (entities that are instances of these QIDs)")
    parser.add_argument("--max-entities", type=int, default=None,
                      help="Maximum number of entities to extract (default: no limit)")
    parser.add_argument("--force-streaming", action="store_true",
                      help="Force using the streaming parser (lower memory usage but slower)")
    parser.add_argument("--checkpoint-interval", type=int, default=100000,
                      help="How often to save progress checkpoint (in entities processed)")
    parser.add_argument("--test", action="store_true",
                      help="Run test extraction on a small sample")
    parser.add_argument("--memory-limit", type=float, default=None,
                      help="Memory limit in GB. If the process exceeds this limit, it will try to recover by forcing garbage collection.")
    parser.add_argument("--batch-size", type=int, default=10000,
                      help="Number of entities to process in a batch for the full loading method.")
    
    args = parser.parse_args()
    
    # Run tests if requested
    if args.test:
        test_extraction()
        return
    
    # Ensure required args are provided for normal operation
    if not args.qids:
        parser.error("--qids is required unless --test is specified")
    
    # Log the arguments
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Target QIDs: {args.qids}")
    logger.info(f"Max entities: {args.max_entities}")
    
    # Choose extraction method based on available memory
    available_memory = get_available_memory_gb()
    logger.info(f"Available memory: {available_memory:.2f} GB")
    
    if args.force_streaming or available_memory < 4.0:
        # Use streaming for low memory systems
        logger.info("Using streaming parser due to memory constraints or force flag")
        extract_with_streaming(args.input, args.output, args.qids, args.max_entities, 
                             args.checkpoint_interval, args.memory_limit)
    else:
        # Use partial loading for systems with more memory (faster)
        logger.info("Using partial loading parser (faster but uses more memory)")
        if args.batch_size:
            logger.info(f"Using custom batch size: {args.batch_size}")
        extract_with_full_loading(args.input, args.output, args.qids, args.max_entities)
    
    logger.info(f"Extraction complete! Results saved to {args.output}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)