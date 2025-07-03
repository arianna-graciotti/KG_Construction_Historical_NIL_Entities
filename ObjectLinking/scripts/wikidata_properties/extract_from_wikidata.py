#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikidata Entity Extraction Wrapper

This scripts provides an easy interface to extract entities from a Wikidata dump
by choosing the most appropriate method based on your system's memory constraints.

It targets entities that are instances of (P31) specific QIDs:
- Q101352    # Family name
- Q18972245  # Given name
- Q18972207  # Name used for males
- Q11455398  # Name used for females
- Q98775491  # Sex-neutral name
- Q829026    # Unisex name
- Q27951364  # Human name

It outputs a CSV file with QID, label, and aliases.
"""

import os
import sys
import argparse
import logging
import psutil
import subprocess
from pathlib import Path

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

def main():
    """Main function to parse arguments and run processing."""
    parser = argparse.ArgumentParser(
        description="Extract entities from Wikidata dump based on P31 values")
    parser.add_argument("--input", default="latest-all.json.gz",
                      help="Path to Wikidata JSON dump (compressed with gzip)")
    parser.add_argument("--output", default="wikidata_extracted_names.csv",
                      help="Path to output CSV file")
    parser.add_argument("--max-entities", type=int, default=None,
                      help="Maximum number of entities to extract (default: no limit)")
    parser.add_argument("--force-method", choices=["standard", "streamed"],
                      help="Force using a specific method instead of auto-selection")
    parser.add_argument("--reset", action="store_true",
                      help="Ignore checkpoint and start from beginning (standard method only)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Get file size
    input_size_gb = os.path.getsize(args.input) / (1024**3)
    logger.info(f"Input file size: {input_size_gb:.2f} GB")
    
    # Get available memory
    available_memory_gb = get_available_memory_gb()
    logger.info(f"Available system memory: {available_memory_gb:.2f} GB")
    
    # Determine which method to use
    if args.force_method:
        method = args.force_method
        logger.info(f"Using forced method: {method}")
    else:
        # Auto-select based on memory
        if available_memory_gb > input_size_gb * 0.25:  # If we have at least 25% of the file size in RAM
            method = "standard"
            logger.info("Sufficient memory available, using standard method")
        else:
            method = "streamed"
            logger.info("Limited memory available, using streamed method")
    
    # Get paths to scripts
    script_dir = get_script_directory()
    if method == "standard":
        script_path = os.path.join(script_dir, "extract_targeted_entities.py")
    else:
        script_path = os.path.join(script_dir, "extract_targeted_entities_streamed.py")
    
    # Build command
    cmd = [sys.executable, script_path, 
           "--input", args.input,
           "--output", args.output]
    
    # Add max entities if specified
    if args.max_entities:
        cmd.extend(["--max-entities", str(args.max_entities)])
    
    # Add reset if specified (standard method only)
    if args.reset and method == "standard":
        cmd.append("--reset")
    
    # Run extraction
    logger.info(f"Running extraction with {method} method...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Execute the selected scripts
    exit_code = subprocess.call(cmd)
    
    if exit_code == 0:
        logger.info(f"Extraction complete. Results saved to: {args.output}")
    else:
        logger.error(f"Extraction failed with exit code: {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)