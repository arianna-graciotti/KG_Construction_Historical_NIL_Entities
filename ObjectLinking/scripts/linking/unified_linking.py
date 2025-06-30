#!/usr/bin/env python3
"""
Unified Entity Linking Script

This scripts serves as the main entry point for the unified entity linking system.
It provides a command-line interface to run entity linking for any supported entity type.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional

from entity_linker import EntityLinker
from entity_configs import get_entity_config

# Check if we're using the updated version with multi-entity support
print("Checking EntityLinker implementation:")
print("Has split_answer_into_entities method:", hasattr(EntityLinker, "split_answer_into_entities"))


def parse_arguments():
    """Parse command-line arguments for the unified linking scripts"""
    parser = argparse.ArgumentParser(
        description='Unified Entity Linking Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Link country entities in RAG and ZS folders
  python unified_linking.py --entity-type country --folders /path/to/RAG/CoC /path/to/ZS/CoC
  
  # Link family names with custom output directory and more worker threads
  python unified_linking.py --entity-type family_name --output-dir /path/to/output --max-workers 4 --folders /path/to/RAG/FamilyName
        """
    )
    
    parser.add_argument(
        '--entity-type',
        type=str,
        required=True,
        choices=['country', 'family_name', 'given_name', 'occupation', 'sex_gender'],
        help='The type of entity to link'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for linked files and reports'
    )
    
    parser.add_argument(
        '--folders',
        type=str,
        nargs='+',
        required=True,
        help='Folders to process (space separated)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=2,
        help='Maximum number of worker threads'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for processing'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode with limited files'
    )
    
    parser.add_argument(
        '--fuzzy-threshold',
        type=float,
        default=0.85,
        help='Threshold for fuzzy matching (0.0-1.0)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level'
    )
    
    return parser.parse_args()


def setup_basic_logging(log_level_name: str):
    """Set up basic logging before the EntityLinker configures its own logging"""
    log_level = getattr(logging, log_level_name)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def main():
    """Main entry point for the unified entity linking tool"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up initial logging
    setup_basic_logging(args.log_level)
    
    try:
        # Get the appropriate config
        config = get_entity_config(args.entity_type)
        
        # Log startup information
        logging.info(f"Starting unified entity linking for {config.entity_name}")
        logging.info(f"Processing folders: {', '.join(args.folders)}")
        
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
        
        logging.info(f"Entity linking for {config.entity_name} completed successfully")
        
    except Exception as e:
        logging.error(f"Error in unified linking: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()