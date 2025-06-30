#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup scripts for Wikidata extraction tools.
Installs all required dependencies.
"""

import subprocess
import sys
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install required packages for the Wikidata extraction scripts."""
    required_packages = [
        'ijson',     # For streamed JSON parsing
        'psutil',    # For memory detection
        'pandas',    # For lookup_tables handling
    ]
    
    logger.info("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + required_packages)
        logger.info("All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing packages: {e}")
        return False
    
    return True

def create_data_dir():
    """Create output directory for extracted lookup_tables."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lookup_tables")
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"Created lookup_tables directory: {data_dir}")
    return data_dir

def print_usage_instructions(data_dir):
    """Print usage instructions."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wrapper_script = os.path.join(script_dir, "extract_from_wikidata.py")
    
    print("\n" + "="*80)
    print("WIKIDATA EXTRACTION TOOLS - SETUP COMPLETE")
    print("="*80)
    print("\nTo extract entities from Wikidata dump, run:")
    print(f"\n    python {wrapper_script} --input /path/to/latest-all.json.gz --output {data_dir}/extracted_names.csv")
    print("\nFor more options, see the README.md file or run:")
    print(f"\n    python {wrapper_script} --help")
    print("\nAvailable methods:")
    print("  - Standard: Good performance with moderate memory usage")
    print("  - Streamed: Low memory usage for limited RAM systems")
    print("\nThe wrapper scripts will automatically choose the best method based on your system.")
    print("="*80 + "\n")

def main():
    """Main function to setup the Wikidata extraction environment."""
    logger.info("Setting up Wikidata extraction environment...")
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install required dependencies.")
        return 1
    
    # Create lookup_tables directory
    data_dir = create_data_dir()
    
    # Make scripts executable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for script_name in ["extract_from_wikidata.py", "extract_targeted_entities.py", 
                       "extract_targeted_entities_streamed.py"]:
        script_path = os.path.join(script_dir, script_name)
        try:
            os.chmod(script_path, 0o755)  # -rwxr-xr-x
            logger.info(f"Made {script_name} executable")
        except OSError as e:
            logger.warning(f"Could not make {script_name} executable: {e}")
    
    # Print usage instructions
    print_usage_instructions(data_dir)
    
    logger.info("Setup complete! You can now use the Wikidata extraction tools.")
    return 0

if __name__ == "__main__":
    sys.exit(main())