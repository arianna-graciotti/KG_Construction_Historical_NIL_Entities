#!/usr/bin/env python3
"""
Orchestration scripts to run unified entity linking for all entity types.
Includes robust error handling, checkpointing, and progress tracking.
Uses the new unified_linking.py scripts instead of individual scripts.
"""

import os
import sys
import time
import json
import logging
import subprocess
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configure paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "run_all_unified_linking_checkpoint.json")

# Entity types to process with their respective folders
ENTITY_TASKS = [
    {
        "entity_type": "country", 
        "folders": [
            "/home/arianna/NIL_Grounding_Eval/lookup_tables/archive_080525/RAG/CoC",
            "/home/arianna/NIL_Grounding_Eval/lookup_tables/archive_080525/ZS/CoC"
        ]
    },
    {
        "entity_type": "occupation", 
        "folders": [
            "/home/arianna/NIL_Grounding_Eval/lookup_tables/archive_080525/RAG/occupation",
            "/home/arianna/NIL_Grounding_Eval/lookup_tables/archive_080525/ZS/occupation"
        ]
    },
    {
        "entity_type": "sex_gender", 
        "folders": [
            "/home/arianna/NIL_Grounding_Eval/lookup_tables/archive_080525/RAG/sexGender",
            "/home/arianna/NIL_Grounding_Eval/lookup_tables/archive_080525/ZS/sexGender"
        ]
    },
    {
        "entity_type": "family_name", 
        "folders": [
            "/home/arianna/NIL_Grounding_Eval/lookup_tables/archive_080525/RAG/FamilyName",
            "/home/arianna/NIL_Grounding_Eval/lookup_tables/archive_080525/ZS/FamilyName"
        ]
    },
    {
        "entity_type": "given_name", 
        "folders": [
            "/home/arianna/NIL_Grounding_Eval/lookup_tables/archive_080525/RAG/GivenName",
            "/home/arianna/NIL_Grounding_Eval/lookup_tables/archive_080525/ZS/GivenName"
        ]
    }
]

# Path to the unified linking scripts
UNIFIED_SCRIPT = os.path.join(SCRIPT_DIR, "unified_linking.py")

# Configure logging
LOG_FILE = os.path.join(LOG_DIR, "run_all_unified_linking.log")
ERROR_LOG_FILE = os.path.join(LOG_DIR, "run_all_unified_linking_errors.log")
LOG_LEVEL = logging.INFO

# Configure execution options
SCRIPT_TIMEOUT = 24 * 60 * 60  # 24 hours in seconds
MAX_RETRIES = 2  # Number of times to retry a failed task
RETRY_DELAY = 300  # Seconds to wait before retrying (5 minutes)
# Additional parameters for the unified scripts
MAX_WORKERS = 2  # Number of worker threads
BATCH_SIZE = 10  # Batch size for processing

# Setup logging
def setup_logging():
    """Set up logging with both main log and separate error log"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Configure main logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatters
    main_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    error_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s\n%(pathname)s:%(lineno)d\n")

    # Set up main log file
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(main_formatter)
    file_handler.setLevel(LOG_LEVEL)
    root_logger.addHandler(file_handler)

    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(main_formatter)
    console_handler.setLevel(LOG_LEVEL)
    root_logger.addHandler(console_handler)

    # Set up error log file (only errors and criticals)
    error_handler = logging.FileHandler(ERROR_LOG_FILE)
    error_handler.setFormatter(error_formatter)
    error_handler.setLevel(logging.ERROR)
    root_logger.addHandler(error_handler)

    logging.info(f"Logging initialized. Main log: {LOG_FILE}, Error log: {ERROR_LOG_FILE}")

# Ensure all necessary directories exist
def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        OUTPUT_DIR,
        LOG_DIR
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")

# Save checkpoint
def save_checkpoint(progress: Dict[str, Any]):
    """Save current execution state for resuming later"""
    try:
        # Add timestamp to track when checkpoint was saved
        progress["last_updated"] = datetime.now().isoformat()
        
        # Save to temp file first to avoid corruption if interrupted
        temp_file = f"{CHECKPOINT_FILE}.tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        
        # Replace the checkpoint file atomically
        if os.path.exists(CHECKPOINT_FILE):
            os.replace(temp_file, CHECKPOINT_FILE)
        else:
            os.rename(temp_file, CHECKPOINT_FILE)
            
        logging.info(f"Checkpoint saved: {progress['completed']}/{len(ENTITY_TASKS)} tasks completed")
    except Exception as e:
        logging.error(f"Error saving checkpoint: {str(e)}")
        logging.error(traceback.format_exc())

# Load checkpoint
def load_checkpoint() -> Dict[str, Any]:
    """Load checkpoint to resume from previous execution"""
    try:
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                logging.info(f"Checkpoint loaded: {checkpoint['completed']}/{len(ENTITY_TASKS)} tasks already completed")
                
                # Validate checkpoint lookup_tables
                if "completed" not in checkpoint or "task_statuses" not in checkpoint:
                    logging.warning("Invalid checkpoint lookup_tables, starting from beginning")
                    return initialize_progress()
                
                return checkpoint
        else:
            logging.info("No checkpoint found, starting from beginning")
    except Exception as e:
        logging.error(f"Error loading checkpoint: {str(e)}")
        logging.error(traceback.format_exc())
    
    # If anything goes wrong, start from scratch
    return initialize_progress()

# Initialize progress tracking
def initialize_progress() -> Dict[str, Any]:
    """Create initial progress tracking structure"""
    task_statuses = {}
    for idx, task in enumerate(ENTITY_TASKS):
        entity_type = task["entity_type"]
        task_statuses[entity_type] = {
            "status": "pending",
            "attempts": 0,
            "last_attempt": None,
            "execution_time": 0
        }
    
    return {
        "started": datetime.now().isoformat(),
        "completed": 0,
        "total": len(ENTITY_TASKS),
        "task_statuses": task_statuses,
        "last_updated": datetime.now().isoformat()
    }

# Run a single task with timeout and logging
def run_task(task: Dict[str, Any]) -> bool:
    """Run entity linking for a single entity type"""
    entity_type = task["entity_type"]
    folders = task["folders"]
    
    logging.info("=" * 80)
    logging.info(f"Starting entity linking for {entity_type}")
    logging.info("=" * 80)
    
    start_time = time.time()
    success = False
    
    try:
        # Build the command to run the unified scripts
        command = [
            sys.executable, 
            UNIFIED_SCRIPT,
            "--entity-type", entity_type,
            "--max-workers", str(MAX_WORKERS),
            "--batch-size", str(BATCH_SIZE),
            "--folders"
        ] + folders
        
        # Run the scripts with timeout
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Setup variables for timeout tracking
        timed_out = False
        elapsed = 0
        
        # Monitor the process while it runs
        while process.poll() is None:
            # Check if we've exceeded timeout
            elapsed = time.time() - start_time
            if elapsed > SCRIPT_TIMEOUT:
                logging.warning(f"Task {entity_type} timed out after {elapsed:.1f} seconds")
                timed_out = True
                process.terminate()
                # Give some time for graceful termination
                time.sleep(5)
                if process.poll() is None:
                    # Force kill if still running
                    process.kill()
                break
            
            # Sleep a bit to reduce CPU usage
            time.sleep(5)
            
            # Log execution progress every 10 minutes
            if elapsed > 0 and elapsed % 600 < 5:
                hours, remainder = divmod(elapsed, 3600)
                minutes, seconds = divmod(remainder, 60)
                logging.info(f"Task {entity_type} has been running for {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Get output and return code
        stdout, stderr = process.communicate()
        return_code = process.returncode
        
        # Handle the result
        if timed_out:
            logging.error(f"Task {entity_type} was terminated due to timeout after {elapsed:.1f} seconds")
            success = False
        elif return_code == 0:
            logging.info(f"Task {entity_type} completed successfully in {elapsed:.1f} seconds")
            success = True
        else:
            logging.error(f"Task {entity_type} failed with return code {return_code}")
            logging.error(f"STDERR: {stderr}")
            success = False
        
        # Always log the output for debugging
        if stdout:
            with open(os.path.join(LOG_DIR, f"{entity_type}_stdout.log"), 'w') as f:
                f.write(stdout)
        if stderr:
            with open(os.path.join(LOG_DIR, f"{entity_type}_stderr.log"), 'w') as f:
                f.write(stderr)
        
    except Exception as e:
        elapsed = time.time() - start_time
        logging.error(f"Error running task {entity_type}: {str(e)}")
        logging.error(traceback.format_exc())
        success = False
    
    execution_time = time.time() - start_time
    logging.info(f"Execution of {entity_type} {'succeeded' if success else 'failed'} after {execution_time:.1f} seconds")
    
    return success

# Main execution function
def main():
    """Main orchestration function to run all entity linking tasks"""
    start_time = time.time()
    
    # Set up logging and directories
    setup_logging()
    ensure_directories()
    
    logging.info("=" * 80)
    logging.info("Starting execution of all entity linking tasks using unified approach")
    logging.info("=" * 80)
    
    # Load or initialize progress
    progress = load_checkpoint()
    
    # Check which tasks need to be executed
    tasks_to_run = []
    for idx, task in enumerate(ENTITY_TASKS):
        entity_type = task["entity_type"]
        
        # Check if this task is already completed
        if entity_type in progress["task_statuses"]:
            status = progress["task_statuses"][entity_type]["status"]
            if status == "completed":
                logging.info(f"Skipping {entity_type} (already completed)")
                continue
            elif status == "failed":
                if progress["task_statuses"][entity_type]["attempts"] >= MAX_RETRIES:
                    logging.warning(f"Task {entity_type} has failed {MAX_RETRIES} times, skipping")
                    continue
        
        # Add to the list to run
        tasks_to_run.append((idx, task))
    
    # Exit if no tasks to run
    if not tasks_to_run:
        logging.info("All tasks have been completed or exceeded retry attempts")
        return
    
    # Execute each task
    for idx, task in tasks_to_run:
        entity_type = task["entity_type"]
        
        # Update task status
        if entity_type not in progress["task_statuses"]:
            progress["task_statuses"][entity_type] = {
                "status": "running",
                "attempts": 1,
                "last_attempt": datetime.now().isoformat(),
                "execution_time": 0
            }
        else:
            progress["task_statuses"][entity_type]["status"] = "running"
            progress["task_statuses"][entity_type]["attempts"] += 1
            progress["task_statuses"][entity_type]["last_attempt"] = datetime.now().isoformat()
        
        # Save checkpoint before running
        save_checkpoint(progress)
        
        # Run the task
        task_start_time = time.time()
        success = run_task(task)
        execution_time = time.time() - task_start_time
        
        # Update task status after execution
        if success:
            progress["task_statuses"][entity_type]["status"] = "completed"
            progress["completed"] += 1
        else:
            progress["task_statuses"][entity_type]["status"] = "failed"
            
            # If we should retry, add delay
            if progress["task_statuses"][entity_type]["attempts"] < MAX_RETRIES:
                retry_delay = RETRY_DELAY * progress["task_statuses"][entity_type]["attempts"]
                logging.info(f"Will retry {entity_type} after {retry_delay} seconds (attempt {progress['task_statuses'][entity_type]['attempts']} of {MAX_RETRIES})")
                time.sleep(retry_delay)
                # Reduce the counter since we'll retry right away
                idx -= 1
        
        progress["task_statuses"][entity_type]["execution_time"] = execution_time
        
        # Save checkpoint after running
        save_checkpoint(progress)
    
    # Calculate total runtime
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Log completion
    logging.info("=" * 80)
    logging.info(f"All entity linking tasks processed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Count completed tasks
    completed_count = sum(1 for status in progress["task_statuses"].values() if status["status"] == "completed")
    failed_count = sum(1 for status in progress["task_statuses"].values() if status["status"] == "failed")
    
    logging.info(f"Tasks completed: {completed_count}/{len(ENTITY_TASKS)}")
    logging.info(f"Tasks failed: {failed_count}/{len(ENTITY_TASKS)}")
    logging.info("=" * 80)

# Run the scripts
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)