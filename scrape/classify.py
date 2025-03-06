#!/usr/bin/env python3
"""
AllSides News Classifier - Main launcher for model-specific classifiers

This script is a simple launcher that will:
1. Run update_database.py to ensure the central database is up to date
2. Run model-specific processors to classify articles
"""

import os
import sys
import argparse
import subprocess

# Available models
AVAILABLE_MODELS = ['deepseek']  # Add more as they are implemented

def list_models():
    """List all available models"""
    print("Available models:")
    for model in AVAILABLE_MODELS:
        print(f"- {model}")

def update_database(args=[]):
    """Update the central article database"""
    # Build the command
    cmd = ['python', 'update_database.py'] + args
    
    # Run the command
    try:
        print("Updating central article database...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("Error updating database. Please check the output above.")
            return False
        return True
    except Exception as e:
        print(f"Error updating database: {e}")
        return False

def run_model_processor(model, args):
    """Run the processor for the specified model with the given arguments"""
    # Build the path to the model's processor script
    model_dir = os.path.join('models', model)
    processor_script = os.path.join(model_dir, f"{model}_processor.py")
    
    # Check if the processor script exists
    if not os.path.exists(processor_script):
        print(f"Error: Processor script for model '{model}' not found at {processor_script}")
        return 1
    
    # Build the command
    cmd = ['python', processor_script] + args
    
    # Run the command
    try:
        print(f"Running {model} processor...")
        result = subprocess.run(cmd)
        return result.returncode
    except Exception as e:
        print(f"Error running {model} processor: {e}")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AllSides News Classifier - Multi-model launcher")
    
    # Main model selection
    parser.add_argument('--model', choices=AVAILABLE_MODELS,
                        help='Model to use for classification')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models')
    
    # Database update options
    parser.add_argument('--update-db', action='store_true',
                        help='Update the central database before classification')
    parser.add_argument('--initial-setup', action='store_true',
                        help='Run initial setup for the central database')
    parser.add_argument('--extract-missing', action='store_true',
                        help='Extract missing text from articles in the database')
    
    # Classification options
    parser.add_argument('--classify-all', action='store_true',
                        help='Classify all articles (including already classified ones)')
    
    args, remaining_args = parser.parse_known_args()
    
    if args.list_models:
        list_models()
        sys.exit(0)
    
    # Check if we need to update the database first
    if args.update_db or args.initial_setup or args.extract_missing:
        db_args = []
        if args.initial_setup:
            db_args.append('--initial-setup')
        elif args.extract_missing:
            db_args.append('--extract-missing')
        
        if not update_database(db_args):
            sys.exit(1)
    
    # If no model specified, just update the database and exit
    if not args.model:
        if not (args.update_db or args.initial_setup or args.extract_missing):
            parser.print_help()
            print("\nError: Must specify a model using --model or a database operation")
        sys.exit(0)
    
    # Prepare model-specific arguments
    model_args = []
    if args.classify_all:
        model_args.append('--classify-all')
    
    # Add any remaining arguments
    model_args.extend(remaining_args)
    
    # Run the model processor
    sys.exit(run_model_processor(args.model, model_args)) 