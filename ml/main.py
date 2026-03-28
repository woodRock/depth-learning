#!/usr/bin/env python3
"""
Universal CLI for Depth Learning ML Library.
Provides a single entry point for training, serving, and evaluation.
"""

import sys
import os
import argparse
import uvicorn
from typing import List

# Ensure the 'ml' directory is in the path so we can import modules
# This is a temporary measure until we structure as a proper package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logging import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger("ml_cli")

def cmd_train(args):
    """Route to training logic."""
    from cli.train import main as train_main
    # We need to simulate the sys.argv for the sub-script
    # or refactor the sub-script to take args.
    # For now, we'll call its main if it exists, otherwise we'll refactor.
    logger.info(f"Starting training for model: {args.command}")
    # The train script already has its own parser, so we'll pass the remaining args
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    train_main()

def cmd_serve(args):
    """Route to serving logic."""
    from cli.serve import app
    logger.info(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

def cmd_evaluate(args):
    """Route to evaluation logic."""
    # We'll implement this by calling the evaluate logic in serve.py or a new evaluator
    from cli.serve import evaluate
    import asyncio
    
    async def run_eval():
        result = await evaluate(
            architecture=args.arch,
            dataset=args.dataset,
            model_type=args.mode,
            test_dataset=args.test_dataset
        )
        import json
        print(json.dumps(result, indent=2))

    logger.info(f"Evaluating {args.arch} on {args.dataset}")
    asyncio.run(run_eval())

def cmd_experiment(args):
    """Route to experiment logic."""
    from cli.experiment_runner import run_experiments_from_yaml
    logger.info(f"Running experiments from: {args.config}")
    run_experiments_from_yaml(args.config)

def main():
    parser = argparse.ArgumentParser(
        description="Depth Learning Universal CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("command", help="Model type (jepa, lewm, lewm_plus, etc.)")
    # Remaining args will be passed to train.py
    
    # Serve subcommand
    serve_parser = subparsers.add_parser("serve", help="Start the prediction server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--arch", required=True, help="Architecture (JEPA, LeWM, etc.)")
    eval_parser.add_argument("--dataset", required=True, help="Dataset trained on")
    eval_parser.add_argument("--mode", default="Multi-modal", help="Mode (Multi-modal or Acoustic-only)")
    eval_parser.add_argument("--test-dataset", default="same", help="Dataset to test on")

    # Experiment subcommand
    experiment_parser = subparsers.add_parser("experiment", help="Run experiments from YAML")
    experiment_parser.add_argument("--config", required=True, help="Path to YAML config file")

    # If no arguments, print help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # For 'train', we want to stop parsing here and let train.py handle its own args
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # We handle this specially to pass all subsequent args to the train script
        args, unknown = parser.parse_known_args()
        cmd_train(args)
    else:
        args = parser.parse_args()
        if args.action == "serve":
            cmd_serve(args)
        elif args.action == "evaluate":
            cmd_evaluate(args)
        elif args.action == "experiment":
            cmd_experiment(args)
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
