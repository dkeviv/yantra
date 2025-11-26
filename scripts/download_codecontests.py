#!/usr/bin/env python3
"""
Download and prepare CodeContests dataset for GraphSAGE training

Dataset: deepmind/code_contests from Hugging Face
Total: 13,328 examples
Target: ~8,000 Python examples for training

Usage:
    python scripts/download_codecontests.py --output ~/.yantra/datasets/codecontests
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import sys

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    print("ERROR: Required packages not installed")
    print("Install with: pip install datasets tqdm")
    sys.exit(1)


def download_codecontests(output_dir: str, limit: Optional[int] = None) -> None:
    """
    Download CodeContests dataset and filter for Python examples
    
    Args:
        output_dir: Directory to save processed dataset
        limit: Maximum number of examples to process (None = all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("📦 Downloading CodeContests dataset from Hugging Face...")
    print("   This may take several minutes on first download...")
    
    try:
        # Load dataset (will cache automatically)
        dataset = load_dataset("deepmind/code_contests", split="train")
        print(f"✓ Loaded {len(dataset)} examples")
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Install datasets: pip install datasets")
        print("3. Try with HF token if private: huggingface-cli login")
        sys.exit(1)
    
    print(f"\n🔍 Filtering for Python examples...")
    
    train_examples = []
    validation_examples = []
    stats = {
        'total': 0,
        'python': 0,
        'has_tests': 0,
        'has_solutions': 0,
        'valid': 0
    }
    
    # Process examples
    for idx, example in enumerate(tqdm(dataset, desc="Processing")):
        if limit and idx >= limit:
            break
            
        stats['total'] += 1
        
        # Extract solutions
        solutions = example.get('solutions', {})
        if not solutions:
            continue
        
        stats['has_solutions'] += 1
        
        # Look for Python solutions (language code 3 = Python)
        python_solutions = []
        if 'language' in solutions and 'solution' in solutions:
            languages = solutions['language']
            solution_codes = solutions['solution']
            
            for lang, code in zip(languages, solution_codes):
                if lang == 3:  # Python
                    python_solutions.append(code)
        
        if not python_solutions:
            continue
            
        stats['python'] += 1
        
        # Check for test cases
        tests = example.get('public_tests', {})
        private_tests = example.get('private_tests', {})
        
        has_tests = (
            (tests and tests.get('input') and tests.get('output')) or
            (private_tests and private_tests.get('input') and private_tests.get('output'))
        )
        
        if not has_tests:
            continue
            
        stats['has_tests'] += 1
        
        # Create training example
        processed_example = {
            'id': f"codecontests_{idx}",
            'problem': example.get('description', ''),
            'solutions': python_solutions,
            'public_tests': {
                'input': tests.get('input', []),
                'output': tests.get('output', [])
            },
            'private_tests': {
                'input': private_tests.get('input', []),
                'output': private_tests.get('output', [])
            },
            'difficulty': example.get('difficulty', 'unknown'),
            'source': example.get('source', 'unknown')
        }
        
        stats['valid'] += 1
        
        # 80/20 train/val split
        if stats['valid'] % 5 == 0:
            validation_examples.append(processed_example)
        else:
            train_examples.append(processed_example)
    
    # Save datasets
    train_file = output_path / "train.jsonl"
    val_file = output_path / "validation.jsonl"
    stats_file = output_path / "stats.json"
    
    print(f"\n💾 Saving datasets...")
    
    with open(train_file, 'w') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    
    with open(val_file, 'w') as f:
        for example in validation_examples:
            f.write(json.dumps(example) + '\n')
    
    # Save statistics
    final_stats = {
        **stats,
        'train_size': len(train_examples),
        'validation_size': len(validation_examples),
        'train_file': str(train_file),
        'validation_file': str(val_file)
    }
    
    with open(stats_file, 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    # Print summary
    print(f"\n✓ Dataset preparation complete!")
    print(f"\n📊 Statistics:")
    print(f"   Total examples processed: {stats['total']}")
    print(f"   Python solutions found: {stats['python']}")
    print(f"   With test cases: {stats['has_tests']}")
    print(f"   Valid examples: {stats['valid']}")
    print(f"   Train examples: {len(train_examples)}")
    print(f"   Validation examples: {len(validation_examples)}")
    print(f"\n📁 Files saved:")
    print(f"   Train: {train_file}")
    print(f"   Validation: {val_file}")
    print(f"   Stats: {stats_file}")
    
    if stats['valid'] < 1000:
        print(f"\n⚠️  Warning: Only {stats['valid']} valid examples found.")
        print(f"   Expected ~8,000 Python examples with tests.")
        print(f"   Dataset may have changed or filters too strict.")


def main():
    parser = argparse.ArgumentParser(description='Download CodeContests dataset')
    parser.add_argument(
        '--output',
        type=str,
        default='~/.yantra/datasets/codecontests',
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of examples to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Expand user path
    output_dir = os.path.expanduser(args.output)
    
    print("=" * 60)
    print("CodeContests Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    if args.limit:
        print(f"Processing limit: {args.limit} examples")
    print()
    
    download_codecontests(output_dir, args.limit)
    
    print("\n" + "=" * 60)
    print("✓ Done! Dataset ready for training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
