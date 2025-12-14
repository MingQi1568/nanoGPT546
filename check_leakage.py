# Note this is 100% gemini-generated and I kevin han have not looked at it yet
# it's a sanity check for me so don't trust it with ur life or anything

import os
import argparse
import sys

# --- CONFIGURATION ---
train_file = 'data/ese546data/training.txt'
test_file = 'data/ese546data/testing.txt'

def get_train_prompts(filename):
    """Parses training file into a SET for fast lookup."""
    print(f"Reading training data from {filename}...")
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        sys.exit(1)

    prompts = set()
    problem_lines = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            problem_lines.append(line)
            if line.strip().endswith('\\&\\'):
                if problem_lines:
                    prompts.add(problem_lines[0].strip())
                problem_lines = []
    return prompts

def get_test_prompts(filename, limit=None):
    """Parses test file into a LIST to preserve order, optional limit."""
    limit_str = f"first {limit}" if limit else "all"
    print(f"Reading {limit_str} problems from {filename}...")
    
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        sys.exit(1)

    prompts = []
    problem_lines = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            problem_lines.append(line)
            if line.strip().endswith('\\&\\'):
                if problem_lines:
                    prompts.append(problem_lines[0].strip())
                    
                    # Check limit
                    if limit is not None and len(prompts) >= limit:
                        break
                problem_lines = []
    return prompts

def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description='Check for data leakage between train and test sets.')
    
    # Add the specific flag you requested
    parser.add_argument('--1000', action='store_true', help='Check only the first 1000 test problems')
    
    # Add a flexible limit option (e.g., --limit 500)
    parser.add_argument('--limit', type=int, default=None, help='Check a specific number of test problems')

    args = parser.parse_args()

    # Determine the limit
    limit = None
    if getattr(args, '1000'): # Handle the --1000 flag
        limit = 1000
    elif args.limit:          # Handle --limit N
        limit = args.limit

    # 2. Load Training Data (The "Database")
    train_prompts = get_train_prompts(train_file)
    print(f"-> Indexed {len(train_prompts)} unique training prompts.")

    # 3. Load Test Problems
    test_prompts = get_test_prompts(test_file, limit=limit)
    total_test = len(test_prompts)
    print(f"-> Loaded {total_test} test problems.")

    if total_test == 0:
        print("No test problems found.")
        return

    # 4. Check Leakage
    leakage_count = 0
    overlapping_examples = []

    for t_prompt in test_prompts:
        if t_prompt in train_prompts:
            leakage_count += 1
            if len(overlapping_examples) < 3:
                overlapping_examples.append(t_prompt)

    # 5. Results
    print("\n" + "="*50)
    if limit:
        print(f"ANALYSIS OF FIRST {total_test} TEST PROBLEMS")
    else:
        print(f"ANALYSIS OF FULL TEST SET ({total_test} PROBLEMS)")
    print("="*50)
    
    print(f"Leaked (seen in training): {leakage_count}")
    print(f"Unique (new to model):     {total_test - leakage_count}")
    print(f"Leakage Rate:              {(leakage_count / total_test) * 100:.2f}%")
    print("="*50)

    if leakage_count > 0:
        print("\n[!] Examples of leaked problems:")
        for ex in overlapping_examples:
            print(f"   - {ex}")

if __name__ == "__main__":
    main()