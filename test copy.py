
import os
import pickle
import re
import math
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from edited_sample import calculate

# -----------------------------------------------------------------------------
# Configuration
init_from = 'resume' # 'resume' (from an out_dir) or a gpt2 variant
out_dir = '/content/drive/MyDrive/nanoGPT_Output' # ignored if init_from is not 'resume'
test_file = 'data/ese546data/testing.txt'
max_new_tokens = 1000 # maximum tokens to generate per sample
temperature = 0.1 # sampling temperature
top_k = 200 # top-k sampling
device = 'cuda' # 'cpu', 'cuda', 'cuda:0', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False # use PyTorch 2.0 to compile the model
precision = 4  # decimal places for numerical answers, of calculator output
logging = False  # whether to log calculation steps
# -----------------------------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model
print("Loading model...")
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    print("Compiling model...")
    model = torch.compile(model)

# Load vocabulary
print("Loading vocabulary...")
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# Parse testing.txt to extract problems and answers
print(f"Reading test file: {test_file}...")
with open(test_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Split into problems (each problem ends with \&\)
problems = []
problem_lines = []
for line in content.split('\n'):
    problem_lines.append(line)
    if line.endswith('\\&\\'):
        # This is the end of a problem
        if problem_lines:
            # First line is the initial expression (prompt)
            prompt = problem_lines[0]
            # The line with \&\ contains the answer (extract number before \&\)
            answer_line = problem_lines[-1]
            answer_str = answer_line.replace('\\&\\', '').strip()
            problems.append({
                'prompt': prompt,
                'answer': answer_str,
                'full_text': '\n'.join(problem_lines)
            })
        problem_lines = []

print(f"Found {len(problems)} problems to test")

# Extract numeric answer from a string (handles nan, inf, and regular numbers)
def extract_answer(text):
    """Extract the final numeric answer from model output."""
    # Look for lines ending with \&\ first (this is the standard format)
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line.endswith('\\&\\'):
            # Extract the number before \&\
            answer = line.replace('\\&\\', '').strip()
            return answer
    
    # If no \&\ found, look for the last number in the text
    # Try to find patterns like "=number" or just "number" at the end
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        # Remove = sign if present
        if line.startswith('='):
            line = line[1:].strip()
        # Check if it's a valid number (including nan, inf, negative numbers)
        if re.match(r'^-?\d+\.?\d*$|^nan$|^inf$|^-inf$', line, re.IGNORECASE):
            return line
    
    # If no clear answer found, return the last non-empty line
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return ""

# Normalize answers for comparison (handle floating point precision)
def normalize_answer(ans_str):
    """Normalize answer string for comparison."""
    ans_str = ans_str.strip().lower()
    # Handle nan, inf
    if ans_str == 'nan' or ans_str == '-nan':
        return 'nan'
    if ans_str == 'inf' or ans_str == '-inf':
        return ans_str.lower()
    # Try to parse as float and round to 4 decimal places (matching training precision)
    try:
        val = float(ans_str)
        return f"{val:.{precision}f}"
    except ValueError:
        return ans_str

# -----------------------------------------------------------------------------
# Evaluation Logic
# -----------------------------------------------------------------------------

def run_evaluation(use_calculator_flag):
    """Runs the generation loop for all problems with specific calc setting."""
    results = {}
    print(f"\n--- Running evaluation with use_calculator={use_calculator_flag} ---")
    
    for i, problem in enumerate(problems):
        prompt = "\n" + problem['prompt'] + '\n'
        
        x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
        
        try:
            with torch.no_grad():
                with ctx:
                    # Generate first 2 tokens
                    y = model.generate(x, 2, temperature=temperature, top_k=top_k)
                    decoded = decode(y[0].tolist())
                    
                    # Generation Loop
                    while "\\&\\" not in decoded[-5:] and len(decoded) < max_new_tokens:
                        if use_calculator_flag:
                            # Use the imported calculate function
                            calc_out = calculate(decoded, precision=precision)
                        else:
                            # Pass through raw text
                            calc_out = decoded
                            
                        y = torch.tensor(encode(calc_out))[None, ...].to(device)
                        y = model.generate(y, 2, temperature=temperature, top_k=top_k)
                        decoded = decode(y[0].tolist())
            
            pred = extract_answer(decoded)
            is_correct = normalize_answer(pred) == normalize_answer(problem['answer'])
            
            results[i] = {
                'pred': pred,
                'correct': is_correct,
                'decoded': decoded
            }
            
        except Exception as e:
            results[i] = {'pred': 'ERROR', 'correct': False, 'decoded': str(e)}
        
        if (i+1) % 100 == 0:
            print(f"Progress: {i+1}/{len(problems)}")
            
    return results

# -----------------------------------------------------------------------------
# Run Experiment
# -----------------------------------------------------------------------------

# 1. Run both modes
no_calc_results = run_evaluation(False)
yes_calc_results = run_evaluation(True)

# 2. Analyze Symmetric Difference
# Cases where Calc HELPED (Model Wrong -> Calc Right)
calc_helped = []
# Cases where Calc HURT (Model Right -> Calc Wrong)
calc_hurt = []

for i in range(len(problems)):
    nc = no_calc_results[i]
    yc = yes_calc_results[i]
    
    if not nc['correct'] and yc['correct']:
        calc_helped.append(i)
    elif nc['correct'] and not yc['correct']:
        calc_hurt.append(i)

# 3. Print Results
print("\n" + "="*80)
print("SYMMETRIC DIFFERENCE ANALYSIS")
print("="*80)
print(f"Total Samples: {len(problems)}")
print(f"Cases where Calculator HELPED (Fixed the error): {len(calc_helped)}")
print(f"Cases where Calculator HURT   (Broke the answer): {len(calc_hurt)}")
print("="*80)

print("\n--- Examples where Calculator HURT (Model was right, Calc made it wrong) ---")
for idx in calc_hurt[:10]:
    p = problems[idx]
    nc = no_calc_results[idx]
    yc = yes_calc_results[idx]
    print(f"\nProblem {idx+1}: {p['prompt']}")
    print(f"Expected: {p['answer']}")
    print(f"WITHOUT Calc (Correct): {nc['pred']}")
    print(f"WITH Calc    (Wrong):   {yc['pred']}")
    print(f"Calc Trace Tail: {yc['decoded'][-200:].replace(chr(10), ' ')}") 

print("\n--- Examples where Calculator HELPED (Model was wrong, Calc fixed it) ---")
for idx in calc_helped[:10]:
    p = problems[idx]
    nc = no_calc_results[idx]
    yc = yes_calc_results[idx]
    print(f"\nProblem {idx+1}: {p['prompt']}")
    print(f"Expected: {p['answer']}")
    print(f"WITHOUT Calc (Wrong):   {nc['pred']}")
    print(f"WITH Calc    (Correct): {yc['pred']}")