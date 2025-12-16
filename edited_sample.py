"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import re
import math
from model import GPTConfig, GPT

def calculate(string, roundign_precision=4, printint_precision=4, logging=False): 
    lines = string.splitlines()
    output = []

    # Matches: 5, -5, 5.0, -5.0001
    num_pattern = r'(-?\d+(?:\.\d+)?)'

    for i in range(len(lines)):
        line = lines[i]
        output.append(line)
        result = None
        
        is_valid_location = (i + 1 == len(lines)) or (not lines[i + 1].strip().startswith("="))
        
        if not is_valid_location:
            continue

        op_name = ""
        
        if match := re.search(rf'\\add\s+{num_pattern}\s+{num_pattern}\s', line):
            a, b = float(match.group(1)), float(match.group(2))
            result = a + b
            op_name = f"ADD {a} {b}"

        # the elifs save time in case one match is found
        elif match := re.search(rf'\\sub\s+{num_pattern}\s+{num_pattern}\s', line):
            a, b = float(match.group(1)), float(match.group(2))
            result = a - b
            op_name = f"SUB {a} {b}"

        elif match := re.search(rf'\\mul\s+{num_pattern}\s+{num_pattern}\s', line):
            a, b = float(match.group(1)), float(match.group(2))
            result = a * b
            op_name = f"MUL {a} {b}"

        elif match := re.search(rf'\\div\s+{num_pattern}\s+{num_pattern}\s', line):
            a, b = float(match.group(1)), float(match.group(2))
            try:
                result = a / b
            except ZeroDivisionError:
                result = float('nan')
            op_name = f"DIV {a} {b}"

        elif match := re.search(rf'\\abs\s+{num_pattern}\s', line):
            a = float(match.group(1))
            result = abs(a)
            op_name = f"ABS {a}"

        elif match := re.search(rf'\\sin\s+{num_pattern}\s', line):
            a = float(match.group(1))
            result = math.sin(a)
            op_name = f"SIN {a}"

        elif match := re.search(rf'\\cos\s+{num_pattern}\s', line):
            a = float(match.group(1))
            result = math.cos(a)
            op_name = f"COS {a}"

        elif match := re.search(rf'\\log\s+{num_pattern}\s', line):
            a = float(match.group(1))
            try:
                result = math.log(a)
            except ValueError:
                result = float('nan')
            op_name = f"LOG {a}"

        elif match := re.search(rf'\\exp\s+{num_pattern}\s', line):
            a = float(match.group(1))
            try:
                result = math.exp(a)
            except OverflowError:
                result = float('inf')
            op_name = f"EXP {a}"

        if result is not None:
            if result == 0: result = 0.0 # Standardize negative zero
            value = float(f"{result:.{rounding_precision}}f") # round as input precision
            formatted_res = f"={value:.{printing_precision}f}" # but print out as output precision
            
            if logging: print(f"[{op_name}] -> {formatted_res}") 

            output.append(formatted_res + "\n")

    return "\n".join(output)

def main():
    # -----------------------------------------------------------------------------
    init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    out_dir = 'out-546' # ignored if init_from is not 'resume'
    start = """
    abs(-16)
    $|\\abs -16 
    =16
    16
    16\\&\\
    abs(-8)
    $|\\abs -8 
    =8
    8
    8\\&\\
    abs(abs(-12))
    abs($)|\\abs -12 
    =12
    abs(12)
    $|\\abs 12 
    =12
    12
    12\\&\\
    19+34""" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    num_samples = 1 # number of samples to draw
    max_new_tokens = 20 # number of tokens generated in each sample
    temperature = 0.1 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1337
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    compile = False # use PyTorch 2.0 to compile the model to be faster
    exec(open('configurator.py').read()) # overrides from command line or config file
    # -----------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # model
    if init_from == 'resume':
        # init from a model saved in a specific directory
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
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        print(f"Available characters: {repr(''.join(sorted(stoi.keys())))}")
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation HELLO
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                
                y = model.generate(x, 2, temperature=temperature, top_k=top_k)
                decoded = decode(y[0].tolist())
                while "\\&\\" not in decoded[-5:] and len(decoded) < 1000:
                    y = torch.tensor(encode(calculate(decoded)))[None, ...].to(device)
                    y = model.generate(y, 2, temperature=temperature, top_k=top_k)
                    decoded = decode(y[0].tolist())

                            
                print(decoded)
                print('---------------')

if __name__ == '__main__':
    main()
