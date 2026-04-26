import argparse
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from accuwm.mc_synthid import mc_synthid_sample_generator
import unbiased_watermark as uwm
from unbiased_watermark.synthid import SynthID_Reweight_fast
from experiments.tasks import get_oeg_human_tokens, get_wiki_human_tokens, get_eli5_human_tokens
from .worker import MaxLengthLogitsProcessor, StopWordsLogitsProcessor
from tqdm import tqdm
import numpy as np
import pickle
import os

def to_display_model_name(s: str) -> str:
    if "/" in s:
        s = s.split("/")[-1]
    s = s.replace("-chat-hf", "")
    return s

def main():
    parser = argparse.ArgumentParser(description='Run SynthID watermarking experiment')
    parser.add_argument('--private_key', type=int, default=0, required=True, help='Private key for watermarking')
    parser.add_argument('--mc_private_key', type=int, default=1, required=True, help='Private key for watermarking')
    parser.add_argument('--model', type=str, default='facebook/opt-1.3b', required=True, help='Model name or path')
    parser.add_argument('--task', type=str, choices=['wiki', 'oeg', 'eli5'], required=True, help='Task to perform')
    parser.add_argument('--ds_cut_len', type=int, default=None, help='Number of examples to use from dataset')
    parser.add_argument('--access_token', type=str, default='YOUR_ACCESS_TOKEN', help='Access token for Hugging Face')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling table')
    parser.add_argument('--context_size', type=int, default=4, help='Context size for watermarking')
    parser.add_argument('--print_result', type=bool, default=False, help='Print result')
    args = parser.parse_args()
    print(args)
    # check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model and tokenizer
    transformers.utils.logging.disable_progress_bar()
    print('='*50)
    print(f"Loading tokenizer {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.access_token)
    
    # Get dataset based on task
    print('='*50)
    print(f"Loading dataset for human data...")
    if args.task == 'wiki':
        ds = get_wiki_human_tokens(tokenizer, length=400, access_token=args.access_token, ds_cut_len=args.ds_cut_len)
    elif args.task == 'oeg':
        ds = get_oeg_human_tokens(tokenizer, length=400, ds_cut_len=args.ds_cut_len)
    elif args.task == 'eli5':
        ds = get_eli5_human_tokens(tokenizer, length=400, access_token=args.access_token, ds_cut_len=args.ds_cut_len)
    else:
        raise ValueError(f"Task {args.task} not supported")
    
    # Initialize watermark components
    reweight = SynthID_Reweight_fast(
        sampling_table_size=2**16,
        sampling_table_seed=args.seed,
        device=device,
        ngram_len=args.context_size,
        private_key=args.private_key
    )
    mc_reweight = SynthID_Reweight_fast(
        sampling_table_size=2**16,
        sampling_table_seed=args.seed + 1,
        device=device,
        ngram_len=args.context_size,
        private_key=args.mc_private_key
    )
    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=args.context_size)
    cch = uwm.lm.ContextCodeHistory(batch_shape=(1,)) # only support batch size 1 for now
    
    
    results = []
    
    # Process each example
    print('='*50)
    for example in tqdm(ds, desc="Processing examples"):
        output_ids = torch.tensor(example['tokens']).unsqueeze(0).to(device)
        output_text = example['text']
        # Store results
        result = {
            'output_text': output_text,
            'output_ids': output_ids.tolist(),
        }

        # Compute g_values
        g_values = reweight.compute_g_values(output_ids)
        mc_g_values = mc_reweight.compute_g_values(output_ids)
        assert g_values.shape[1] == output_ids.shape[1]
        assert mc_g_values.shape[1] == output_ids.shape[1]
        result['g_values'] = g_values.cpu().numpy().squeeze(0)
        result['g_values_mean'] = g_values.cpu().numpy().squeeze(0).mean()
        result['mc_g_values'] = mc_g_values.cpu().numpy().squeeze(0)
        result['mc_g_values_mean'] = mc_g_values.cpu().numpy().squeeze(0).mean()

        # Compute acceptance rate
        cch_r = uwm.lm.ContextCodeHistory(batch_shape=(1,))
        r_values = uwm.lm.get_r_values(cc_extractor, cch_r, args.mc_private_key, output_ids)
        assert r_values.shape[1] == output_ids.shape[1], f"get r_values shape: {r_values.shape}, output_ids shape: {output_ids.shape}"
        result['r_values'] = r_values.squeeze(0)

        if args.print_result:
            print(result)
        results.append(result)
    
    # Save results
    save_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data_root",
        "synthid_mc_human_test",
        f"human_size{args.ds_cut_len}_{args.task}.pkl",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print('='*50)
    print(f"Saving results to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()