import argparse
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from accuwm.basic_synthid import basic_synthid_generator
import unbiased_watermark as uwm
from unbiased_watermark.synthid import SynthID_Reweight_fast
from experiments.tasks import get_summarization_ds, get_oeg_ds, get_oeg_human_tokens, get_eli5_ds
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

def _process_raw_prompt(tokenizer, prompt) -> str:
  """Add chat template to the raw prompt."""
  return tokenizer.apply_chat_template(
      [{'role': 'user', 'content': prompt.decode().strip('"')}],
      tokenize=False,
      add_generation_prompt=True,
  )


def main():
    parser = argparse.ArgumentParser(description='Run SynthID watermarking experiment')
    parser.add_argument('--private_key', type=int, default=0, required=True, help='Private key for watermarking')
    parser.add_argument('--model', type=str, default='google/gemma-7b-it', required=True, help='Model name or path')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=100, help='Top-k sampling parameter')
    parser.add_argument('--n', type=int, default=1, help='Number of tokens to generate per step')
    parser.add_argument('--task', type=str, choices=['summarization', 'oeg', 'oeg_human', 'eli5'], required=True, help='Task to perform')
    parser.add_argument('--ds_cut_len', type=int, default=None, help='Number of examples to use from dataset')
    parser.add_argument('--ds_begain', type=int, default=None, help='Number of examples to use from dataset')
    parser.add_argument('--ds_end', type=int, default=None, help='Number of examples to use from dataset')
    parser.add_argument('--access_token', type=str, default='YOUR_ACCESS_TOKEN', help='Access token for Hugging Face')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling table')
    parser.add_argument('--context_size', type=int, default=3, help='Context size for watermarking')
    parser.add_argument('--print_result', type=bool, default=False, help='Print result')
    parser.add_argument('--max_length', type=int, default=450, help='Maximum sequence length')
    parser.add_argument('--stop_words', type=str, nargs='+', default=None, help='Stop words to use for early stopping')
    args = parser.parse_args()
    print(args)
    # check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model and tokenizer
    print(f"Loading model {args.model}...")
    transformers.utils.logging.disable_progress_bar()
    model = AutoModelForCausalLM.from_pretrained(args.model, token=args.access_token).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.access_token)
    model.eval()
    
    # Get dataset based on task
    print(f"Loading dataset for task: {args.task}...")
    if args.task == 'summarization':
        ds = get_summarization_ds(args.ds_cut_len)
    elif args.task == 'oeg':
        ds = get_oeg_ds(args.ds_cut_len)
    elif args.task == 'oeg_human':
        ds = get_oeg_human_tokens(tokenizer, length=400, ds_cut_len=args.ds_cut_len)
    elif args.task == 'eli5':
        ds = get_eli5_ds(args.access_token, args.ds_begain, args.ds_end)
        print(f"Dataset selected from {args.ds_begain} to {args.ds_end}")
        print(f"Dataset length: {len(ds)}")

    # Initialize watermark components
    reweight = SynthID_Reweight_fast(
        sampling_table_size=2**16,
        sampling_table_seed=args.seed,
        device=device,
        ngram_len=args.context_size,
        private_key=args.private_key
    )
    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=args.context_size)
    cch = uwm.lm.ContextCodeHistory(batch_shape=(1,)) # only support batch size 1 for now
    
    
    # Add max length processor
    max_length_processor = MaxLengthLogitsProcessor(
        max_length=args.max_length,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Add stop words processor if stop words are provided
    if args.stop_words is not None:
        stop_words_processor = StopWordsLogitsProcessor(
            stop_words_ids=[],  # Will be set by set_stop_words
            eos_token_id=tokenizer.eos_token_id
        )
        stop_words_processor.set_stop_words(args.stop_words, tokenizer, device)
    
    results = []
    
    # Process each example
    for example in tqdm(ds, desc="Processing examples"):
        if args.task == 'oeg_human':
            input_ids = torch.tensor([example['tokens']])
        elif args.task == 'eli5':
            prompt = _process_raw_prompt(tokenizer, example.encode())
            input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
        else:
            input_ids = tokenizer(example['prompt'], return_tensors='pt')['input_ids'].to(device)
        
        # Initialize watermark generator with logits processors
        gen = basic_synthid_generator(
            reweight=reweight,
            cc_extractor=cc_extractor,
            cch=cch,
            model=model,
            input_ids=input_ids,
            temperature=args.temperature,
            top_k=args.top_k,
            n=args.n,
            apply_top_k=True,
            process_logits_kwargs={
                "logits_processor": transformers.LogitsProcessorList(
                    ([max_length_processor] if args.max_length else [])
                    + ([stop_words_processor] if args.stop_words else [])
                )
            }
        )
        
        # Generate watermarked text
        output_ids_list = []
        g_values_list = []
        
        for output_ids, g_values in gen:
            output_ids_list.append(output_ids)
            g_values_list.extend(g_values)
        
        # Combine all generated tokens
        output_ids = torch.cat(output_ids_list, dim=1)
        g_values = np.array(g_values_list)
        
        # Decode output
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Store results
        result = {
            'input_text': example['prompt'] if args.task != 'eli5' else example,
            'output_text': output_text,
            'output_ids': output_ids.tolist(),
            'g_values': g_values,   # shape (seq_len, depth)
            'g_values_mean': g_values.mean()
        }

        if args.print_result:
            print(result)
        results.append(result)
    
    # Save results
    save_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data_root",
        "synthid_basic",
        f"{to_display_model_name(args.model)}_{args.task}_range{args.ds_begain}_{args.ds_end}_temp{args.temperature}.pkl",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving results to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main() 