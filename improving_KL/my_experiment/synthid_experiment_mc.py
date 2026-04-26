import argparse
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from accuwm.mc_synthid import mc_synthid_sample_generator
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
    parser.add_argument('--mc_private_key', type=int, default=1, required=True, help='Private key for watermarking')
    parser.add_argument('--method', type=str, choices=['mc_mse', 'mc_mws', 'mc_2keys', 'mc_comb1', 'mc_comb2'], required=True, help='method for speculative sampling')
    parser.add_argument('--model', type=str, default='facebook/opt-6.7b', required=True, help='Model name or path')
    parser.add_argument('--ref_model', type=str, default='facebook/opt-1.3b', required=True, help='Reference model name or path')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=100, help='Top-k sampling parameter')
    parser.add_argument('--n', type=int, default=2, help='Number of tokens to generate per step')
    parser.add_argument('--task', type=str, choices=['summarization', 'oeg', 'oeg_human', 'eli5'], required=True, help='Task to perform')
    parser.add_argument('--dataset', type=str, choices=['cnn', 'c4'], default='cnn', help='Dataset to use for oeg task')
    parser.add_argument('--ds_cut_len', type=int, default=None, help='Number of examples to use from dataset')
    parser.add_argument('--ds_begain', type=int, default=None, help='Number of examples to use from dataset')
    parser.add_argument('--ds_end', type=int, default=None, help='Number of examples to use from dataset')
    parser.add_argument('--access_token', type=str, default='YOUR_ACCESS_TOKEN', help='Access token for Hugging Face')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling table')
    parser.add_argument('--context_size', type=int, default=4, help='Context size for watermarking')
    parser.add_argument('--print_result', type=bool, default=False, help='Print result')
    parser.add_argument('--max_length', type=int, default=250, help='Maximum sequence length')
    parser.add_argument('--stop_words', type=str, nargs='+', default=None, help='Stop words to use for early stopping')
    parser.add_argument('--folder_name', type=str, default='test', required=True, help='Folder name to save results')
    args = parser.parse_args()
    print(args)
    # check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model and tokenizer
    transformers.utils.logging.disable_progress_bar()
    print('='*50)
    print(f"Loading model {args.model} and ref model {args.ref_model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, token=args.access_token).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model, token=args.access_token).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.access_token)
    model.eval()
    ref_model.eval()
    
    # Get dataset based on task
    print('='*50)
    print(f"Loading dataset for task: {args.task}...")
    if args.task == 'summarization':
        ds = get_summarization_ds(args.ds_cut_len)
    elif args.task == 'oeg':
        ds = get_oeg_ds(args.ds_cut_len, args.dataset)
    elif args.task == 'oeg_human':
        ds = get_oeg_human_tokens(tokenizer, length=400, ds_cut_len=args.ds_cut_len)
    elif args.task == 'eli5':
        ds = get_eli5_ds(args.access_token, args.ds_begain, args.ds_end)
        print(f"Dataset selected from {args.ds_begain} to {args.ds_end}")
        print(f"Dataset length: {len(ds)}")
    save_path = os.path.join(
            os.path.dirname(__file__),
            "..",
        "data_root",   
        args.folder_name,
        f"{to_display_model_name(args.model)}_{to_display_model_name(args.ref_model)}_{args.task}_{args.method}_range{args.ds_begain}_{args.ds_end}_temp{args.temperature}_n{args.n}.pkl",
    )

    # Check if results already exist
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            existing_results = pickle.load(f)
        num_existing = len(existing_results)
        print(f"Found existing results with {num_existing} entries, will skip these.")
    else:
        existing_results = []
        num_existing = 0

    # Skip already processed examples, here use dataset.select, do not use ds = ds[num_existing:]
    if args.task == 'eli5':
        ds = ds[num_existing:]
    else:
        ds = ds.select(range(num_existing, len(ds)))
    print(f"Dataset length after skipping: {len(ds)}")

    # Initialize watermark components
    reweight = SynthID_Reweight_fast(
        sampling_table_size=2**16,
        sampling_table_seed=args.seed,
        device=device,
        ngram_len=args.context_size,
        private_key=args.private_key
    )
    if args.method == 'mc_2keys':
        mc_reweight = SynthID_Reweight_fast(
            sampling_table_size=2**16,
            sampling_table_seed=args.seed + 1,  # different seed for mc reweight
            device=device,
            ngram_len=args.context_size,
            private_key=args.mc_private_key
        )
    else:
        mc_reweight = None
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
    
    # Initialize results
    results = existing_results

    # Process each example
    print('='*50)
    for example in tqdm(ds, desc="Processing examples"):
        if args.task == 'oeg_human':
            input_ids = torch.tensor([example['tokens']])
        elif args.task == 'eli5':
            if tokenizer.chat_template is not None:
                prompt = _process_raw_prompt(tokenizer, example.encode())
            else:
                prompt = example
            input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
        else:
            input_ids = tokenizer(example['prompt'], return_tensors='pt')['input_ids'].to(device)
        
        # Initialize watermark generator with logits processors
        gen = mc_synthid_sample_generator(
            method=args.method,
            reweight=reweight,
            cc_extractor=cc_extractor,
            cch=cch,
            model=model,
            ref_model=ref_model,
            input_ids=input_ids,
            mc_private_key=args.mc_private_key,
            temperature=args.temperature,
            top_k=args.top_k,
            n=args.n,
            seed=args.seed + 1,  # This seed is used for mc sampling, different seed for mc reweight
            psedo_r=True,
            process_logits_kwargs={
                "logits_processor": transformers.LogitsProcessorList(
                    ([max_length_processor] if args.max_length else [])
                    + ([stop_words_processor] if args.stop_words else [])
                )
            }
        )
        
        # Generate watermarked text
        output_ids_list = []
        gen_seq_lens = []
        prob_ratios = []
        
        for output_ids, prob_ratio in gen:
            output_ids_list.append(output_ids)
            gen_seq_lens.append(output_ids.shape[-1])
            prob_ratios.extend(prob_ratio)
        # Combine all generated tokens
        output_ids = torch.cat(output_ids_list, dim=1)  # shape (1, seq_len)

        # Decode output
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Store results
        result = {
            'input_text': example['prompt'] if args.task != 'eli5' else example,
            'output_text': output_text,
            'output_ids': output_ids.tolist(),
            'gen_seq_lens': gen_seq_lens,
            'avg_seq_len': np.mean(gen_seq_lens),
            'prob_ratios': np.array(prob_ratios)
        }

        # Compute g_values
        g_values = reweight.compute_g_values(output_ids, input_ids)
        assert g_values.shape[1] == output_ids.shape[1]
        result['g_values'] = g_values.cpu().numpy().squeeze(0)
        result['g_values_mean'] = g_values.cpu().numpy().squeeze(0).mean()
        if args.method == 'mc_2keys':
            mc_g_values = mc_reweight.compute_g_values(output_ids, input_ids)
        else:
            mc_g_values = None
        if mc_g_values is not None:
            assert mc_g_values.shape[1] == output_ids.shape[1]
            result['mc_g_values'] = mc_g_values.cpu().numpy().squeeze(0)
            result['mc_g_values_mean'] = mc_g_values.cpu().numpy().squeeze(0).mean()

        # Compute r
        cch_r = uwm.lm.ContextCodeHistory(batch_shape=(1,))
        r_values = uwm.lm.get_r_values(cc_extractor, cch_r, args.mc_private_key, output_ids, input_ids)
        assert r_values.shape[1] == output_ids.shape[1], f"get r_values shape: {r_values.shape}, output_ids shape: {output_ids.shape}"
        result['r_values'] = r_values.squeeze(0)

        # Compute context repetition mask
        cch_detect = uwm.lm.ContextCodeHistory(batch_shape=(1,))
        context_repetition_mask = uwm.lm.compute_context_repetition_mask(cc_extractor, cch_detect, output_ids, input_ids)
        assert context_repetition_mask.shape[1] == output_ids.shape[1], f"context_repetition_mask shape: {context_repetition_mask.shape}, output_ids shape: {output_ids.shape}"
        result['context_repetition_mask'] = context_repetition_mask.squeeze(0)

        if args.print_result:   # test mode
            print(result)
            
        else:
            results.append(result)
            # Save results
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)

    print('='*50)
    print(f"Saving results to {save_path}...")
    
if __name__ == '__main__':
    main() 