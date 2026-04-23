from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from datasets import load_dataset
from generator import BasicGenerator, SpeculativeGenerator, InvariantGenerator
import strategy as spec_strategy
from strategy import SingleDraftStrategy, SpecInferStrategy, SpecTrStrategy
try:
    from strategy import PoissonSingleDraftStrategy
except ImportError:
    PoissonSingleDraftStrategy = None
from strategy import InvariantSingleDraftStrategy, InvariantMultiDraftStrategy, StrongMultiDraftStrategy
from tqdm import tqdm
from datetime import datetime
import argparse
import json
import os
import random
import torch
import csv
import numpy as np

WARMUP = 10


class _CacheTensorList:
    def __init__(self, cache, attr):
        self.cache = cache
        self.attr = attr

    def __len__(self):
        return len(self.cache.layers)

    def __getitem__(self, idx):
        return getattr(self.cache.layers[idx], self.attr)

    def __setitem__(self, idx, value):
        setattr(self.cache.layers[idx], self.attr, value)


def install_dynamic_cache_compat():
    if not hasattr(DynamicCache, "key_cache"):
        DynamicCache.key_cache = property(lambda self: _CacheTensorList(self, "keys"))
    if not hasattr(DynamicCache, "value_cache"):
        DynamicCache.value_cache = property(lambda self: _CacheTensorList(self, "values"))


def install_quiet_sampling():
    def quiet_gumbel_sample(logits, noise=None, dim=-1):
        if noise is None:
            noise = torch.zeros_like(logits).uniform_(0, 1)
        return (logits + (-torch.log(-torch.log(noise.clamp_min(1e-20))))).argmax(dim=dim)

    spec_strategy.gumbel_sample = quiet_gumbel_sample

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_prompt(inputs, dset_name):
    if dset_name == 'openai/gsm8k':
        question = inputs['question']
        prompt = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': f'{question}'}
        ]
    elif dset_name == 'openai/openai_humaneval':
        question = inputs['prompt']
        prompt = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': f'Complete the code:\n{question}'}
        ]
    elif dset_name == 'facebook/natural_reasoning':
        question = inputs['question']
        prompt = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': f'{question}'}
        ]
    elif dset_name == 'mandarjoshi/trivia_qa':
        question = inputs['question']
        prompt = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': f'{question}'}
        ]
    elif dset_name == 'google-research-datasets/mbpp':
        question = inputs['text']
        prompt = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': f'{question} Use Python.'}
        ]
    elif dset_name == 'ucinlp/drop':
        passage = inputs['passage']
        question = inputs['question']
        prompt = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': f'{passage}\n\n{question}'}
        ]
    return prompt

def test_config(config, model_small, model_large, tokenizer, dset, gen_length, num_tests, writer, dset_name):
    max_num_drafts = config['max_num_drafts']
    max_draft_len = config['max_draft_len']
    temperature = config['temperature']

    if config['strategy'] == 'single_draft':
        strategy = SingleDraftStrategy(model_large, model_small, tokenizer, max_draft_len, 1)
    elif config['strategy'] == 'specinfer':
        strategy = SpecInferStrategy(model_large, model_small, tokenizer, max_draft_len, max_num_drafts)
    elif config['strategy'] == 'spectr':
        strategy = SpecTrStrategy(model_large, model_small, tokenizer, max_draft_len, max_num_drafts)
    elif config['strategy'] == 'invariant':
        strategy = InvariantSingleDraftStrategy(model_large, model_small, tokenizer, max_draft_len, 1)
    elif config['strategy'] == 'invariant_multi_draft':
        strategy = InvariantMultiDraftStrategy(model_large, model_small, tokenizer, max_draft_len, max_num_drafts)
    elif config['strategy'] == 'strong_multi_draft':
        strategy = StrongMultiDraftStrategy(model_large, model_small, tokenizer, max_draft_len, max_num_drafts)
    elif config['strategy'] == 'ppr_single_draft':
        if PoissonSingleDraftStrategy is None:
            raise ImportError("PoissonSingleDraftStrategy is not available in strategy.py")
        strategy = PoissonSingleDraftStrategy(
            model_large, model_small, tokenizer,
            max_draft_len,  # 可以写 1
            alpha=config.get('alpha', 1.0),
            top_k=config.get('top_k', 50)
        )

    if config['strategy'] == 'basic':
        # Don't use speculative decoding
        generator = BasicGenerator(model_large)
    elif config['strategy'] == 'invariant' or config['strategy'] == 'invariant_multi_draft' or config['strategy'] == 'strong_multi_draft':
        generator = InvariantGenerator(strategy)
    else:
        generator = SpeculativeGenerator(strategy)

    for i in tqdm(range(WARMUP), total=WARMUP, desc="Warming up"):
        prompt = create_prompt(dset[i], dset_name)
        input_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, 
                                                  return_tensors='pt').to(model_large.device)
        outputs = generator(input_ids=input_ids, eos_token_id=tokenizer.eos_token_id, 
                            max_new_tokens=gen_length, temperature=temperature)
    
    for i in tqdm(range(num_tests), total=num_tests, desc="Running tests"):
        prompt = create_prompt(dset[i], dset_name)
        input_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, 
                                              return_tensors='pt').to(model_large.device)
        outputs = generator(input_ids=input_ids, eos_token_id=tokenizer.eos_token_id, 
                            max_new_tokens=gen_length, temperature=temperature)
        dict = {
            'config_name': config['name'],
            'test_num': i,
            'acceptance_rate': outputs.acceptance_rate, 
            'token_rate': outputs.token_rate, 
            'avg_generation_time': outputs.avg_generation_time, 
            'avg_verification_time': outputs.avg_verification_time, 
            'num_invocations': outputs.num_invocations, 
            'total_time': outputs.total_time
        }
        writer.writerow(dict)

def run_test(test, fname):
    model_name_large = test['target_model']
    tokenizer = AutoTokenizer.from_pretrained(model_name_large)
    model_large = AutoModelForCausalLM.from_pretrained(model_name_large, device_map='auto', 
                                                       torch_dtype=torch.bfloat16)
    model_large.eval()

    model_name_small = test['draft_model']
    model_small = AutoModelForCausalLM.from_pretrained(model_name_small, device_map='auto',
                                                       torch_dtype=torch.bfloat16)
    model_small.eval()

    for dset_name in test['datasets']:
        if dset_name == 'openai/gsm8k':
            dset = load_dataset(dset_name, 'main')['train']
        elif dset_name == 'openai/openai_humaneval':
            dset = load_dataset(dset_name, 'openai_humaneval')['test']
        elif dset_name == 'facebook/natural_reasoning':
            dset = load_dataset(dset_name, 'default')['train']
        elif dset_name == 'mandarjoshi/trivia_qa':
            dset = load_dataset(dset_name, 'rc')['train']
        elif dset_name == 'google-research-datasets/mbpp':
            dset = load_dataset(dset_name, 'full')['train']
        elif dset_name == 'ucinlp/drop' :
            dset = load_dataset(dset_name, 'default')['train']
        else:
            raise ValueError(f"Unsupported dataset ({dset_name})")
        
        gen_length = test['gen_length']
        num_tests = test['num_test_prompts']
        if num_tests == -1 or num_tests > len(dset):
            num_tests = len(dset)
        
        csv_fname = f'{fname}_{dset_name.split("/")[-1]}.csv'
        field_names = ['config_name', 'test_num', 'acceptance_rate', 'token_rate', 'avg_generation_time', 
                       'avg_verification_time', 'num_invocations', 'total_time']
        
        with open(csv_fname, 'w', newline='') as ofp:
            writer = csv.DictWriter(ofp, fieldnames=field_names)
            writer.writeheader()

            for config in test['configurations']:
                test_config(config, model_small, model_large, tokenizer, dset, gen_length, num_tests,
                            writer, dset_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_script', type=str, required=True,
        help='Script listing the tests to run'
    )
    parser.add_argument(
        '--seed', type=int, required=True,
        help='PyTorch random seed'
    )
    args = parser.parse_args()
    install_dynamic_cache_compat()
    install_quiet_sampling()
    seed_everything(args.seed)

    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d-%H%M')
    fname = f'outputs/{args.test_script[:-5]}_{formatted_time}'

    with open(args.test_script) as fp:
        j = json.load(fp)
        run_test(j, fname)
