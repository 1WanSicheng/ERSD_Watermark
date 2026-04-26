import os
import json
import _jsonnet
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys

# Add parent directory to Python path to import tasks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments import tasks

from .worker import Worker, process_human_data


def to_display_model_name(s: str) -> str:
    if "/" in s:
        s = s.split("/")[-1]
    s = s.replace("-chat-hf", "")
    return s


def estimate_fpass(config: Dict[str, Any]) -> int:
    n_fm1 = (
        1  # basic
        + len(config["ns"])  # mc
        + len(config["reweights"])  # basic_uwm
        + len(config["ns"]) * len(config["reweights"])  # mc_uwm_strength
        + len(config["ns"]) * len(config["reweights"])  # mc_uwm_speed
    )
    n_fm2 = len(config["seeds"])
    return config["ds_cut_len"] * n_fm1 * n_fm2


def estimate_time(config: Dict[str, Any]) -> float:
    fpass = estimate_fpass(config)
    # for max_length=128, a6000, 4.6s per fpass
    t = fpass * 4.6
    return t


def enrich_experiment_params(d: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Enrich experiment parameters based on config settings."""
    experiments = []
    
    if "basic" in config["methods"]:
        experiments.append({**d, "method": "basic", "n": 1, "reweight": "none"})
        
    if "basic_uwm" in config["methods"]:
        experiments.extend([
            {**d, "method": "basic_uwm", "n": 1, "reweight": reweight}
            for reweight in config["reweights"]
        ])
        
    if "mc" in config["methods"]:
        experiments.extend([
            {**d, "method": "mc", "n": n, "reweight": "none"}
            for n in config["ns"]
        ])
        
    if "mc_uwm_strength" in config["methods"]:
        experiments.extend([
            {**d, "method": "mc_uwm_strength", "n": n, "reweight": reweight}
            for n in config["ns"]
            for reweight in config["reweights"]
        ])
        
    if "mc_uwm_speed" in config["methods"]:
        experiments.extend([
            {**d, "method": "mc_uwm_speed", "n": n, "reweight": reweight}
            for n in config["ns"]
            for reweight in config["reweights"]
        ])
    
    if "mc_uwm_synthid" in config["methods"]:
        experiments.extend([
            {**d, "method": "mc_uwm_synthid", "n": n, "reweight": reweight, "mc_private_key": bytes(config["mc_private_key"], "utf-8")}
            for n in config["ns"]
            for reweight in config["reweights"]
        ])
        
    if "mc_uwm_synthid_psedo_r" in config["methods"]:
        experiments.extend([
            {**d, "method": "mc_uwm_synthid_psedo_r", "n": n, "temperature": config["temperature"], "reweight": reweight, "mc_private_key": bytes(config["mc_private_key"], "utf-8")}
            for n in config["ns"]
            for reweight in config["reweights"]
        ])
    return experiments


def run_experiment(config: Dict[str, Any], worker: Worker, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run experiments for a given config and dataset."""
    results = []
    
    # Process each item in the dataset
    for item in tqdm(dataset, desc="Processing dataset"):
        # Add common parameters
        item = {
            **item,
            "max_length": config["max_length"],
            "private_key": bytes(config["private_key"], "utf-8"),
        }
        
        # Generate all experiment variations
        experiments = enrich_experiment_params(item, config)
        
        # Run experiments for each seed
        for seed in config["seeds"]:
            for exp in experiments:
                exp["seed"] = seed
                result = worker.process(exp)
                results.append(result)
                
    return results


def run(config: Dict[str, Any], args):
    """Main experiment runner."""
    # Load dataset
    if config["ds_name"] == "summarization":
        ds = tasks.get_summarization_ds(config.get("ds_cut_len", None))
    elif config["ds_name"] == "oeg":
        ds = tasks.get_oeg_ds(config.get("ds_cut_len", None))
    elif config["ds_name"] == "eli5":
        ds = tasks.get_eli5_ds_dataset(config.get("ds_cut_len", None))
    else:
        raise ValueError(f"Unknown dataset: {config['ds_name']}")
        
    # Setup worker
    worker_param = {
        k: config[k]
        for k in [
            "model_str",
            "ref_model_str",
            "task",
            "device",
            "print_output",
            "assert_cch",
            "assert_log_p_values",
        ]
    }
    worker = Worker(param=worker_param)
    
    # Run experiments
    results = run_experiment(config, worker, ds)
    
    # Save results directly to pickle
    save_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data_root",
        config["data_folder"],
        config["task"], 
        f"{to_display_model_name(config['model_str'])}_{to_display_model_name(config['ref_model_str'])}_{args.method}_size{config['ds_cut_len']}_temp{config['temperature']}_n{config['ns'][0]}.pkl",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    import pickle
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {save_path}")


def main():
    import json
    import _jsonnet
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="basic")
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--n", type=int, default=2)
    args = parser.parse_args()
    print(args)
    # Load configs
    configs = json.loads(
        _jsonnet.evaluate_file(
            os.path.join(os.path.dirname(__file__), "..", "my_experiment", "configs", f"{args.method}_{args.model}_n{args.n}_config.jsonnet")
        )
    )
    
    # Run each config
    for config in configs:
        seconds = estimate_time(config)
        print(
            f"Running {config['task']}. Estimated time: {seconds / 3600:.2f} GPU hours"
        )
        run(config, args)


def test_worker():
    """Test the worker with a simple example."""
    # model_str = "huggyllama/llama-7b"
    # ref_model_str = "JackFram/llama-68m"
    model_str = "google/gemma-7b-it"
    ref_model_str = "google/gemma-2b-it"
    import numpy as np
    
    np.seterr(all="raise")
    worker_param = {
        "model_str": model_str,
        "ref_model_str": ref_model_str,
        "task": "oeg_scan_n",
        "device": "cuda:0",
        "hf_token": os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        "print_output": True,
    }
    
    worker = Worker(param=worker_param)
    
    for seed in tqdm(range(1)):
        for method in ["mc_uwm_synthid_psedo_r"]:
            if "uwm" in method:
                _rws = ["deltagumbel"]
            else:
                _rws = ["none"]
            if "mc" in method:
                _ns = [3]
            else:
                _ns = [1]
            for reweight in _rws:
                for n in _ns:
                    r = worker.process(
                        {
                            "prompt": "<bos><start_of_turn>user\nwhat's the difference between a forest and a wood?<end_of_turn>\n<start_of_turn>model\n",
                            "seed": seed,
                            "method": method,
                            "reweight": reweight,
                            "private_key": b"1234",
                            "n": n,
                            "max_length": 128,
                            "mc_private_key": b"4321",
                            "temperature": 0.7,
                        }
                    )

def get_human_data():

    import transformers

    model_str = "huggyllama/llama-7b"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_str,
        token=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )
    ds = tasks.get_oeg_human_tokens(tokenizer, length=400, ds_cut_len=1000)
    results = []
    for item in tqdm(ds):
        results.append(process_human_data(item["tokens"],
                                          vocab_size=tokenizer.vocab_size,
                                          seed=1,
                                          private_key=b"1234",
                                          mc_private_key=b"4321"))
    save_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data_root",
        "human_data",
        "human_data_1000.pkl",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    import pickle
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    if os.environ.get("EXP_DEBUG", None) == "0":
        test_worker()
        exit()
    elif os.environ.get("GET_HUMAN_DATA", None) == "0":
        get_human_data()
        exit()
    main() 