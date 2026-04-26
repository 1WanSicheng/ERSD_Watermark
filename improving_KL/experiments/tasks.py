from datasets import load_dataset
import os


#  def get_translation_ds():
#      wmt16 = load_dataset("wmt16", "ro-en")
#      ds = wmt16["test"]
#      ds = ds.select(range(0, 1000))
#      ds = exp_debug_cut(ds)
#      ds = ds.flatten()
#
#      def create_prompt(d, idx):
#          s = {}
#          s["idx"] = idx
#          #  s["prompt"] = "Translate from English to Romanian: " + d["translation.en"]
#          s[
#              "prompt"
#          ] = f"System:Translate from English to Romanian.\nINPUT:{d['translation.en']}\nOUTPUT:"
#          return s
#
#      ds = ds.map(create_prompt, with_indices=True, remove_columns=ds.column_names)
#      return ds


def get_summarization_ds(ds_cut_len=None):
    cnn_daily = load_dataset("cnn_dailymail", "3.0.0").shuffle(seed=42)
    ds = cnn_daily["test"]
    ds = ds.filter(lambda x: len(x["article"]) < 3000)
    ds = ds.select(range(0, 1000))
    if ds_cut_len is not None:
        ds = ds.select(range(0, ds_cut_len))

    def create_prompt(d, idx):
        s = {}
        s["idx"] = idx
        #  s["prompt"] = d["article"] + "\nTL;DR:"
        #  s["prompt"] = d["article"][:1000] + "\nTL;DR:\n"
        s[
            "prompt"
        ] = f"System:Summarize the following article.\nINPUT:{d['article'][:1000]}\nOUTPUT:"
        #  s["prompt"] = d["article"][:1000] + "\nRe-type (copy the above word by word):\n"
        return s

    ds = ds.map(create_prompt, with_indices=True, remove_columns=ds.column_names)
    return ds


def get_oeg_ds(ds_cut_len=None, dataset='cnn'):
    if dataset == 'cnn':
        cnn_daily = load_dataset("cnn_dailymail", "3.0.0").shuffle(seed=42)
        ds = cnn_daily["test"]
        ds = ds.select(range(0, 1000))
    elif dataset == 'c4':
        c4 = load_dataset("allenai/c4", "realnewslike", split="validation", token='YOUR_ACCESS_TOKEN').shuffle(seed=42)
        ds = c4.select(range(0, 1000))
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    if ds_cut_len is not None:
        ds = ds.select(range(0, ds_cut_len))

    def smart_truncate(content, length):
        if len(content) <= length:
            return content
        else:
            return " ".join(content[: length + 1].split(" ")[0:-1])

    def create_prompt(d, idx):
        s = {}
        s["idx"] = idx
        if dataset == 'cnn':
            s["prompt"] = smart_truncate(d["article"], length=100)
        elif dataset == 'c4':
            s["prompt"] = smart_truncate(d["text"], length=100)
        return s

    ds = ds.map(create_prompt, with_indices=True, remove_columns=ds.column_names)
    return ds


def get_oeg_human_tokens(tokenizer, length, ds_cut_len=None):
    cnn_daily = load_dataset("cnn_dailymail", "3.0.0").shuffle(seed=42)
    ds = cnn_daily["train"]
    ds = ds.select(range(0, 50000))
    if ds_cut_len is not None:
        ds = ds.select(range(0, ds_cut_len))

    def create_tokens(d, idx, length=length):
        tokens = tokenizer(d["article"], return_tensors="pt", truncation=True, max_length=length)["input_ids"][0]
        if tokens.shape[-1] < 400:
            return None  # Drop this example
        return {
            "idx": idx,
            "text": d["article"],
            "tokens": tokens.tolist()  # keeping list structure like original
        }

    ds = ds.map(create_tokens, with_indices=True, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x is not None)  # Remove None entries
    return ds


def get_eli5_ds(access, begain=None, end=None):
    eli5 = load_dataset("Pavithree/eli5", token=access)
    ds = eli5['train']['title']
    if begain is not None and end is not None:
        ds = ds[begain:end]
    else:
        ds = ds.select(range(0, 1000))

    return ds   #  return a list of strings


def get_eli5_ds_dataset(ds_cut_len=None):
    eli5 = load_dataset("Pavithree/eli5", token='YOUR_ACCESS_TOKEN')    # replace with your own token
    data = eli5['train']['title'][:10000]
    if ds_cut_len is not None:
        data = data[:ds_cut_len]

    ds = []
    for d in data:
        item = {}
        item['prompt'] = d
        ds.append(item)

    return ds   #  return a list of dicts


def get_eli5_human_tokens(tokenizer, length, access_token, ds_cut_len=None):
    eli5 = load_dataset("Pavithree/eli5", token=access_token)
    data = eli5['train']['answers'][:50000]
    if ds_cut_len is not None:
        data = data[:ds_cut_len]
    ds = []
    for d in data:
        answer = d['text'][0]   # use the first answer
        tokens = tokenizer(answer, return_tensors="pt", truncation=True, max_length=length)["input_ids"][0]
        ds.append({"text": answer, "tokens": tokens.tolist()})

    return ds 


def get_wiki_human_tokens(tokenizer, length, access_token, ds_cut_len=None):
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", token=access_token).shuffle(seed=42)
    data = wiki["train"]
    data = data.select(range(0, 50000))
    if ds_cut_len is not None:
        data = data.select(range(0, ds_cut_len))
    ds = []
    for row in data:
        tokens = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=length)["input_ids"][0]
        ds.append({"text": row["text"], "tokens": tokens.tolist()})

    return ds