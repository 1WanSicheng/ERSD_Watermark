# Prompt configurations (paper appendix material)

> Auto-generated reference for the paper's "Prompt templates" appendix
> section. All numbers reported in the main tables use the prompts below.

## 1. Underlying message structure (before chat template)

Defined in [`experiments/_shared.py:load_prompts`](../experiments/_shared.py).

### CNN/DailyMail (`dataset: cnn_dailymail`)
```python
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user",
   "content": "Summarize the following article in 3-5 sentences:\n\n"
              + article[:1500]}
]
```
Article is truncated to **1500 characters** before tokenization.

### ELI5 (`dataset: eli5`)
```python
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user",
   "content": "Please explain like I'm five:\n\n" + question}
]
```

### Hu & Huang paper-exact summarization (`dataset: cnn_paper_summarization`)
Used only when reproducing Hu & Huang 2024 Table 1 directly. Single user
message, no system role; replicates `improving_KL/experiments/tasks.py:get_summarization_ds`.
```python
[
  {"role": "user",
   "content": "System:Summarize the following article.\n"
              + "INPUT:" + article[:1000] + "\n"
              + "OUTPUT:"}
]
```

## 2. What each model actually sees after the chat template wrapper

The chat template is applied by `tokenizer.apply_chat_template(...,
add_generation_prompt=True)` when `use_chat_template=True` (default).
For base targets we set `use_chat_template=False` — the runner then
emits **only the last user message as raw text**, dropping the system role.

### 2.1 Qwen2.5-7B-Instruct on CNN/DailyMail
Native ChatML template:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Summarize the following article in 3-5 sentences:

{article[:1500]}<|im_end|>
<|im_start|>assistant
```

### 2.2 Vicuna-7B-v1.5 on CNN/DailyMail
The released Vicuna checkpoint ships *without* a `chat_template`
attribute, so we explicitly inject the FastChat v1.1 template before
tokenization. Effective string:
```
You are a helpful assistant. USER: Summarize the following article in 3-5 sentences:

{article[:1500]} ASSISTANT:
```

### 2.3 Llama-7B (base, `huggyllama/llama-7b`) on CNN/DailyMail
Base targets are run with `use_chat_template: false` because the
tokenizer inherits a Llama-2-Chat template that the base model was not
trained on (feeding `[INST]<<SYS>>...` would produce chat-template
artifacts in the output). The model receives only the user message:
```
Summarize the following article in 3-5 sentences:

{article[:1500]}
```
The system role (`"You are a helpful assistant."`) is dropped on this path.

### 2.4 ELI5 — same per-model pattern

| Target | Effective prompt |
|---|---|
| Qwen2.5-7B-Instruct | `<\|im_start\|>system\nYou are a helpful assistant.<\|im_end\|>\n<\|im_start\|>user\nPlease explain like I'm five:\n\n{question}<\|im_end\|>\n<\|im_start\|>assistant\n` |
| Vicuna-7B-v1.5 | `You are a helpful assistant. USER: Please explain like I'm five:\n\n{question} ASSISTANT:` |
| Llama-7B base  | `Please explain like I'm five:\n\n{question}` |

## 3. Sampling configuration

Identical across all (model, dataset) cells:

| Parameter | Value |
|---|---|
| `top_k` | 50 |
| `top_p` | 1.0 (effectively disabled; only `top_k` truncates) |
| `temperature` | 1.0 |
| `max_new_tokens` | 128 |
| `private_key` | fixed across all runs (32-bit value, see config file) |

## 4. LaTeX-ready snippet

```latex
\subsection{Prompt templates}\label{app:prompts}
For all instruction-tuned targets we apply each model's native chat
template via the HuggingFace \texttt{apply\_chat\_template} interface;
for base targets (\texttt{huggyllama/llama-7b}) we feed the user message
as raw text to avoid the inherited Llama-2-Chat template producing
chat-format artifacts. The Vicuna v1.5 release ships without a
\texttt{chat\_template} attribute, so we explicitly inject the
FastChat v1.1 template before tokenization.

\paragraph{CNN/DailyMail (summarization).}
The message list passed to the chat template is
\begin{verbatim}
[
  {"role": "system",
   "content": "You are a helpful assistant."},
  {"role": "user",
   "content": "Summarize the following article in 3-5 sentences:\n\n"
              + article[:1500]}
]
\end{verbatim}
The article is truncated to 1500 characters before tokenization.

\paragraph{ELI5 (open-ended QA).}
\begin{verbatim}
[
  {"role": "system",
   "content": "You are a helpful assistant."},
  {"role": "user",
   "content": "Please explain like I'm five:\n\n" + question}
]
\end{verbatim}

For each prompt the model is asked to generate up to 128 new tokens
under \texttt{top\_k}\,$=50$, \texttt{top\_p}\,$=1.0$, and temperature
$T=1.0$.  The watermark key is fixed across all runs.
```

## 5. Pointers

- Code: [`experiments/_shared.py:load_prompts`](../experiments/_shared.py),
  [`experiments/_shared.py:encode_prompt`](../experiments/_shared.py),
  [`experiments/_shared.py:_maybe_inject_chat_template`](../experiments/_shared.py)
- Hu & Huang reference task definitions:
  [`improving_KL/experiments/tasks.py`](../improving_KL/experiments/tasks.py)
- Per-experiment config files referencing each `dataset` value:
  [`experiments/configs/`](../experiments/configs/)
