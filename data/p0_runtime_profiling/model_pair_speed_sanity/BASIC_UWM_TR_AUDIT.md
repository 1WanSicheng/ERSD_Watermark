# Basic-UWM versus PFR token-rate audit

## Setup

- Qwen2.5-7B-Instruct / Qwen2.5-0.5B-Instruct
- Same A100, process, 10 CNN/DailyMail prompts, seeds, top-k 50, and maximum
  output length
- L=4, 128 maximum output tokens, one warm-up prompt per method
- End-to-end timing and a separate synchronized component pass

## Counting and aggregation

Both methods count only actually generated output tokens and use CUDA
synchronization immediately before and after the generation timing window.
Detection and quality metrics are outside the window.

| Method | Output tokens | Mean prompt TR | Global `sum(tokens)/sum(time)` TR |
|---|---:|---:|---:|
| Basic-UWM | 1044 | 36.903 | 36.902 |
| PFR | 1066 | 21.615 | 21.532 |

Changing from the paper's mean-of-prompt-rates aggregation to global throughput
changes PFR by only 0.4% and Basic-UWM by less than 0.01%. It cannot explain
the observed gap.

## Direct component result

All times below are synchronized, directly measured milliseconds per decoding
block. For Basic-UWM, one block is one autoregressive output step. For PFR, one
block is one target verification step with four autoregressive draft steps.

| Component | Basic-UWM | PFR |
|---|---:|---:|
| Target forward | 24.85 ms | 26.30 ms |
| Four draft forwards | -- | 79.69 ms |
| Logits processing | 0.20 ms | 1.62 ms |
| UWM step / PFR arrival sampling | 1.75 ms | 0.94 ms |
| Final sampling | 0.21 ms | -- |
| Other cache/Python work | 0.35 ms | 3.45 ms |
| Instrumented total | 27.37 ms | 112.01 ms |

Basic-UWM's watermark operation is included in its timing and accounts for
6.4% of its step time. PFR arrival sampling is only 0.8% of its block time.
Therefore the result is not caused by omitting Basic-UWM watermark latency or
by heavy PFR keyed sampling.

## Break-even calculation

The uninstrumented Basic-UWM cost is 27.10 ms/output token. PFR emits 2.369
tokens/block, so it must finish a block within

`2.369 * 27.10 = 64.20 ms`

to beat Basic-UWM. Its measured block time is 110.02 ms. Target verification
already costs 26.30 ms, leaving at most about 37.9 ms for all draft and control
work. The four draft forwards alone cost 79.69 ms, or 19.92 ms each.

Even ignoring every non-model PFR operation, the four draft forwards would
need to fall below roughly 8 ms each for L=4 to cross the measured break-even
point. This is the dominant failure mode.

## Conclusion

The current runner uses a correct and effectively matched token-rate
denominator. The lack of end-to-end speedup comes from the latency behavior of
the Hugging Face single-token drafter path: despite a 0.5B-versus-7B parameter
gap, one 0.5B draft call takes about 80% as long as one 7B autoregressive target
call (19.92 versus 24.85 ms). Four strictly sequential draft calls therefore
overwhelm the saved target calls. This is shared by VSPS and PFR and explains
why their TR decreases as lookahead increases.

Raw result: `qwen25_basic_pfr_audit_n10_L4.json`.
