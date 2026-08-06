# Official Review Weaknesses and Questions

Source: OpenReview PDF exported on July 26, 2026.

This document contains the `Weaknesses` and `Questions` from the three
official reviews. Reviewer text is preserved in English. Our responses will
be drafted below each item.

## Reviewer crNB

### Weaknesses

#### crNB-W1: Readability and missing intuition

> The paper is quite difficult to read and follow. Much of the presentation
> is dominated by mathematical formalism, while the underlying intuition is
> often missing. Many sections abruptly reference previous works or other
> parts of the paper without sufficient explanation (e.g., Lines 176–189 and
> 196–197 in Section 4.3). Sections 3, 4, and 5, as well as much of the
> appendix, consist primarily of technical derivations with little effort
> devoted to building intuition. A high-level overview figure illustrating
> the overall PFR/MPFR workflow would substantially improve readability.

**Response (draft 1):** We thank the reviewer for identifying this
presentation problem. We will add a one-panel overview that follows one
speculative block through: (i) generating up to \(L\) draft levels,
(ii) evaluating the corresponding target distributions in parallel, and
(iii) using the same context-indexed keyed Poisson source to select both the
draft proposals and the target token. The central intuition is simple: the
target-side Poisson winner is always distributed exactly as the target model;
the drafter only determines whether that winner is already in the proposal
set, and therefore how long the block can continue. We will introduce this
intuition before the PFR definitions, add a short roadmap for Sections 3--5,
and move supporting derivations out of the main narrative.

#### crNB-W2: Motivation, contributions, and Poisson-process intuition

> The motivation and key contributions are not clearly communicated. It is
> difficult to understand how the proposed method fundamentally differs from
> prior speculative decoding and watermarking approaches, and why it can
> avoid the trade-off identified in previous work. The central intuition
> behind the Poisson-process framework is not sufficiently explained.

**Response (draft 2):** The relevant starting point for our method is PFR
coupling without communication, rather than maximal coupling. These are
different coupling constraints: PFR need not attain the maximal matching
probability of VSPS. Our claim is therefore not that watermarking makes PFR
match maximal coupling. Instead, once the no-communication PFR coupling is
fixed, its shared randomness can be keyed and made detectable without
introducing a further acceptance penalty.

PFR-NOWM is the direct ablation for this claim: it uses the same PFR coupling
with an unkeyed Poisson source. Across the eight Qwen settings in Tables 1--2
(two tasks and \(L=1,2,3,4\)), PFR and PFR-NOWM differ by only
0.09--0.48% in AATPS. Their token-rate differences have mixed signs, with a
mean PFR/NOWM ratio of 1.0005. Thus the watermark preserves the acceptance and
runtime behavior of the underlying PFR coupling.

The same comparison extends naturally to multiple drafts. INVARIANT is the
appropriate algorithm-level reference because it also targets
no-communication, drafter-invariant multi-draft sampling. MPFR achieves
comparable AATPS to INVARIANT while adding a recoverable watermark, and its
watermark evidence remains at the single-draft PFR level: ANLPPT changes by at
most 0.001 as \(B\) increases, while PFR/MPFR track the strong-watermark
references in TPR@1%FPR.

Coupling without communication also induces drafter invariance because the
emitted target token is determined by the context, target distribution, and
shared source rather than by a message from the drafter. We view this as a
useful design constraint—not as a necessary condition for strong detection in
every watermarking method. Its practical value is that the watermark remains
tied to target-side keyed randomness when the drafter is changed, and the
detector requires only the output and key. We will revise the paper to make
this contribution boundary explicit.

#### crNB-W3: Interpretation of Figures 1 and 2

> The experimental section could be significantly strengthened. Figures 1
> and 2 appear to provide the primary empirical evidence supporting the
> paper's claims, yet the figures, captions, and surrounding discussion offer
> limited guidance for interpretation. For example, the meaning of the
> "lookahead" parameter and its role in the decoding process are not clearly
> explained. In addition, the paper provides little analysis of the trends
> observed in these figures, leaving readers to infer the conclusions on
> their own.

**Response (draft 1):** We agree that the figures currently require too much
inference from the reader. The lookahead \(L\) is the number of sequential
draft levels proposed in one speculative block. A block invokes target
verification once and emits between 1 and \(L+1\) target-distributed tokens;
larger \(L\) can therefore raise AATPS, but also requires more drafter work.

We will revise the captions and discussion to make the intended readings
explicit. In Figure 1, moving right means fewer target invocations per output
token (higher AATPS), while moving up means stronger per-token watermark
evidence (higher ANLPPT). PFR nearly overlaps PFR-NOWM horizontally, showing
that adding the key does not reduce acceptance; MPFR moves further right
without diluting the watermark signal. Figure 2 then answers a different
question: PFR/MPFR track the strong-watermark references as the detection
budget grows, and under drafter substitution PFR changes by less than one TPR
point, compared with 12--22 points for MSE and 6--11 points for MSE-PSEUDO.

#### crNB-W4: Narrow evaluation scope

> The evaluation scope is relatively narrow. The experiments appear to focus
> on a single generation task (summarization) and a single target-drafter
> model combination (Qwen 2.5). This makes it difficult to assess how well
> the proposed method generalizes across different tasks, model families, or
> decoding configurations. Additional experiments would help establish the
> robustness of the approach.

**Response (draft 1):** We apologize that the main-text emphasis on one
headline cell made the evaluation appear narrower than it is. Appendix D
already reports the following \(2\times2\) evaluation matrix:

| Target / drafter | CNN/DailyMail | ELI5 |
|---|---|---|
| Qwen2.5-7B / Qwen2.5-0.5B | Summarization, 1000 prompts | Open-ended explanation, 1000 prompts |
| Vicuna-7B / Vicuna-68M | Summarization, 1000 prompts | Open-ended explanation, 1000 prompts |

For every model--task cell, Appendix D reports the single-draft
\(L\in\{1,2,3,4\}\) sweep and multi-draft
\(B\in\{2,4,6,8\}\) sweep, together with three watermark-score variants,
TPR@1%FPR, and quality audits. We additionally vary the Qwen drafter in scale
and temperature. We will surface this table and its main trends in the
main-text experiment setup instead of referring readers only to the appendix.

We have not evaluated code generation or formal reasoning, and will not imply
otherwise. The distributional correctness and watermark construction are
task-agnostic because they operate only on next-token distributions, so they
apply without changing the algorithm. However, realized AATPS and token rate
depend on target--drafter agreement and must be measured for each domain. ELI5
provides current evidence beyond summarization; code and reasoning remain
useful extensions rather than established empirical claims.

#### crNB-W5: Practical overhead and memory

> The practical implications of the method are not discussed in sufficient
> depth. Since speculative decoding is primarily motivated by inference
> efficiency, it would be valuable to include a more detailed analysis of
> computational overhead, implementation complexity, memory requirements,
> and runtime costs associated with PFR and MPFR.

**Response (draft 1):** We performed an additional 100-prompt A100 profile
using Qwen2.5-7B/0.5B, \(L=4\), top-\(k=50\), and 128-token outputs, with
separate uninstrumented throughput and synchronized component passes.

| Regime | Method | AATPS ↑ | TR (tok/s) ↑ | Direct ms/block ↓ | Max incremental peak ↓ | Keyed RNG |
|---|---|---:|---:|---:|---:|---:|
| Single | VSPS | 2.525 | 22.09 | 114.29 | 149.9 MiB | -- |
| Single | PFR-NOWM | 2.400 | 21.44 | 111.93 | 151.2 MiB | -- |
| Single | PFR | 2.479 | 22.16 | 111.84 | 148.8 MiB | 0.28 ms |
| Multi, \(B=4\) | INVARIANT | 3.092 | 24.56 | 125.91 | 910 MiB | -- |
| Multi, \(B=4\) | MPFR | 3.089 | 24.12 | 128.13 | 609 MiB | 0.52 ms |
| Multi, \(B=8\) | INVARIANT | 3.358 | 25.57 | 131.32 | 1816 MiB | -- |
| Multi, \(B=8\) | MPFR | 3.350 | 24.73 | 135.53 | 1182 MiB | 0.74 ms |

All values use the same first 100 CNN/DailyMail prompts on one A100 and are
reported as per-prompt means, matching the paper's aggregation convention.
AATPS, TR, direct block time, and memory are from the uninstrumented pass.
Keyed RNG is measured in the separate synchronized component pass and is
already part of block time.

Three observations follow directly. First, single-draft PFR has the same
wall-clock and memory profile as PFR-NOWM and VSPS; in the component pass,
target and drafter forwards account for 93.6% of PFR time, all arrival
sampling for 1.9%, and keyed RNG for only 0.28 ms/block (0.24%). The
paper-scale 1000-prompt results further show no systematic PFR/PFR-NOWM
acceptance or throughput loss across tasks and lookaheads.

Second, MPFR and INVARIANT have effectively identical AATPS. MPFR's TR is
1.8% lower at \(B=4\) and 3.3% lower at \(B=8\), while its incremental peak
memory is 33% and 35% lower, respectively. The complete keyed operation
remains only 0.52--0.74 ms/block. Component profiling localizes the block-time
difference to logits/arrival processing and non-model tree/cache bookkeeping,
not additional target or drafter calls.

The memory difference is primarily explained by how randomness is materialized
in the two reference implementations. INVARIANT eagerly allocates a float32
Gumbel-noise tensor of shape
\((T+L+1)\times B\times |V|\) for the full generation horizon, together with
a smaller persistent \(B\times L\times |V|\) buffer. With \(T=128\),
\(L=4\), and the Qwen vocabulary, these buffers occupy approximately 318 MiB
at \(B=4\) and 636 MiB at \(B=8\). The observed INVARIANT--MPFR peak-memory
differences are 301 MiB and 634 MiB, respectively. MPFR instead generates the
required \(m\times|V|\) Poisson arrivals on demand for the current context
(\(m\leq B\)) and releases them after selecting that context's proposals; it
does not retain randomness for all future output positions. Both methods
still allocate batched KV caches and logits, so MPFR memory also grows with
\(B\). This is an implementation-level memory advantage of the measured
reference paths, not a claim that every INVARIANT implementation must use
more memory; streaming its Gumbel randomness could reduce the gap.

Third, the table separates target-invocation efficiency (AATPS) from realized
wall-clock throughput. We will make this distinction explicit in the revision
rather than using “efficiency” without qualification.

In implementation terms, MPFR adds a context-indexed keyed source and
proposal-tree bookkeeping, but does not add target or drafter forward calls.
Detection uses only the final tokens and the key; it does not require storing
or communicating draft trajectories or acceptance metadata. We will include
this implementation summary alongside the expanded pseudocode.

> **TODO — Vicuna runtime diagnosis before submission:** Repeat the controlled
> component profile for Vicuna-7B/Vicuna-68M, ideally on the same 100
> CNN/DailyMail and ELI5 prompts, with \(L=4\), top-\(k=50\), and 128 output
> tokens. Include VSPS/PFR-NOWM/PFR and INVARIANT/MPFR at \(B=4,8\), and
> record AATPS, TR, direct ms/block, target/draft forward time,
> logits/arrival sampling, other tree/cache bookkeeping, and peak memory.
>
> The current tables already narrow the explanation. On Vicuna, PFR and
> PFR-NOWM have similar TR, so the large PFR--VSPS gap is not caused by adding
> the watermark key. PFR and VSPS also have close AATPS (within roughly
> 1--2.4% at \(L=4\)); MPFR and INVARIANT show the same AATPS parity.
> Therefore, the model-pair difference is primarily a per-block runtime
> question rather than an acceptance/block-count question.
>
> A likely cause is that the Vicuna drafter has only 68M parameters, compared
> with the 0.5B Qwen drafter. Its model-forward path is much cheaper, so fixed
> logits, Poisson-arrival, Python, and cache/tree bookkeeping can occupy a
> larger fraction of total block time. The vectorized VSPS/INVARIANT paths
> would then benefit more from the cheap drafter. This interpretation is
> consistent with the Qwen component profile, but it must remain a hypothesis
> until the Vicuna component times are measured. Replace this TODO with the
> measured result before submitting the rebuttal.

#### crNB-W6: Presentation issues

> Several presentation issues affect readability. For example, citation
> formatting and writing style could be improved in multiple places (e.g.,
> Lines 30 and 55), and the overall presentation would benefit from clearer
> organization and exposition.

**Response (draft 1):** We agree and will revise the cited lines, standardize
citation placement and terminology, and reorganize the exposition around the
overview described in crNB-W1. In particular, we will define each symbol at
first use, state the purpose of each technical subsection before its
derivation, and place the detailed PFR/MPFR proofs after the algorithmic
intuition and pseudocode.

### Questions

#### crNB-Q1: Practical computational overhead

> What is the practical computational overhead of PFR and MPFR compared to
> existing speculative decoding approaches?

**Response (draft 1):** Please see crNB-W5 for the full controlled profile.
In brief, PFR matches VSPS throughput in the controlled A100 profile (22.16
versus 22.09 token/s), while adding the watermark key to PFR contributes only
0.28 ms per block and no measurable memory increase. Against the
no-communication multi-draft baseline INVARIANT, MPFR has comparable AATPS
and TR gaps of 1.8% at \(B=4\) and 3.3% at \(B=8\). Component profiling
localizes the difference to non-model logits, arrival, and tree/cache
operations rather than extra model inference, while MPFR uses approximately
one-third less incremental peak GPU memory.

#### crNB-Q2: Tasks beyond summarization

> Have the authors evaluated the proposed framework on generation tasks
> beyond summarization, such as reasoning, code generation, or open-ended
> text generation? If not, do they expect the same benefits to transfer to
> these settings, and why?

**Response (draft 1):** Please see crNB-W4. The submitted appendix already
evaluates open-ended explanation on ELI5 in addition to summarization, across
both Qwen and Vicuna model families with 1000 prompts per model--task cell.
The exact sampling and watermark guarantees transfer to other autoregressive
tasks without task-specific changes, but end-to-end acceptance and runtime
remain empirical because target--drafter agreement is domain dependent. We
have not yet established results on code generation or formal reasoning and
will present these as future evaluation rather than extrapolating from the
current evidence.

## Reviewer WyYi

### Weaknesses

#### WyYi-W1: Larger models

> The model sizes used in the experiments are too small. Testing only on 7B
> models is not enough. It remains unknown how this heavy sampling logic
> performs with 72B or large size LLMs.

**Response (draft 1):** We agree that evidence at a larger model scale is
important. We therefore ran an additional controlled experiment with
Qwen2.5-72B-Instruct as the target and Qwen2.5-0.5B-Instruct as the drafter on
8 A100-40GB GPUs. All methods use exactly the same balanced 8-GPU sharding,
the same 100 CNN/DailyMail prompts, \(L=4\), top-\(k=50\), and at most 128 new
tokens.

| Method | \(B\) | TR | AATPS | Incremental peak GPU memory | ANLPPT-U | LPPL | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| VSPS | 1 | 10.025 | 2.380 | 141.2 MiB | 0.0031 | 0.2429 | 0.2334 |
| PFR-NOWM | 1 | 9.838 | 2.339 | 141.6 MiB | 0.0024 | 0.2364 | 0.2435 |
| PFR | 1 | 9.771 | 2.321 | 139.3 MiB | 0.0178 | 0.2357 | 0.2354 |
| MPFR | 4 | 10.218 | 2.988 | 563.3 MiB | 0.0156 | 0.2428 | 0.2378 |
| INVARIANT | 4 | 10.328 | 3.057 | 870.2 MiB | 0.0018 | 0.2645 | 0.2354 |
| MPFR | 8 | 9.922 | 3.274 | 1103.8 MiB | 0.0160 | 0.2465 | 0.2376 |
| INVARIANT | 8 | 9.633 | 3.307 | 1735.7 MiB | 0.0019 | 0.2609 | 0.2353 |

For each no-watermark row, ANLPPT-U is computed with its corresponding
detector and serves only as the matched null reference.

The heavy-sampling concern is not borne out at this scale. In single-draft
PFR, target and draft forwards account for 95.2% of measured time, arrival
sampling for 1.24%, and keyed RNG for 0.22%. In MPFR, model forwards account
for 90.2% at \(B=4\) and 89.4% at \(B=8\); arrival sampling accounts for
1.87--2.31% and keyed RNG for 0.33--0.41%. MPFR is within 1.1% of INVARIANT
in TR at \(B=4\), is 3.0% faster at \(B=8\), and uses 35--36% less
incremental peak memory in the current implementations. The watermark signal
also remains separated from the matched no-watermark references, while LPPL
and ROUGE-L remain comparable. PFR/MPFR KL/WS ratios are 0.966, 1.013, and
1.032 for \(B=1,4,8\), respectively.

#### WyYi-W2: Novelty

> The proposed method seems a simple mixture of existing methods. It just
> merges recent work for speculative sampling with the standard watermark.

**Response (draft 1):** We agree that the high-level ingredients are related
to prior speculative sampling and keyed Gumbel sampling, and we will sharpen
the novelty boundary. Appendix A.2 already establishes the equivalence
between the single-draft target-side PFR primitive and keyed Gumbel-max; we
will not claim this primitive alone as new, nor describe MPFR as a generic
wrapper for arbitrary watermark schemes.

The contribution is the joint multi-draft construction rather than a simple
sequential composition of two existing algorithms. A single context-indexed
keyed Poisson object must simultaneously produce the correct draft marginals,
preserve the exact target marginal for every emitted token, provide an
output-only recoverable watermark, and remain invariant to drafter-side
trajectories conditional on the stopping time. MPFR further handles repeated
branches, heterogeneous drafts, target-tree verification, and the realized
target path. Our correctness, invariance, and acceptance results establish
these properties jointly. We will use the narrower term “construction” or
“algorithm,” rather than a generic watermark “framework.”

#### WyYi-W3: Token rate

> The token rates in the appendix show that this method is slower than the
> baselines.

**Response (draft 1):** We agree that AATPS and wall-clock token rate must be
reported separately. The new 72B results in WyYi-W1 provide a controlled
token-rate comparison and clarify where the remaining differences arise.

Against the matched no-watermark construction, PFR is only 0.7% slower than
PFR-NOWM (9.771 versus 9.838 token/s), with nearly identical measured AATPS,
block time, and memory. The keyed operation itself is only 0.22% of measured
time. Relative to VSPS, PFR is 2.5% slower in TR and has correspondingly lower
AATPS (2.321 versus 2.380); this reflects the difference between the PFR
coupling without communication and maximal-coupling VSPS, rather than a
watermark-embedding penalty.

For the matched no-communication multi-draft comparison, MPFR and INVARIANT
are close at \(B=4\) (10.218 versus 10.328 token/s), while MPFR is faster at
\(B=8\) (9.922 versus 9.633 token/s). We will therefore avoid an unqualified
claim of speedup over optimized autoregressive or maximal-coupling decoding.
Our supported efficiency claim is that watermarking introduces negligible
additional cost within the PFR/MPFR coupling, and that MPFR preserves the
acceptance benefit of the corresponding no-communication multi-draft method
without a material end-to-end throughput penalty.

### Questions

#### WyYi-Q1: Refer to weaknesses

> Please refer to weakness.

**Response:** Addressed through WyYi-W1–W3.

## Reviewer v9yM

### Weaknesses

> Overall I view this work as a technic solid but novelty limited paper, see
> the following details:

#### v9yM-W1: Single-draft novelty and the no-go theorem

> Single-draft case is not new. PFR is functionally identical to MWS with
> Gumbel-max from [25] (correct me if I'm wrong). The claim of "circumventing
> the no-go theorem" is quite overclaimed since MWS already does this and
> under the single-draft setting the proposed method stills falls into this
> trade-off. The no-go theorem is framed under the single draft setting,
> while using multi draft to overcome it is quite 'unfair'.

**Response (draft 1):** We agree with the reviewer that the single-draft
target-side sampling primitive is equivalent to keyed Gumbel-max; Appendix
A.2 already proves this equivalence. We will revise the presentation so that
single-draft PFR is not claimed as a new watermark primitive. Its role is to
establish the coupling viewpoint and provide the controlled PFR-NOWM
ablation.

The comparison boundary is important. PFR is a coupling without
communication and is not maximal coupling in general. We do not claim that it
matches maximal-coupling VSPS. The supported single-draft claim is narrower:
once this PFR coupling is fixed, keying its shared source adds a recoverable
watermark without further acceptance or runtime loss. In our controlled
100-prompt profile, PFR and PFR-NOWM have essentially identical measured block
time (111.84 versus 111.93 ms), while PFR has AATPS 2.479 and TR 22.16
token/s, versus 2.400 and 21.44 token/s for PFR-NOWM.

We also agree that “circumventing the no-go theorem” is too broad if read as
a claim about the theorem's single-draft setting. We will replace it with the
more precise statement that our coupling-level construction lies outside the
sequential-composition setting analyzed by [25], and that the substantive
gain is the multi-draft extension: it recovers acceptance under the
no-communication constraint while retaining the same strong watermark
signal. We will not present multi-draft sampling as invalidating a
single-draft impossibility result.

#### v9yM-W2: Multi-draft novelty

> Multi-draft extension is straightforward extension. Given single-draft PFR,
> the extension amounts to generating |V| × B exponentials from the same seed
> and taking order statistics. The Poisson process theory proves correctness,
> but the algorithmic step itself is natural and unsurprising. Also, the
> acceptance rate gain is a very intuitive result.

**Response (draft 1):** We agree that “take the first \(B\) arrivals” is the
natural high-level idea once the Poisson representation is available. The
nontrivial part is making this idea into an exact, watermarkable speculative
tree rather than merely drawing \(B\) samples at one context.

At every occupied context, MPFR must generate \(B\) i.i.d. mapped samples,
merge repeated tokens into branch multiplicities, evaluate the target over
the resulting unique contexts, and follow only the realized target path. The
same context-indexed keyed source must simultaneously guarantee: (i) each
draft has the correct marginal, (ii) every emitted token has the exact target
marginal, (iii) the final text carries an output-only recoverable watermark,
and (iv) the result is invariant to drafter-side trajectories conditional on
the stopping time. Our mapping-theorem construction, correctness proof,
heterogeneous-draft extension, and acceptance bound establish these jointly.

The controlled 100-prompt results show that these properties do not impose an
acceptance penalty relative to the closest no-communication baseline:
MPFR/INVARIANT AATPS is 3.089/3.092 at \(B=4\) and 3.350/3.358 at \(B=8\),
while MPFR retains a stable watermark signal (ANLPPT-U 0.0489 and 0.0490).
We will present the novelty as this joint multi-draft construction and
guarantee, rather than the order-statistics operation alone.

#### v9yM-W3: Stopping-time drafter invariance

> Stopping-time drafter invariance is a post-hoc characterization. In my view,
> it follows naturally from the construction (target output depends only on
> k, context, P) rather than being a design principle. It is more like a bonus
> property, not an independent contribution. Also, I'm not persuaded why this
> property is necessary, does it equivalent to maintain the watermark
> detection? Happy to discuss this.

**Response (draft 1):** We agree that stopping-time drafter invariance is not
necessary for strong detection. MWS is an immediate counterexample: it can
produce a strong watermark without satisfying our invariance property.
Detection strength and robustness to drafter replacement are distinct
requirements.

Our motivation is deployment stability. A detector normally observes only the
final text and secret key, while a speculative decoder also has hidden draft
trajectories and acceptance decisions. Stopping-time invariance allows the
drafter to affect block length and efficiency, but, conditional on that
length, prevents it from introducing additional dependence into the emitted
target-keyed prefix. Consequently, changing or upgrading the drafter does not
change which target-side randomness the detector is testing, and no draft
trace or acceptance metadata is needed.

We view this as a useful guarantee induced by the construction, not as a
universal prerequisite for watermarking or a standalone source of detection
power. The drafter-substitution experiment measures its practical
consequence: PFR changes by less than one TPR point when only the drafter is
changed, whereas MSE changes by 12--22 points and MSE-PSEUDO by 6--11 points.
We will revise the contribution statement accordingly.

#### v9yM-W4: End-to-end speed and runtime breakdown

> No actual end-to-end speedup. Token rate shows MPFR is slower than basic
> autoregressive. The overhead of generating and sorting |top-k| × B
> exponentials plus watermark bookkeeping outweighs the AATPS gains. In my
> view, the overhead is dominated by three parts: target model inference
> (largest), draft model inference, and watermark embedding, while the
> proposed method adds a heavy latency on watermark embedding part. Also, why
> is there such a large difference in token rates between different mode
> pairs?

**Response (draft 1):** We agree that AATPS alone does not establish
end-to-end acceleration over an optimized autoregressive implementation. We
will distinguish target-invocation efficiency (AATPS) from realized
wall-clock throughput and avoid using unqualified “speedup” for the current
reference implementation.

We additionally ran a controlled 100-prompt A100 profile of the multi-draft
setting with Qwen2.5-7B/0.5B, \(L=4\), top-\(k=50\), and 128-token outputs.
The uninstrumented end-to-end results are:

| \(B\) | Method | AATPS ↑ | TR (tok/s) ↑ | Direct ms/block ↓ |
|---:|---|---:|---:|---:|
| 4 | INVARIANT | 3.064 | 24.56 | 125.91 |
| 4 | MPFR | 3.076 | 24.12 | 128.13 |
| 8 | INVARIANT | 3.337 | 25.57 | 131.32 |
| 8 | MPFR | 3.334 | 24.73 | 135.53 |

The matched AATPS values show that the TR difference is not caused by lower
acceptance or additional target-model invocations. We then synchronized CUDA
around each runtime component in a separate diagnostic pass:

| \(B\) | Method | Target forward | Draft forwards | Logits | Arrival/Gumbel sampler | Other control | Keyed RNG subset |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | INVARIANT | 30.65 ms | 89.79 ms | 0.91 ms | 1.03 ms | 6.48 ms | -- |
| 4 | MPFR | 28.54 ms | 83.47 ms | 2.91 ms | 4.07 ms | 13.28 ms | 0.52 ms |
| 8 | INVARIANT | 34.64 ms | 90.59 ms | 1.03 ms | 1.08 ms | 6.87 ms | -- |
| 8 | MPFR | 30.64 ms | 84.39 ms | 4.29 ms | 5.83 ms | 15.85 ms | 0.74 ms |

All entries are directly measured milliseconds per decoding block. The MPFR
arrival timer includes exponential generation and first-arrival selection;
the keyed-RNG value is a measured subset of that timer and must not be added
again. Model inference indeed dominates MPFR, accounting for 84.7% and 81.6%
of the instrumented block time at \(B=4\) and \(B=8\). However, keyed RNG is
only 0.4--0.5% of block time. Even if the complete arrival sampler is counted
as watermark embedding, it occupies only 3.1% and 4.1% of MPFR block time,
respectively. Therefore, the measurements do not support the hypothesis that
watermark bookkeeping dominates total latency. The remaining
MPFR--INVARIANT difference is distributed across logits/arrival processing
and multi-draft tree/cache/control operations. We nevertheless agree with the
reviewer's end-to-end observation: these non-model costs can outweigh a
modest AATPS gain. The precise conclusion is therefore not that MPFR has zero
overhead, but that the overhead is not dominated by the watermark key or by
an additional model call.

The memory result is also favorable in the measured implementation. INVARIANT
eagerly stores float32 Gumbel randomness for all future positions with shape
\((T+L+1)\times B\times|V|\), whereas MPFR generates arrivals on demand for
the current context. This accounts for almost all of the measured 301 MiB
(\(B=4\)) and 634 MiB (\(B=8\)) peak-memory differences. We treat this as an
implementation-level result; streaming INVARIANT's randomness could reduce
the gap.

The large TR variation across model pairs is likewise not explained by the
watermark or acceptance: on Vicuna, PFR is not slower than PFR-NOWM, while
PFR/VSPS and MPFR/INVARIANT have close AATPS. A likely reason is that the 68M
Vicuna drafter makes model forwards much cheaper, exposing fixed
logits/arrival and Python/cache costs more strongly. We are adding a matched
Vicuna component profile to test this explanation and will replace this
hypothesis with measured component times in the final response.

#### v9yM-W5: Specific algorithm versus general framework

> The proposed method is a specific watermark algorithm (or extension of
> Gumbel-max) rather than a general framework (like [22, 25]) can apply to
> other watermark schemes. In this way, again, I view it as a technic solid
> but novelty limited paper.

**Response (draft 1):** We agree with this scope distinction. MPFR is a
specific keyed-Poisson/Gumbel-style watermark construction, not a generic
wrapper that can turn an arbitrary watermark scheme into a multi-draft
decoder. We will replace broad uses of “framework” with “construction” or
“algorithm” and state this limitation explicitly.

The contribution we intend to claim is narrower: a single random-object
construction that jointly provides exact target marginals, multiple i.i.d.
draft proposals, an output-only recoverable watermark, and stopping-time
drafter invariance, together with correctness and acceptance guarantees. The
same generated text can be evaluated with the U, Li, and PL score families,
but this detector compatibility should not be confused with support for
arbitrary watermark embeddings.

#### v9yM-W6: Readability

> A minor point: the writing is a little bit hard to follow.

**Response (draft 1):** We agree and will add a high-level PFR/MPFR workflow
figure before the formal definitions. We will explain the coupling intuition
first, define lookahead and draft multiplicity at their first use, move
supporting derivations behind the algorithmic description, and expand the
captions of Figures 1--2 to state the intended trends. We will also narrow the
no-go, framework, and invariance claims as described above.

### Questions

#### v9yM-Q1: Necessity of stopping-time drafter invariance

> Why stopping-time drafter invariance is necessary? I understand the
> importance of detection power but this stopping-time drafter invariance is
> not equivalent to a strong detection performance right? i.e., strong
> detection power can be achieved without this stopping-time drafter
> invariance?

**Response (draft 1):** Stopping-time drafter invariance is not necessary for
strong watermark detection, and we will not claim equivalence between the two.
It addresses a different axis: whether the detector-visible target-keyed
signal remains stable when hidden draft trajectories or the drafter model
change. Conditional on the emitted block length, our property removes
drafter-specific dependence from the output prefix, so detection needs only
the final text and key. The <1-point PFR TPR change under drafter substitution,
versus 12--22 points for MSE and 6--11 for MSE-PSEUDO, is evidence for this
robustness benefit rather than evidence that invariance is required for
detection.

#### v9yM-Q2: Scaling beyond top-k 50

> How does performance scale if top-k is increased beyond 50 or removed
> entirely? Also, for MSE or MWS, when using Gumbel-max, they do not need this
> top-k sampling to reduce the watermarking latency. I also wonder, what is
> the performance like token rate without top-k.

**Response (draft 1):** We evaluated
top-\(k\in\{50,100,500,\mathrm{None}\}\) on the same 100 CNN/DailyMail
prompts, with \(L=4\) and 128 output tokens on the same A100 server. Each
PFR/MPFR runtime cell used one GPU without same-GPU contention; runtime and
watermark signal were measured in separate passes. The CPU/NumPy-sensitive
MSE/MWS cells were rerun strictly serially.

For PFR:

| top-\(k\) | TR | AATPS | ANLPPT-U |
|---:|---:|---:|---:|
| 50 | 22.16 | 2.479 | 0.0507 |
| 100 | 21.93 | 2.477 | 0.0519 |
| 500 | 22.00 | 2.470 | 0.0519 |
| None | 21.94 | 2.462 | 0.0525 |

For MPFR, we additionally report the directly measured block time and the
arrival-sampling component that contains exponential generation and
first-arrival selection:

| \(B\) | top-\(k\) | TR | AATPS | Direct ms/block | Arrival ms/block | Keyed RNG subset | ANLPPT-U |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 50 | 24.12 | 3.076 | 128.13 | 4.07 | 0.52 | 0.0489 |
| 4 | 100 | 24.26 | 3.062 | 126.97 | 4.11 | 0.53 | 0.0507 |
| 4 | 500 | 24.07 | 3.055 | 127.67 | 4.05 | 0.51 | 0.0512 |
| 4 | None | 23.95 | 3.057 | 128.39 | 4.14 | 0.52 | 0.0512 |
| 8 | 50 | 24.73 | 3.334 | 135.53 | 5.83 | 0.74 | 0.0490 |
| 8 | 100 | 25.05 | 3.322 | 133.47 | 5.70 | 0.72 | 0.0504 |
| 8 | 500 | 25.27 | 3.311 | 131.87 | 5.69 | 0.73 | 0.0507 |
| 8 | None | 24.69 | 3.306 | 134.76 | 5.92 | 0.75 | 0.0511 |

Removing top-\(k\) therefore changes PFR TR by -1.0%, MPFR TR by -0.7% at
\(B=4\) and -0.2% at \(B=8\), while AATPS and watermark signal remain
stable. More directly addressing the proposed bottleneck, MPFR arrival
sampling remains within 4.05--4.14 ms/block at \(B=4\) and 5.69--5.92
ms/block at \(B=8\); keyed RNG remains within 0.51--0.53 and 0.72--0.75
ms/block, respectively. Thus, the observed runtime does not grow
proportionally with \(k\). In these runs, model outputs are vocabulary-shaped
in every condition and top-\(k\) is applied during logits processing.
Increasing or removing top-\(k\) changes the decoding distribution but not
the model-forward tensor shapes.

The reviewer is also correct that the Gumbel-max baselines do not require
top-\(k\) to control watermark latency. In a separate serial run, removing
top-\(k\) changes MSE TR from 20.15 to 20.31 token/s and MWS TR from 19.36 to
19.31 token/s. Their Gumbel watermark step remains 9.17--9.72 ms/block and
incremental peak memory is unchanged. We will add these results and clarify
that top-\(k=50\) is a shared decoding configuration, not a requirement of
MSE/MWS.

#### v9yM-Q3: Refer to weaknesses

> For other questions see "weakness".

**Response:** Addressed through v9yM-W1–W6.
