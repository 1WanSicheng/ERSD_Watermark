#!/usr/bin/env python3
from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

EXPECTED_DATASETS = ["openai/gsm8k", "openai/openai_humaneval", "facebook/natural_reasoning"]
EXPECTED_METHODS = ["ashish_invariant", "mpfr_batched_torchgen_cached", "multi_draft_pfr_batched_cached"]
EXPECTED_B = [2, 4, 6, 8]
DISPLAY_METRICS = ["token_rate", "aatps", "be", "acceptance_fraction", "target_forward_calls_per_token", "draft_forward_calls_per_token"]
METRICS = ["num_generated_tokens", "total_time", "token_rate", "num_steps", "accepted_tokens_total", "attempted_draft_tokens_total", "aatps", "be", "tokens_per_step", "acceptance_fraction", "normalized_aatps", "avg_block_time", "target_forward_calls_total", "draft_forward_calls_total", "target_forward_calls_per_token", "draft_forward_calls_per_token"]
METHOD_LABEL = {"ashish_invariant":"GLS/invariant", "mpfr_batched_torchgen_cached":"MPFR-TorchGen", "multi_draft_pfr_batched_cached":"MultiDraft-PFR"}

def timestamp(name: str) -> str:
    m = re.search(r"(20\d\d-\d\d-\d\d-\d{6})", name)
    return m.group(1) if m else "0000-00-00-000000"

def seed_from_name(name: str):
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else np.nan

def norm_method(x: str) -> str:
    return {"mpfr_torchgen_cached":"mpfr_batched_torchgen_cached", "multi_draft_pfr_cached":"multi_draft_pfr_batched_cached"}.get(str(x), str(x))

def read_raw(input_dir: Path):
    rows, frames = [], []
    for p in sorted(input_dir.glob("three_method_raw*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            rows.append({"file":p.name,"rows":0,"kind":"read_error","error":str(e)})
            continue
        if len(df) == 0:
            rows.append({"file":p.name,"rows":0,"kind":"empty/useless","seed_in_filename":seed_from_name(p.name),"timestamp":timestamp(p.name)})
            continue
        df = df.copy()
        if "seed" not in df.columns:
            df["seed"] = seed_from_name(p.name)
        if "method" not in df.columns:
            df["method"] = df["config_name"].astype(str).str.replace(r"_4_\d+$", "", regex=True)
        df["method"] = df["method"].map(norm_method)
        if "max_num_drafts" not in df.columns:
            df["max_num_drafts"] = df["config_name"].astype(str).str.extract(r"_4_(\d+)$").astype(int)
        df["source_file"] = p.name
        df["source_timestamp"] = timestamp(p.name)
        frames.append(df)
        id_col = "prompt_index" if "prompt_index" in df.columns else "test_num"
        counts = df.groupby(["seed","dataset","config_name"], dropna=False)[id_col].nunique().reset_index(name="n")
        rows.append({"file":p.name,"rows":len(df),"kind":"raw","seed_values":";".join(map(str,sorted(df.seed.dropna().unique()))),"seed_in_filename":seed_from_name(p.name),"timestamp":timestamp(p.name),"datasets":";".join(sorted(df.dataset.dropna().unique())),"configs":df.config_name.nunique(),"complete_blocks_at_150":int((counts.n>=150).sum()),"partial_blocks_at_150":int(((counts.n>0)&(counts.n<150)).sum()),"block_counts":"; ".join(f"{r.dataset}|{r.config_name}:{int(r.n)}" for r in counts.itertuples())})
    all_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return all_raw, pd.DataFrame(rows)

def expected_grid(seeds: List[int]) -> pd.DataFrame:
    rec=[]
    for seed in sorted(seeds):
        for ds in EXPECTED_DATASETS:
            for method in EXPECTED_METHODS:
                prefix = {"ashish_invariant":"ashish_invariant", "mpfr_batched_torchgen_cached":"mpfr_torchgen_cached", "multi_draft_pfr_batched_cached":"multi_draft_pfr_cached"}[method]
                for B in EXPECTED_B:
                    rec.append({"seed":seed,"dataset":ds,"method":method,"max_num_drafts":B,"config_name":f"{prefix}_4_{B}"})
    return pd.DataFrame(rec)

def select_blocks(all_raw: pd.DataFrame, target_prompts: int):
    id_col = "prompt_index" if "prompt_index" in all_raw.columns else "test_num"
    block_cols = ["seed","dataset","method","config_name","max_num_drafts"]
    fc = all_raw.groupby(block_cols+["source_file","source_timestamp"], dropna=False)[id_col].nunique().reset_index(name="unique_prompt_count")
    fc["complete"] = fc.unique_prompt_count >= target_prompts
    selection=[]
    selected_parts=[]
    for key, g in fc.groupby(block_cols, dropna=False):
        gg = g.copy().sort_values(["complete","unique_prompt_count","source_timestamp","source_file"])
        complete = gg[gg.complete]
        if len(complete):
            chosen = complete.sort_values(["source_timestamp","source_file"]).iloc[-1]
            status="selected_complete"
            part = all_raw
            for col,val in zip(block_cols,key):
                part = part[part[col] == val]
            part = part[part.source_file == chosen.source_file].sort_values(id_col).drop_duplicates(id_col, keep="first").head(target_prompts).copy()
            selected_parts.append(part)
        else:
            chosen=gg.iloc[-1]
            status="incomplete_unselected"
        row=dict(zip(block_cols,key)); row.update({"status":status,"selected_file":chosen.source_file if status=="selected_complete" else "","best_file":chosen.source_file,"best_unique_prompts":int(chosen.unique_prompt_count),"candidate_files":"; ".join(f"{r.source_file}:{int(r.unique_prompt_count)}" for r in g.itertuples())})
        selection.append(row)
    clean=pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    return clean, pd.DataFrame(selection)

def summarize_seed(clean: pd.DataFrame) -> pd.DataFrame:
    group_cols=["seed","dataset","method","config_name","max_num_drafts"]
    rec=[]
    for key,g in clean.groupby(group_cols, dropna=False):
        row=dict(zip(group_cols,key)); row["prompt_count"]=len(g)
        for m in METRICS:
            if m in g.columns:
                row[f"{m}_mean"]=g[m].mean(); row[f"{m}_std"]=g[m].std(ddof=1); row[f"{m}_count"]=g[m].count()
        rec.append(row)
    return pd.DataFrame(rec).sort_values(group_cols).reset_index(drop=True)

def avg_over_seed_summaries(ss: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    mean_cols=[c for c in ss.columns if c.endswith("_mean")]
    rec=[]
    for key,g in ss.groupby(group_cols, dropna=False):
        row=dict(zip(group_cols,key if isinstance(key, tuple) else (key,)))
        row["seed_count"]=g.seed.nunique(); row["seeds"]=";".join(map(str,sorted(g.seed.unique())))
        for c in mean_cols:
            base=c[:-5]
            row[f"{base}_mean"]=g[c].mean(); row[f"{base}_std_across_seeds"]=g[c].std(ddof=1); row[f"{base}_se_across_seeds"]=g[c].std(ddof=1)/np.sqrt(len(g)) if len(g)>1 else np.nan
        rec.append(row)
    return pd.DataFrame(rec).reset_index(drop=True)

def method_B_overall(ss: pd.DataFrame) -> pd.DataFrame:
    metric_cols=[f"{m}_mean" for m in DISPLAY_METRICS if f"{m}_mean" in ss.columns]
    per_seed=ss.groupby(["seed","method","max_num_drafts"], dropna=False)[metric_cols].mean().reset_index()
    rec=[]
    for key,g in per_seed.groupby(["method","max_num_drafts"], dropna=False):
        row={"method":key[0],"max_num_drafts":key[1],"seed_count":g.seed.nunique(),"seeds":";".join(map(str,sorted(g.seed.unique())))}
        for m in DISPLAY_METRICS:
            c=f"{m}_mean"
            if c in g.columns:
                row[f"{m}_mean"]=g[c].mean(); row[f"{m}_std_across_seeds"]=g[c].std(ddof=1); row[f"{m}_se_across_seeds"]=g[c].std(ddof=1)/np.sqrt(len(g)) if len(g)>1 else np.nan
        rec.append(row)
    return pd.DataFrame(rec).sort_values(["max_num_drafts","method"]).reset_index(drop=True)

def rel_vs_baseline(overall: pd.DataFrame, baseline="ashish_invariant") -> pd.DataFrame:
    rec=[]
    for B,g in overall.groupby("max_num_drafts"):
        b=g[g.method==baseline]
        if b.empty: continue
        b=b.iloc[0]
        for _,r in g[g.method!=baseline].iterrows():
            row={"method":r.method,"max_num_drafts":B,"baseline":baseline,"seed_count":r.seed_count,"seeds":r.seeds}
            for m in DISPLAY_METRICS:
                rv=r.get(f"{m}_mean", np.nan); bv=b.get(f"{m}_mean", np.nan)
                row[f"{m}_method"]=rv; row[f"{m}_baseline"]=bv; row[f"{m}_rel_pct"]=100*(rv/bv-1) if pd.notna(rv) and pd.notna(bv) and bv!=0 else np.nan
            rec.append(row)
    return pd.DataFrame(rec)

def plot_overall(outdir: Path, overall: pd.DataFrame, prefix: str):
    if plt is None or overall.empty: return
    pdir=outdir/"plots"; pdir.mkdir(exist_ok=True)
    for m in DISPLAY_METRICS:
        c=f"{m}_mean"
        if c not in overall.columns: continue
        fig,ax=plt.subplots(figsize=(7.5,4.5))
        for method,g in overall.groupby("method"):
            g=g.sort_values("max_num_drafts")
            ax.plot(g.max_num_drafts, g[c], marker='o', label=METHOD_LABEL.get(method,method))
        ax.set_xlabel("Number of drafts B"); ax.set_ylabel(m.replace('_',' ')); ax.set_title(f"{m.replace('_',' ')} by B")
        ax.grid(True, alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(pdir/f"{prefix}_{m}_by_B.png", dpi=180); plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input_dir', default='.')
    ap.add_argument('--output_dir', default='cleaned_three_method_results_v2')
    ap.add_argument('--target_prompts', type=int, default=150)
    args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    raw,manifest=read_raw(Path(args.input_dir)); manifest.to_csv(out/'file_manifest.csv',index=False)
    if raw.empty:
        print('No raw rows found.'); return
    clean,sel=select_blocks(raw,args.target_prompts); sel.to_csv(out/'block_selection_report.csv',index=False); clean.to_csv(out/'clean_raw_selected.csv',index=False)
    seeds=sorted(raw.seed.dropna().unique().astype(int).tolist())
    grid=expected_grid(seeds)
    cov=grid.merge(sel[['seed','dataset','method','config_name','status','selected_file','best_file','best_unique_prompts','candidate_files']], on=['seed','dataset','method','config_name'], how='left')
    cov['status']=cov.status.fillna('missing_no_rows'); cov.to_csv(out/'coverage_expected_blocks.csv',index=False)
    cov[cov.status!='selected_complete'].to_csv(out/'missing_or_incomplete_expected_blocks.csv',index=False)
    ss=summarize_seed(clean); ss.to_csv(out/'summary_by_seed.csv',index=False)
    complete_counts=ss.groupby('seed').size(); complete_seeds=complete_counts[complete_counts==len(EXPECTED_DATASETS)*len(EXPECTED_METHODS)*len(EXPECTED_B)].index.tolist()
    pd.DataFrame({'seed':complete_counts.index,'complete_block_count':complete_counts.values,'is_complete_seed':[s in complete_seeds for s in complete_counts.index]}).to_csv(out/'seed_coverage.csv',index=False)
    avg_all=avg_over_seed_summaries(ss,["dataset","method","config_name","max_num_drafts"]); avg_all.to_csv(out/'summary_average_over_available_seeds_by_dataset_config.csv',index=False)
    strict3=avg_all[avg_all.seed_count==3].copy(); strict3.to_csv(out/'summary_strict_three_seed_blocks_only.csv',index=False)
    ss_bal=ss[ss.seed.isin(complete_seeds)].copy(); ss_bal.to_csv(out/'summary_by_seed_balanced_complete_seeds_only.csv',index=False)
    avg_bal=avg_over_seed_summaries(ss_bal,["dataset","method","config_name","max_num_drafts"]); avg_bal.to_csv(out/'summary_balanced_complete_seeds_by_dataset_config.csv',index=False)
    overall_bal=method_B_overall(ss_bal); overall_bal.to_csv(out/'paper_table_balanced_complete_seeds_overall_by_method_B.csv',index=False)
    rel_bal=rel_vs_baseline(overall_bal); rel_bal.to_csv(out/'comparison_balanced_vs_ashish_invariant_overall_by_B.csv',index=False)
    overall_avail=method_B_overall(ss); overall_avail.to_csv(out/'paper_table_available_seeds_overall_by_method_B.csv',index=False)
    rel_avail=rel_vs_baseline(overall_avail); rel_avail.to_csv(out/'comparison_available_vs_ashish_invariant_overall_by_B.csv',index=False)
    plot_overall(out,overall_bal,'balanced')
    plot_overall(out,overall_avail,'available')
    lines=[]; lines.append('# Three-method experiment cleaning report\n')
    lines.append(f'Target complete block size: `{args.target_prompts}` prompts. Expected full run per seed: `{len(EXPECTED_DATASETS)*len(EXPECTED_METHODS)*len(EXPECTED_B)}` dataset/config blocks.\n')
    lines.append('## File classification\n')
    for _,r in manifest.iterrows(): lines.append(f"- `{r.file}`: kind={r.kind}, rows={r.rows}, seed_values={r.get('seed_values','')}, complete_blocks_at_150={r.get('complete_blocks_at_150','')}, partial_blocks_at_150={r.get('partial_blocks_at_150','')}.")
    lines.append('\n## Seed coverage\n')
    for seed,n in complete_counts.items(): lines.append(f'- seed `{seed}`: {int(n)}/36 complete blocks selected' + (' (complete)' if seed in complete_seeds else ' (incomplete)').replace(' (complete)',' (complete seed)'))
    lines.append('\n## Critical note\n')
    lines.append('The uploaded files contain two complete seeds (`2155929800` and `929093658`). Seed `517798609` has only 16 complete dataset/config blocks after stitching its resume file, so a balanced three-seed average over all methods/datasets is not available from the uploaded files. The reliable paper-style aggregate over all three datasets/configs is therefore the balanced complete-seed table using seeds `2155929800` and `929093658`. The script also writes available-seed and strict-three-seed tables separately.\n')
    lines.append('## Balanced complete-seed overall table\n')
    cols=['method','max_num_drafts','seed_count','token_rate_mean','aatps_mean','be_mean','acceptance_fraction_mean','target_forward_calls_per_token_mean','draft_forward_calls_per_token_mean']
    lines.append(overall_bal[[c for c in cols if c in overall_bal.columns]].to_markdown(index=False, floatfmt='.4f'))
    lines.append('\n## Balanced relative comparison versus ashish_invariant\n')
    cols2=['method','max_num_drafts','token_rate_rel_pct','aatps_rel_pct','be_rel_pct','acceptance_fraction_rel_pct']
    lines.append(rel_bal[[c for c in cols2 if c in rel_bal.columns]].to_markdown(index=False, floatfmt='.3f'))
    (out/'clean_report.md').write_text('\n'.join(lines))
    print(f'Wrote {out}')
    print('Complete seeds:', complete_seeds)
    print('Seed block counts:'); print(complete_counts.to_string())

if __name__=='__main__': main()
