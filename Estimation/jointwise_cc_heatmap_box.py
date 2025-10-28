#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-joint CC heatmap / boxplot generator (图3)
---------------------------------------------
- Reads the long-form CSV exported by eval_jointwise_models.py
  (columns: Subject, Method, Joint, NRMSE, CC, R2)
- Outputs:
  (a) Heatmap of mean CC per (Method × Joint)
  (b) Boxplot of CC distribution per joint for a chosen method
"""
import os, argparse, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def heatmap_cc(df: pd.DataFrame, out_path: str):
    data = df[df['Joint'].apply(lambda x: str(x).isdigit())].copy()
    data['Joint'] = data['Joint'].astype(int)
    pivot = data.pivot_table(index='Method', columns='Joint', values='CC', aggfunc='mean').sort_index(axis=1)
    fig = plt.figure(figsize=(10, 3 + 0.4*len(pivot.index)))
    ax = plt.gca()
    im = ax.imshow(pivot.values, aspect='auto')
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_xlabel('Joint'); ax.set_ylabel('Method')
    ax.set_title('Mean CC per Joint (Heatmap)')
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha='center', va='center')
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    plt.tight_layout(); plt.savefig(out_path, dpi=300)

def boxplot_cc(df: pd.DataFrame, method: str, out_path: str):
    data = df[(df['Method']==method) & (df['Joint'].apply(lambda x: str(x).isdigit()))].copy()
    data['Joint'] = data['Joint'].astype(int)
    joints = sorted(data['Joint'].unique())
    cc_lists = [data[data['Joint']==j]['CC'].values for j in joints]
    plt.figure(figsize=(10, 4))
    plt.boxplot(cc_lists, labels=joints, showmeans=True)
    plt.xlabel('Joint'); plt.ylabel('CC (↑)')
    plt.title(f'Per-Joint CC Distribution — {method}')
    plt.tight_layout(); plt.savefig(out_path, dpi=300)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, required=True)
    ap.add_argument('--out_dir', type=str, default='/mnt/data/jointplots')
    ap.add_argument('--box_method', type=str, default='cdanrpp')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)
    heat_path = os.path.join(args.out_dir, 'cc_heatmap.png')
    box_path  = os.path.join(args.out_dir, f'cc_box_{args.box_method}.png')
    heatmap_cc(df, heat_path)
    boxplot_cc(df, args.box_method, box_path)
    print(f"[Done] Heatmap: {heat_path}")
    print(f"[Done] Boxplot: {box_path}")

if __name__ == '__main__':
    main()
