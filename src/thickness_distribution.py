#!/usr/bin/env python3
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import resize

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',  required=True)
    p.add_argument('--output_dir', default='./results')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # collect per‐subject means
    subj = {}
    for fname in os.listdir(args.data_dir):
        if not fname.endswith('.npz'):
            continue
        path = os.path.join(args.data_dir, fname)
        data = np.load(path, allow_pickle=True)

        # pid fallback
        if 'pid' in data.files:
            pid = data['pid'].item().split('_')[0]
        else:
            pid = os.path.splitext(fname)[0]

        # must have race + rnflt
        if ('race' not in data.files) or ('rnflt' not in data.files):
            continue

        race = data['race'].item()
        rnflt = data['rnflt']

        # resize to 224×224 if needed
        if rnflt.shape != (224,224):
            rnflt = resize(rnflt, (224,224), preserve_range=True)

        # clip & drop zeros
        rnflt = np.clip(rnflt, 0, 350)
        vals = rnflt[rnflt > 0]
        if vals.size == 0:
            continue

        subj.setdefault(pid, {'race': race, 'means': []})
        subj[pid]['means'].append(vals.mean())

    if not subj:
        print(f"❌ No subjects found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Found {len(subj)} subjects")

    # build DataFrame
    rows = []
    for pid, info in subj.items():
        rows.append({
            'pid': pid,
            'race': info['race'],
            'rnflt_mean': np.mean(info['means'])
        })
    df = pd.DataFrame(rows)

    # order exactly as paper
    order = ['Black or African American',
             'White or Caucasian',
             'Asian']
    df['race'] = pd.Categorical(df['race'], categories=order, ordered=True)

    # save CSV
    out_csv = os.path.join(args.output_dir, 'rnflt_by_race.csv')
    df.to_csv(out_csv, index=False)
    print("✅ CSV saved to:", out_csv)

    # plot box + jitter
    plt.figure(figsize=(10,6))
    sns.boxplot(
        data=df, x='race', y='rnflt_mean',
        order=order, palette=['#555','#888','#ccc'],
        showcaps=True, boxprops={'linewidth':2},
        medianprops={'color':'grey','linewidth':2},
        whiskerprops={'linestyle':'-','linewidth':1},
        flierprops={'marker':'o','markersize':4,'alpha':0.5}
    )
    sns.stripplot(
        data=df, x='race', y='rnflt_mean',
        order=order, jitter=0.2, size=3, alpha=0.5, color='black'
    )

    plt.title('Distribution of Mean RNFLT by Race (Per Subject)', fontsize=16)
    plt.xlabel('Race', fontsize=14)
    plt.ylabel('Mean RNFLT (μm)', fontsize=14)

    # annotate counts below
    counts = df['race'].value_counts().reindex(order)
    y_min, y_max = df['rnflt_mean'].min(), df['rnflt_mean'].max()
    for i, r in enumerate(order):
        plt.text(i, y_min - (y_max-y_min)*0.05,
                 f"n={counts[r]}", ha='center', va='top', fontsize=12)

    plt.ylim(y_min - (y_max-y_min)*0.1, y_max - 100)
    plt.tight_layout()

    out_png = os.path.join(args.output_dir, 'rnflt_by_race.png')
    plt.savefig(out_png, dpi=300)
    print("✅ Plot saved to:", out_png)
    plt.show()

if __name__ == '__main__':
    main()
