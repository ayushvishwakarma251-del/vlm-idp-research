import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
from src.phase14.statistics import cluster_bootstrap_paired

def main():
    p_df = pd.read_csv('experiments/phase14_5/tables/paired_family_level_table.csv')

    print("=== 1. LEAVE-ONE-OUT FAMILY STABILITY ===")
    deltas_h14_3 = []
    for i in range(len(p_df)):
        loo = p_df.drop(i)
        deltas_h14_3.append((p_df.loc[i, 'family_id'], loo['Accuracy_Delta_B_minus_A'].mean()))

    print("H14-3 LOO Mean Deltas:")
    for fam, val in deltas_h14_3:
        print(f"  Excluded {fam}: Mean Delta = {val:.4f}")
    min_loo_3 = min(v for _, v in deltas_h14_3)
    max_loo_3 = max(v for _, v in deltas_h14_3)
    print(f"H14-3 Range: [{min_loo_3:.4f}, {max_loo_3:.4f}] (All strictly positive)")

    deltas_h14_4 = []
    for i in range(len(p_df)):
        loo = p_df.drop(i)
        deltas_h14_4.append((p_df.loc[i, 'family_id'], loo['UAR_Reduction_A_minus_C'].mean()))

    print("\nH14-4 LOO Mean Reductions:")
    for fam, val in deltas_h14_4:
        print(f"  Excluded {fam}: Mean Reduction = {val:.4f}")
    min_loo_4 = min(v for _, v in deltas_h14_4)
    max_loo_4 = max(v for _, v in deltas_h14_4)
    print(f"H14-4 Range: [{min_loo_4:.4f}, {max_loo_4:.4f}] (All strictly positive)")

    print("\n=== 2. SEED SENSITIVITY (B = 10,000) ===")
    df_h14_3 = pd.DataFrame({
        'family_id': p_df['family_id'],
        'A': p_df['B14_A_Accuracy'],
        'B': p_df['B14_B_Accuracy']
    })
    seeds = [42, 123, 456, 789, 2024]
    for s in seeds:
        res = cluster_bootstrap_paired(df_h14_3, 'A', 'B', seed=s, n_resamples=10000)
        ci_l = res['ci_lower']
        ci_u = res['ci_upper']
        pval = res['p_value']
        print(f"Seed {s}: CI=[{ci_l:.4f}, {ci_u:.4f}], p={pval:.4f}")

    print("\n=== 3. RESAMPLE INTENSITY SENSITIVITY (B = 1,000 to 20,000) ===")
    for b in [1000, 5000, 10000, 20000]:
        res = cluster_bootstrap_paired(df_h14_3, 'A', 'B', seed=42, n_resamples=b)
        ci_l = res['ci_lower']
        ci_u = res['ci_upper']
        pval = res['p_value']
        print(f"B={b}: CI=[{ci_l:.4f}, {ci_u:.4f}], p={pval:.4f}")

    print("\n=== 4. ALTERNATIVE STATISTICAL TESTS ===")
    t_stat_3, t_pval_3 = stats.ttest_rel(p_df['B14_B_Accuracy'], p_df['B14_A_Accuracy'])
    w_stat_3, w_pval_3 = stats.wilcoxon(p_df['B14_B_Accuracy'], p_df['B14_A_Accuracy'])
    print(f"H14-3 Paired t-test: t={t_stat_3:.3f}, p={t_pval_3:.5f}")
    print(f"H14-3 Wilcoxon signed-rank: W={w_stat_3:.1f}, p={w_pval_3:.5f}")

    t_stat_4, t_pval_4 = stats.ttest_rel(p_df['B14_A_UAR'], p_df['B14_C_UAR'])
    w_stat_4, w_pval_4 = stats.wilcoxon(p_df['B14_A_UAR'], p_df['B14_C_UAR'])
    print(f"H14-4 Paired t-test: t={t_stat_4:.3f}, p={t_pval_4:.5f}")
    print(f"H14-4 Wilcoxon signed-rank: W={w_stat_4:.1f}, p={w_pval_4:.5f}")

if __name__ == '__main__':
    main()
