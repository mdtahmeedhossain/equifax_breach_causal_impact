"""
Main Analysis Script for Equifax Breach Impact
Runs DiD, SCM, and placebo tests
Calculates BOTH immediate impact and sustained average effect
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys

from data_collection import fetch_stock_data, prepare_analysis_data
from causal_methods import DifferenceInDifferences, SyntheticControl, run_placebo_test

# Configuration
TICKERS = ['EFX', 'MCO', 'TRU', 'SPY', 'VTI', 'EXPGY', 'BAH']
START_DATE = '2017-01-01'
END_DATE = '2017-09-30'
EVENT_DATE = '2017-09-08'
TREATED_UNIT = 'EFX'
WINDOW_DAYS = 180
WINDOW_DAYS_AFTER = 21  # 3 weeks post-treatment

print("="*60)
print("EQUIFAX BREACH CAUSAL ANALYSIS")
print("="*60)

print("\n[1/6] Fetching data...")
df = fetch_stock_data(TICKERS, START_DATE, END_DATE)
print(f"Got {len(df)} records")

print("\n[2/6] Preparing data...")
df_full = prepare_analysis_data(df, EVENT_DATE, TREATED_UNIT, WINDOW_DAYS, WINDOW_DAYS_AFTER)
print(f"Window: {df_full['date'].min().date()} to {df_full['date'].max().date()}")
print(f"Pre: {(df_full['post']==0).sum()}, Post: {(df_full['post']==1).sum()}")

df_immediate = prepare_analysis_data(df, EVENT_DATE, TREATED_UNIT, WINDOW_DAYS, window_days_after=0)
print(f"Immediate window: {df_immediate['date'].min().date()} to {df_immediate['date'].max().date()}")

print("\n[3/6] Running DiD...")

print("\n  Immediate (Sept 8):")
did_immediate = DifferenceInDifferences(df_immediate)
did_immediate_results = did_immediate.estimate()
print(f"  Effect: {did_immediate_results['treatment_effect']*100:.2f}% (p={did_immediate_results['p_value']:.4f})")
print(f"  CI: [{did_immediate_results['conf_int'][0]*100:.2f}%, {did_immediate_results['conf_int'][1]*100:.2f}%]")

print("\n  Sustained (3 weeks):")
did_sustained = DifferenceInDifferences(df_full)
did_sustained_results = did_sustained.estimate()
print(f"  Effect: {did_sustained_results['treatment_effect']*100:.2f}% (p={did_sustained_results['p_value']:.4f})")
print(f"  CI: [{did_sustained_results['conf_int'][0]*100:.2f}%, {did_sustained_results['conf_int'][1]*100:.2f}%]")

parallel_trends = did_immediate.test_parallel_trends()
print(f"\n  Parallel trends: p={parallel_trends['p_value']:.4f}")

print("\n[4/6] Running SCM...")

print("\n  Immediate:")
scm_immediate = SyntheticControl(df_immediate, TREATED_UNIT)
scm_immediate_results = scm_immediate.estimate()
print(f"  Effect: {scm_immediate_results['treatment_effect']*100:.2f}% (RMSE: {scm_immediate_results['pre_treatment_rmse']:.6f})")

print("\n  Sustained:")
scm_sustained = SyntheticControl(df_full, TREATED_UNIT)
scm_sustained_results = scm_sustained.estimate()
print(f"  Effect: {scm_sustained_results['treatment_effect']*100:.2f}% (RMSE: {scm_sustained_results['pre_treatment_rmse']:.6f})")

print(f"\n  Weights:")
for unit, weight in sorted(scm_sustained_results['weights'].items(), key=lambda x: x[1], reverse=True):
    if weight > 0.01:
        print(f"    {unit}: {weight:.1%}")

print("\n[5/6] Running placebo tests...")

pre_data = df_full[df_full['post'] == 0].copy()
event_datetime = pd.to_datetime(EVENT_DATE).tz_localize('America/New_York')

n_placebo = 100
placebo_days = np.linspace(30, 150, n_placebo).astype(int)

did_placebo_effects = []
scm_placebo_effects = []

for i, days in enumerate(placebo_days):
    if (i+1) % 25 == 0:
        print(f"  {i+1}/{n_placebo}")

    try:
        did_effect = run_placebo_test(pre_data, 'did', event_datetime, TREATED_UNIT, days)
        did_placebo_effects.append(did_effect)

        scm_effect = run_placebo_test(pre_data, 'scm', event_datetime, TREATED_UNIT, days)
        scm_placebo_effects.append(scm_effect)
    except Exception as e:
        continue

did_placebo_rmse = np.sqrt(np.mean(np.array(did_placebo_effects)**2))
scm_placebo_rmse = np.sqrt(np.mean(np.array(scm_placebo_effects)**2))

print(f"\n  Placebo RMSE: DiD={did_placebo_rmse:.6f}, SCM={scm_placebo_rmse:.6f}")
if scm_placebo_rmse < did_placebo_rmse:
    print(f"  SCM preferred ({(1 - scm_placebo_rmse/did_placebo_rmse)*100:.1f}% tighter)")

print("\n[6/6] Saving results...")

results_summary = {
    # Immediate impact (Sept 8 only)
    'immediate_did_effect': did_immediate_results['treatment_effect'],
    'immediate_did_p_value': did_immediate_results['p_value'],
    'immediate_did_std_error': did_immediate_results['std_error'],
    'immediate_did_conf_int': did_immediate_results['conf_int'],
    'immediate_scm_effect': scm_immediate_results['treatment_effect'],
    'immediate_scm_pre_rmse': scm_immediate_results['pre_treatment_rmse'],
    
    # Sustained average effect (3 weeks)
    'sustained_did_effect': did_sustained_results['treatment_effect'],
    'sustained_did_p_value': did_sustained_results['p_value'],
    'sustained_did_std_error': did_sustained_results['std_error'],
    'sustained_did_conf_int': did_sustained_results['conf_int'],
    'sustained_scm_effect': scm_sustained_results['treatment_effect'],
    'sustained_scm_pre_rmse': scm_sustained_results['pre_treatment_rmse'],
    
    # Model validation
    'scm_weights': scm_sustained_results['weights'],
    'did_placebo_rmse': did_placebo_rmse,
    'scm_placebo_rmse': scm_placebo_rmse,
    'parallel_trends_p_value': parallel_trends['p_value'],
    'parallel_trends_passes': bool(parallel_trends['passes'])
}

# Save to file
import json
with open('../data/analysis_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

# Save placebo effects for visualization
placebo_df = pd.DataFrame({
    'days_before': placebo_days[:len(did_placebo_effects)],
    'did_effect': did_placebo_effects,
    'scm_effect': scm_placebo_effects
})
placebo_df.to_csv('../data/placebo_effects.csv', index=False)

# Save full prepared data for visualizations
df_full.to_csv('../data/prepared_data.csv', index=False)

print("\nDone. Results saved to data/")
print(f"\nImmediate: DiD={did_immediate_results['treatment_effect']*100:.2f}%, SCM={scm_immediate_results['treatment_effect']*100:.2f}%")
print(f"Sustained: DiD={did_sustained_results['treatment_effect']*100:.2f}%, SCM={scm_sustained_results['treatment_effect']*100:.2f}%")
print(f"Immediate effect {abs(did_immediate_results['treatment_effect']/did_sustained_results['treatment_effect']):.1f}x larger")
print("="*60)
