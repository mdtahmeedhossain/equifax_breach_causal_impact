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
print("EQUIFAX DATA BREACH: CAUSAL IMPACT ANALYSIS")
print("="*60)

# Step 1: Fetch data
print("\n[1/6] Fetching stock data...")
df = fetch_stock_data(TICKERS, START_DATE, END_DATE)
print(f"✓ Fetched {len(df)} records for {len(TICKERS)} tickers")

# Step 2: Prepare data (with post-treatment period)
print("\n[2/6] Preparing analysis data...")
df_full = prepare_analysis_data(df, EVENT_DATE, TREATED_UNIT, WINDOW_DAYS, WINDOW_DAYS_AFTER)
print(f"✓ Full analysis window: {df_full['date'].min()} to {df_full['date'].max()}")
print(f"✓ Pre-treatment observations: {(df_full['post']==0).sum()}")
print(f"✓ Post-treatment observations: {(df_full['post']==1).sum()}")

# Also prepare immediate-only data (just Sept 8)
df_immediate = prepare_analysis_data(df, EVENT_DATE, TREATED_UNIT, WINDOW_DAYS, window_days_after=0)
print(f"\n✓ Immediate impact window: {df_immediate['date'].min()} to {df_immediate['date'].max()}")
print(f"✓ Immediate post-treatment observations: {(df_immediate['post']==1).sum()}")

# Step 3: Run Difference-in-Differences (BOTH versions)
print("\n[3/6] Running Difference-in-Differences analysis...")

# 3a. Immediate impact (Sept 8 only)
print("\n  [A] IMMEDIATE IMPACT (Sept 8 only):")
did_immediate = DifferenceInDifferences(df_immediate)
did_immediate_results = did_immediate.estimate()

print(f"  Treatment Effect: {did_immediate_results['treatment_effect']:.4f} ({did_immediate_results['treatment_effect']*100:.2f}%)")
print(f"  Std Error: {did_immediate_results['std_error']:.4f}")
print(f"  P-value: {did_immediate_results['p_value']:.4f}")
print(f"  95% CI: [{did_immediate_results['conf_int'][0]:.4f}, {did_immediate_results['conf_int'][1]:.4f}]")

# 3b. Sustained average effect (3 weeks)
print("\n  [B] SUSTAINED AVERAGE EFFECT (3 weeks):")
did_sustained = DifferenceInDifferences(df_full)
did_sustained_results = did_sustained.estimate()

print(f"  Treatment Effect: {did_sustained_results['treatment_effect']:.4f} ({did_sustained_results['treatment_effect']*100:.2f}%)")
print(f"  Std Error: {did_sustained_results['std_error']:.4f}")
print(f"  P-value: {did_sustained_results['p_value']:.4f}")
print(f"  95% CI: [{did_sustained_results['conf_int'][0]:.4f}, {did_sustained_results['conf_int'][1]:.4f}]")

# Test parallel trends (using immediate data for consistency)
parallel_trends = did_immediate.test_parallel_trends()
print(f"\n  Parallel Trends Test:")
print(f"  Coefficient: {parallel_trends['coefficient']:.6f}")
print(f"  P-value: {parallel_trends['p_value']:.4f}")
print(f"  Result: {parallel_trends['interpretation']}")

# Step 4: Run Synthetic Control Method (BOTH versions)
print("\n[4/6] Running Synthetic Control Method...")

# 4a. Immediate impact
print("\n  [A] IMMEDIATE IMPACT (Sept 8 only):")
scm_immediate = SyntheticControl(df_immediate, TREATED_UNIT)
scm_immediate_results = scm_immediate.estimate()

print(f"  Treatment Effect: {scm_immediate_results['treatment_effect']:.4f} ({scm_immediate_results['treatment_effect']*100:.2f}%)")
print(f"  Pre-treatment RMSE: {scm_immediate_results['pre_treatment_rmse']:.6f}")

# 4b. Sustained average effect
print("\n  [B] SUSTAINED AVERAGE EFFECT (3 weeks):")
scm_sustained = SyntheticControl(df_full, TREATED_UNIT)
scm_sustained_results = scm_sustained.estimate()

print(f"  Treatment Effect: {scm_sustained_results['treatment_effect']:.4f} ({scm_sustained_results['treatment_effect']*100:.2f}%)")
print(f"  Pre-treatment RMSE: {scm_sustained_results['pre_treatment_rmse']:.6f}")
print(f"  Number of control units: {scm_sustained_results['n_control_units']}")

print(f"\n  Donor Weights:")
for unit, weight in sorted(scm_sustained_results['weights'].items(), key=lambda x: x[1], reverse=True):
    if weight > 0.01:  # Only show weights > 1%
        print(f"    {unit}: {weight:.3f}")

# Step 5: Run placebo tests
print("\n[5/6] Running placebo tests (this may take a minute)...")

# Get pre-treatment data only
pre_data = df_full[df_full['post'] == 0].copy()
event_datetime = pd.to_datetime(EVENT_DATE).tz_localize('America/New_York')

# Run 100 placebo tests for each method
n_placebo = 100
placebo_days = np.linspace(30, 150, n_placebo).astype(int)

did_placebo_effects = []
scm_placebo_effects = []

print(f"  Running {n_placebo} placebo tests per method...")

for i, days in enumerate(placebo_days):
    if (i+1) % 20 == 0:
        print(f"    Progress: {i+1}/{n_placebo}")
    
    try:
        # DiD placebo
        did_effect = run_placebo_test(pre_data, 'did', event_datetime, TREATED_UNIT, days)
        did_placebo_effects.append(did_effect)
        
        # SCM placebo
        scm_effect = run_placebo_test(pre_data, 'scm', event_datetime, TREATED_UNIT, days)
        scm_placebo_effects.append(scm_effect)
    except Exception as e:
        print(f"    Warning: Placebo test at {days} days failed: {e}")
        continue

# Calculate RMSE for placebo tests
did_placebo_rmse = np.sqrt(np.mean(np.array(did_placebo_effects)**2))
scm_placebo_rmse = np.sqrt(np.mean(np.array(scm_placebo_effects)**2))

print(f"\n  Placebo Test Results:")
print(f"  DiD Placebo RMSE: {did_placebo_rmse:.6f}")
print(f"  SCM Placebo RMSE: {scm_placebo_rmse:.6f}")
print(f"  Ratio (DiD/SCM): {did_placebo_rmse/scm_placebo_rmse:.2f}x")

if scm_placebo_rmse < did_placebo_rmse:
    print(f"  → SCM shows {(1 - scm_placebo_rmse/did_placebo_rmse)*100:.1f}% tighter placebo distribution")
    print(f"  → Recommendation: SCM is the preferred method")
else:
    print(f"  → DiD shows {(1 - did_placebo_rmse/scm_placebo_rmse)*100:.1f}% tighter placebo distribution")
    print(f"  → Recommendation: DiD is the preferred method")

# Step 6: Save results
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

print("\n✓ Analysis complete! Results saved to data/")
print("\nSUMMARY:")
print(f"  Immediate Impact (Sept 8): DiD={did_immediate_results['treatment_effect']*100:.2f}%, SCM={scm_immediate_results['treatment_effect']*100:.2f}%")
print(f"  Sustained Effect (3 weeks): DiD={did_sustained_results['treatment_effect']*100:.2f}%, SCM={scm_sustained_results['treatment_effect']*100:.2f}%")
print(f"  Interpretation: {abs(did_immediate_results['treatment_effect']/did_sustained_results['treatment_effect']):.1f}x larger immediate shock than sustained average")
print("="*60)
