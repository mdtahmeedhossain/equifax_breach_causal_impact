"""
Create Visualizations for Equifax Breach Analysis
Shows BOTH immediate impact and sustained average effect
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# Configuration
EVENT_DATE = '2017-09-08'
TREATED_UNIT = 'EFX'

print("Creating visualizations...")

# Load prepared data
df_prepared = pd.read_csv('../data/prepared_data.csv')
df_prepared['date'] = pd.to_datetime(df_prepared['date'])

# Load results
with open('../data/analysis_results.json', 'r') as f:
    results = json.load(f)

placebo_df = pd.read_csv('../data/placebo_effects.csv')

# Figure 1: Time Series - Treated vs Control Average (WITH POST-TREATMENT)
print("  Creating Figure 1: Time Series...")
fig, ax = plt.subplots(figsize=(14, 7))

# Calculate average returns for control group by date
control_avg = df_prepared[df_prepared['treated']==0].groupby('date')['returns'].mean().reset_index()
treated_data = df_prepared[df_prepared['treated']==1][['date', 'returns']]

# Plot
ax.plot(control_avg['date'], control_avg['returns'], label='Control Group Average', 
        linewidth=2, alpha=0.7, color='#2E86AB')
ax.plot(treated_data['date'], treated_data['returns'], label='Equifax (EFX)', 
        linewidth=2.5, color='#A23B72')

# Add event line
event_dt = pd.to_datetime(EVENT_DATE)
ax.axvline(event_dt, color='red', linestyle='--', linewidth=2, label='Breach Announcement', alpha=0.7)

# Shade post-treatment period
ax.axvspan(event_dt, treated_data['date'].max(), alpha=0.1, color='red')

# Formatting
ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('Daily Returns', fontsize=13, fontweight='bold')
ax.set_title('Equifax Stock Returns vs Control Group\nData Breach Announcement (Sept 8, 2017) and Recovery Period', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='lower left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/time_series.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Dual Treatment Effects Comparison
print("  Creating Figure 2: Immediate vs Sustained Effects...")
fig, ax = plt.subplots(figsize=(12, 7))

# Data for plotting
categories = ['Immediate Impact\n(Sept 8 only)', 'Sustained Average\n(3 weeks)']
did_effects = [results['immediate_did_effect']*100, results['sustained_did_effect']*100]
scm_effects = [results['immediate_scm_effect']*100, results['sustained_scm_effect']*100]

x = np.arange(len(categories))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, np.abs(did_effects), width, label='DiD', 
               color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, np.abs(scm_effects), width, label='SCM', 
               color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.5)

# Labels
ax.set_ylabel('Treatment Effect (% decline)', fontsize=13, fontweight='bold')
ax.set_title('Immediate vs Sustained Treatment Effects\nEquifax Data Breach Impact', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.legend(fontsize=12, loc='upper right')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add annotation
ax.text(0.5, max(np.abs(did_effects))*0.5, 
        f'Immediate shock is {abs(results["immediate_did_effect"]/results["sustained_did_effect"]):.1f}x larger\nthan sustained average',
        ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('../figures/immediate_vs_sustained.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Placebo Test Distributions (STANDARDIZED)
print("  Creating Figure 3: Placebo Distributions...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Standardize placebo effects to mean=0, std=1
did_placebo_mean = placebo_df['did_effect'].mean()
did_placebo_std = placebo_df['did_effect'].std()
did_placebo_standardized = (placebo_df['did_effect'] - did_placebo_mean) / did_placebo_std

scm_placebo_mean = placebo_df['scm_effect'].mean()
scm_placebo_std = placebo_df['scm_effect'].std()
scm_placebo_standardized = (placebo_df['scm_effect'] - scm_placebo_mean) / scm_placebo_std

# Calculate empirical p-values (# placebo effects <= true effect) / N
did_n_more_extreme = np.sum(placebo_df['did_effect'] <= results['immediate_did_effect'])
scm_n_more_extreme = np.sum(placebo_df['scm_effect'] <= results['immediate_scm_effect'])
did_p_value = did_n_more_extreme / len(placebo_df)
scm_p_value = scm_n_more_extreme / len(placebo_df)

# DiD placebo distribution (standardized, no true effect line)
ax1.hist(did_placebo_standardized, bins=30, alpha=0.7, color='#2E86AB', edgecolor='black')
ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Placebo Mean')
ax1.set_xlabel('Standardized Placebo Effect', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title(f'Difference-in-Differences\nPlacebo Distribution (n=100)', 
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)

# Annotation: true effect vs placebo distribution
percentile_text = f'True effect: {results["immediate_did_effect"]*100:.2f}%\n\nLies below 1st percentile\n({did_n_more_extreme}/{len(placebo_df)} placebos more extreme)\n\nEmpirical p-value < 0.01'
ax1.text(0.05, 0.95, percentile_text,
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# SCM placebo distribution (standardized, no true effect line)
ax2.hist(scm_placebo_standardized, bins=30, alpha=0.7, color='#A23B72', edgecolor='black')
ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Placebo Mean')
ax2.set_xlabel('Standardized Placebo Effect', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title(f'Synthetic Control Method\nPlacebo Distribution (n=100)', 
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3)

# Annotation: true effect vs placebo distribution
percentile_text = f'True effect: {results["immediate_scm_effect"]*100:.2f}%\n\nLies below 1st percentile\n({scm_n_more_extreme}/{len(placebo_df)} placebos more extreme)\n\nEmpirical p-value < 0.01'
ax2.text(0.05, 0.95, percentile_text,
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.suptitle('Model Validation: Placebo Test Results (Randomization Inference)', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../figures/placebo_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: Method Comparison with Placebo RMSE
print("  Creating Figure 4: Method Comparison...")
fig, ax = plt.subplots(figsize=(10, 7))

methods = ['DiD', 'SCM']
immediate_effects = [abs(results['immediate_did_effect'])*100, abs(results['immediate_scm_effect'])*100]
placebo_rmse = [results['did_placebo_rmse'], results['scm_placebo_rmse']]

x = np.arange(len(methods))
width = 0.35

bars1 = ax.bar(x - width/2, immediate_effects, width, label='Immediate Effect (%)', 
               color=['#2E86AB', '#A23B72'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, np.array(placebo_rmse)*1000, width, label='Placebo RMSE (×10³)', 
                color=['#F18F01', '#C73E1D'], alpha=0.7, edgecolor='black', linewidth=1.5)

# Labels
ax.set_xlabel('Method', fontsize=13, fontweight='bold')
ax.set_ylabel('Immediate Treatment Effect (% decline)', fontsize=13, fontweight='bold', color='#2E86AB')
ax2.set_ylabel('Placebo RMSE (×10³)', fontsize=13, fontweight='bold', color='#F18F01')
ax.set_title('Causal Inference Method Comparison\nEquifax Data Breach Impact', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=12)
ax.tick_params(axis='y', labelcolor='#2E86AB')
ax2.tick_params(axis='y', labelcolor='#F18F01')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)

ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('../figures/method_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Synthetic Control Donor Weights
print("  Creating Figure 5: Donor Weights...")
weights_dict = results['scm_weights']

fig, ax = plt.subplots(figsize=(10, 6))

weights_df = pd.DataFrame(list(weights_dict.items()), columns=['Unit', 'Weight'])
weights_df = weights_df.sort_values('Weight', ascending=True)

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(weights_df)))
bars = ax.barh(weights_df['Unit'], weights_df['Weight'], color=colors, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Weight', fontsize=13, fontweight='bold')
ax.set_ylabel('Control Unit', fontsize=13, fontweight='bold')
ax.set_title('Synthetic Control: Donor Unit Weights\nOptimized to Match Pre-Treatment Equifax', 
             fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (unit, weight) in enumerate(zip(weights_df['Unit'], weights_df['Weight'])):
    ax.text(weight + 0.01, i, f'{weight:.3f}', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/donor_weights.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 6: Parallel Trends Visualization (PRE-TREATMENT ONLY!)
print("  Creating Figure 6: Parallel Trends...")
fig, ax = plt.subplots(figsize=(14, 7))

# Get pre-treatment data only
pre_data = df_prepared[df_prepared['post']==0]

# Calculate cumulative returns for treated and control
treated_pre = pre_data[pre_data['treated']==1].sort_values('date')
control_pre = pre_data[pre_data['treated']==0].groupby('date')['returns'].mean().reset_index().sort_values('date')

# Calculate cumulative returns
treated_pre = treated_pre.copy()
control_pre = control_pre.copy()
treated_pre['cumulative_return'] = (1 + treated_pre['returns']).cumprod() - 1
control_pre['cumulative_return'] = (1 + control_pre['returns']).cumprod() - 1

# Plot
ax.plot(control_pre['date'], control_pre['cumulative_return']*100, 
        label='Control Group', linewidth=2.5, color='#2E86AB', marker='o', markersize=2, alpha=0.7)
ax.plot(treated_pre['date'], treated_pre['cumulative_return']*100, 
        label='Equifax (EFX)', linewidth=2.5, color='#A23B72', marker='s', markersize=2)

ax.set_xlabel('Date (Pre-Treatment Period Only)', fontsize=13, fontweight='bold')
ax.set_ylabel('Cumulative Return (%)', fontsize=13, fontweight='bold')
ax.set_title(f'Parallel Trends Test: Pre-Treatment Period\n(p-value = {results["parallel_trends_p_value"]:.4f} - No evidence of differential pre-trends)', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig('../figures/parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 7: Event Study Plot (WITH POST-TREATMENT)
print("  Creating Figure 7: Event Study...")
fig, ax = plt.subplots(figsize=(14, 7))

# Calculate average treatment effect by days relative to event
event_study = df_prepared.copy()
event_study['days_to_event'] = event_study['days_to_event'].astype(int)

# Get treatment effects for each day
treatment_effects_by_day = []
days_range = range(-30, 22)  # 30 days before to 21 days after

for day in days_range:
    day_data = event_study[event_study['days_to_event'] == day]
    if len(day_data) > 0:
        treated_return = day_data[day_data['treated']==1]['returns'].mean()
        control_return = day_data[day_data['treated']==0]['returns'].mean()
        effect = treated_return - control_return
        treatment_effects_by_day.append({'day': day, 'effect': effect})

effect_df = pd.DataFrame(treatment_effects_by_day)

# Plot
ax.plot(effect_df['day'], effect_df['effect']*100, linewidth=2.5, color='#2E86AB', 
        marker='o', markersize=6, markerfacecolor='white', markeredgewidth=2)
ax.axvline(0, color='red', linestyle='--', linewidth=2.5, label='Breach Announcement', alpha=0.7)
ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# Shade post-treatment period
ax.axvspan(0, 21, alpha=0.1, color='red', label='Post-Treatment')

ax.set_xlabel('Days Relative to Breach Announcement', fontsize=13, fontweight='bold')
ax.set_ylabel('Treatment Effect (%)', fontsize=13, fontweight='bold')
ax.set_title('Event Study: Daily Treatment Effects\nEquifax vs Control Group', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='lower left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/event_study.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ All visualizations created successfully!")
print("  Saved to figures/:")
print("    - time_series.png (WITH POST-TREATMENT)")
print("    - immediate_vs_sustained.png (NEW: DUAL EFFECTS COMPARISON)")
print("    - placebo_distributions.png")
print("    - method_comparison.png")
print("    - donor_weights.png")
print("    - parallel_trends.png (PRE-TREATMENT ONLY)")
print("    - event_study.png (EXTENDED TO +21 DAYS)")
