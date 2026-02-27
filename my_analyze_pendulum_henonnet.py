import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def kinetic(p, l=1.0, m=1.0):
    return p**2 / (2*m*l**2)

def potential(q, g=9.81, l=1.0, m=1.0):
    return -m*g*l*np.cos(q)

def total_h(q, p, g=9.81, l=1.0, m=1.0):
    return kinetic(p, l, m) + potential(q, g, l, m)

os.makedirs('Results/Pendulum_HENONNET', exist_ok=True)

full_df = pd.read_hdf('Data/Pendulum_MLP/pendulum_full.h5', key='trajectories')
pred_df = pd.read_hdf('NeuralNets/Pendulum_HENONNET/henonnet_predictions.h5', key='preds')
qpr_df = pd.read_csv('Data/Pendulum_MLP/quarter_period.csv')

n_examples = 6
example_trajs = np.linspace(0, full_df['traj'].max(), n_examples, dtype=int)

# 1. Phase Space
plt.figure(figsize=(16, 8))
for k, traj in enumerate(example_trajs):
    ref = full_df[full_df['traj'] == traj]
    pred = pred_df[pred_df['traj'] == traj].sort_values('t')
    plt.axvspan(ref['q'].min(), ref['q'].max(), ymin=0.0, ymax=1.0,
                color='orange', alpha=0.13 * (1+k%2), label="Training region" if k == 0 else None)
    plt.plot(ref['q'], ref['p'], lw=2, label=f'Traj {traj} true' if k==0 else None, color=f'C{k}')
    plt.plot(pred['q_pred'], pred['p_pred'], '--', lw=2, label=f'Traj {traj} HenonNet' if k==0 else None, color=f'C{k}', alpha=0.7)
plt.xlabel("q (angle)")
plt.ylabel("p (momentum)")
plt.title("Phase Space: True vs. HenonNet Prediction\n(Shaded = Training region)")
plt.legend()
plt.savefig('Results/Pendulum_HENONNET/phase_space.png', dpi=150)
plt.close()

# 2. Energy Conservation Plot
for k, traj in enumerate(example_trajs):
    ref = full_df[full_df['traj'] == traj].sort_values('t')
    pred = pred_df[pred_df['traj'] == traj].sort_values('t')
    q_true = ref['q'].values
    p_true = ref['p'].values
    q_pred = pred['q_pred'].values
    p_pred = pred['p_pred'].values
    t = ref['t'].values
    t_quarter = qpr_df[qpr_df['traj'] == traj]['t'].values[0]
    plt.figure(figsize=(14,6))
    plt.plot(t, total_h(q_true, p_true), label="True H", color='C0')
    plt.plot(t, kinetic(p_true), label='True K', ls='--', color='C1')
    plt.plot(t, potential(q_true), label='True V', ls='-', color='C2')
    plt.plot(t, total_h(q_pred, p_pred), label="HenonNet H", color='C0', alpha=0.5)
    plt.plot(t, kinetic(p_pred), label='HenonNet K', ls='--', color='C1', alpha=0.5)
    plt.plot(t, potential(q_pred), label='HenonNet V', ls='-', color='C2', alpha=0.5)
    plt.axvline(t_quarter, color='k', ls=':', label='1/4 period (end of train)', lw=2)
    plt.title(f"Energy Conservation: Traj {traj} (HenonNet)")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.legend()
    plt.savefig(f'Results/Pendulum_HENONNET/energy_traj_{traj}.png', dpi=150)
    plt.close()

# 3. MAE vs. Time on Test Set
mae_by_time = []
for t_idx in range(full_df['t'].nunique()):
    t_this = full_df[full_df['traj']==0]['t'].values[t_idx]
    mask = (full_df['t'] == t_this) & (~full_df['train'])
    if not np.any(mask):
        continue
    true_q = full_df.loc[mask, 'q'].values
    true_p = full_df.loc[mask, 'p'].values
    pred_this = pred_df.loc[mask, :]
    pred_q = pred_this['q_pred'].values
    pred_p = pred_this['p_pred'].values
    mae = np.mean(np.abs(true_q - pred_q)) + np.mean(np.abs(true_p - pred_p))
    mae_by_time.append((t_this, mae/2))
if mae_by_time:
    mae_by_time = np.array(mae_by_time)
    plt.figure(figsize=(10,5))
    plt.plot(mae_by_time[:,0], mae_by_time[:,1], label="HenonNet MAE (test)")
    plt.axvline(qpr_df['t'].max(), color='k', ls=':', label='Train/test split (max 1/4 T)')
    plt.xlabel("Time (s)")
    plt.ylabel("Mean Abs Error (q,p)")
    plt.title("HenonNet Mean Absolute Error vs. Time (test region)")
    plt.legend()
    plt.savefig('Results/Pendulum_HENONNET/mae_vs_time.png', dpi=150)
    plt.close()

print("Phase space plot: Results/Pendulum_HENONNET/phase_space.png")
print(f"Energy plots: Results/Pendulum_HENONNET/energy_traj_[traj].png (example: {example_trajs.tolist()})")
print("MAE curve: Results/Pendulum_HENONNET/mae_vs_time.png")