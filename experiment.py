import numpy as np
from typing import List, Dict
import random
from scipy import stats
import os
import matplotlib.pyplot as plt
from agent import Agent
from game import TheMindGame

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()

class Experiment:
    def __init__(self, num_rounds: int = 80, games_per_round: int = 30):
        self.num_rounds = num_rounds
        self.games_per_round = games_per_round
        self.game = TheMindGame(cards_per_player=4)
        
    def create_agents(self, condition: str, use_tom: bool = True) -> List[Agent]:
        configs = {
            'similar':      {0: 1.1, 1: 0.9},
            'different':    {0: 1.8, 1: 0.5},
            'very_different':{0: 3.0, 1: 0.4}
        }
        cfg = configs.get(condition, configs['different'])
        
        return [
            Agent(0, wait_factor=cfg[0], learning_rate=0.05, use_tom=use_tom),
            Agent(1, wait_factor=cfg[1], learning_rate=0.05, use_tom=use_tom)
        ]
    
    def run_condition(self, condition: str, use_tom: bool, seed: int) -> Dict:
        random.seed(seed)
        np.random.seed(seed)
        
        agents = self.create_agents(condition, use_tom)
        
        results = {
            'condition': condition,
            'use_tom': use_tom,
            'round_scores': [],
            'tom_effort': [],
            'agent_states': {0: [], 1: []},
            'cg_rounds': {}
        }
        
        for round_num in range(self.num_rounds):
            round_scores = []
            round_mistakes = []
            
            for _ in range(self.games_per_round):
                result = self.game.play_game(agents)
                round_scores.append(result['success_rate'])
                round_mistakes.append(result['mistake_rate'])
                
            avg_score = np.mean(round_scores)
            avg_mistakes = np.mean(round_mistakes)
            
            results['round_scores'].append(avg_score)
            
            # Record per-round effort directly from agents
            round_effort = sum(a.tom_updates_this_round for a in agents)
            results['tom_effort'].append(round_effort)
            
            for agent in agents:
                agent.update_after_round(avg_score, avg_mistakes, round_num)
                results['agent_states'][agent.agent_id].append(agent.get_state().copy())
                
                if agent.cg_established and agent.agent_id not in results['cg_rounds']:
                    results['cg_rounds'][agent.agent_id] = round_num
                    
        return results

def run_full_experiment(n_runs: int = 40, num_rounds: int = 100):
    """
    Run complete experiment comparing ToM vs non-ToM agents.
    Args:
    n_runs: Number of full repetitions (for statistical power)
    num_rounds: Number of rounds per run (Lifespan/Time to learn)
    """
    print("="*70)
    print("THE MIND - COMMON GROUND EXPERIMENT")
    print(f"Configuration: {n_runs} Runs | {num_rounds} Rounds per Run")
    print("Testing H1 (ToM improves performance) and H2 (CG reduces effort)")
    print("="*70)
    
    all_results = {
        'tom': {'similar': [], 'different': [], 'very_different': []},
        'no_tom': {'similar': [], 'different': [], 'very_different': []}
    }
    
    # Pass num_rounds to Experiment class
    exp = Experiment(num_rounds=num_rounds, games_per_round=30)
    
    for run in range(n_runs):
        if (run + 1) % 5 == 0:
            print(f"Run {run + 1}/{n_runs}...")
            
        for ci, condition in enumerate(['similar', 'different', 'very_different']):
            # Deterministic seed: no hash() which varies across Python sessions
            base_seed = 10000 + run * 100 + ci * 10
            
            # ToM condition
            result_tom = exp.run_condition(condition, use_tom=True, seed=base_seed)
            all_results['tom'][condition].append(result_tom)
            
            # Non-ToM condition
            result_no_tom = exp.run_condition(condition, use_tom=False, seed=base_seed)
            all_results['no_tom'][condition].append(result_no_tom)        
    
    return all_results

def analyze_hypotheses(results: Dict):
    """
    Statistical analysis of H1 (ToM improves performance) and H2 (CG reduces effort).
    Uses paired t-tests (same seed per run) and Cohen's d effect sizes.
    """
    conditions = ['similar', 'different', 'very_different']
    
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)
    
    # =====================================================================
    # H1: ToM improves final performance
    # =====================================================================
    print("\n--- H1: ToM Improves Final Performance (Paired t-tests) ---\n")
    print(f"{'Condition':<18} {'ToM Mean':>10} {'NoToM Mean':>10} {'t-stat':>10} {'p-value':>12} {'Cohen d':>10} {'Sig':>6}")
    print("-" * 78)
    
    for cond in conditions:
        tom_final = [r['round_scores'][-1] for r in results['tom'][cond]]
        no_tom_final = [r['round_scores'][-1] for r in results['no_tom'][cond]]
        
        # Paired t-test (same seeds)
        t_stat, p_val = stats.ttest_rel(tom_final, no_tom_final)
        
        # Cohen's d for paired samples
        diff = np.array(tom_final) - np.array(no_tom_final)
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        
        print(f"{cond:<18} {np.mean(tom_final):>10.4f} {np.mean(no_tom_final):>10.4f} "
              f"{t_stat:>10.3f} {p_val:>12.6f} {cohens_d:>10.3f} {sig:>6}")
    
    # =====================================================================
    # H1b: ToM improves convergence speed
    # =====================================================================
    print("\n--- H1b: ToM Accelerates Strategy Convergence ---\n")
    print(f"{'Condition':<18} {'ToM GapR50':>12} {'NoToM GapR50':>12} {'t-stat':>10} {'p-value':>12}")
    print("-" * 66)
    
    for cond in conditions:
        tom_gaps_r50 = []
        no_tom_gaps_r50 = []
        
        for r in results['tom'][cond]:
            wf0 = r['agent_states'][0][min(49, len(r['agent_states'][0])-1)]['wait_factor']
            wf1 = r['agent_states'][1][min(49, len(r['agent_states'][1])-1)]['wait_factor']
            tom_gaps_r50.append(abs(wf0 - wf1))
        
        for r in results['no_tom'][cond]:
            wf0 = r['agent_states'][0][min(49, len(r['agent_states'][0])-1)]['wait_factor']
            wf1 = r['agent_states'][1][min(49, len(r['agent_states'][1])-1)]['wait_factor']
            no_tom_gaps_r50.append(abs(wf0 - wf1))
        
        t_stat, p_val = stats.ttest_rel(tom_gaps_r50, no_tom_gaps_r50)
        print(f"{cond:<18} {np.mean(tom_gaps_r50):>12.4f} {np.mean(no_tom_gaps_r50):>12.4f} "
              f"{t_stat:>10.3f} {p_val:>12.6f}")
    
    # =====================================================================
    # CG Establishment Rates
    # =====================================================================
    print("\n--- Common Ground Establishment Rates ---\n")
    print(f"{'Condition':<18} {'ToM CG%':>10} {'NoToM CG%':>10} {'ToM AvgRound':>14} {'NoToM AvgRound':>14}")
    print("-" * 68)
    
    for cond in conditions:
        tom_cg_count = sum(1 for r in results['tom'][cond] if len(r['cg_rounds']) == 2)
        no_tom_cg_count = sum(1 for r in results['no_tom'][cond] if len(r['cg_rounds']) == 2)
        n_runs = len(results['tom'][cond])
        
        tom_cg_rounds = [max(r['cg_rounds'].values()) for r in results['tom'][cond] if len(r['cg_rounds']) == 2]
        no_tom_cg_rounds = [max(r['cg_rounds'].values()) for r in results['no_tom'][cond] if len(r['cg_rounds']) == 2]
        
        tom_avg_r = f"{np.mean(tom_cg_rounds):.1f}" if tom_cg_rounds else "N/A"
        no_tom_avg_r = f"{np.mean(no_tom_cg_rounds):.1f}" if no_tom_cg_rounds else "N/A"
        
        print(f"{cond:<18} {tom_cg_count/n_runs*100:>9.1f}% {no_tom_cg_count/n_runs*100:>9.1f}% "
              f"{tom_avg_r:>14} {no_tom_avg_r:>14}")
    
    # =====================================================================
    # H2: Cognitive effort drops after CG
    # =====================================================================
    print("\n--- H2: Cognitive Effort Before vs After CG (ToM only) ---\n")
    print(f"{'Condition':<18} {'Pre-CG Effort':>14} {'Post-CG Effort':>14} {'Reduction%':>12} {'t-stat':>10} {'p-value':>12}")
    print("-" * 82)
    
    for cond in conditions:
        pre_efforts = []
        post_efforts = []
        
        for r in results['tom'][cond]:
            if len(r['cg_rounds']) == 2:
                cg_round = max(r['cg_rounds'].values())
                effort = r['tom_effort']
                
                # Average effort in 10 rounds before CG
                pre_start = max(0, cg_round - 10)
                pre = np.mean(effort[pre_start:cg_round]) if cg_round > 0 else 0
                
                # Average effort in 10 rounds after CG  
                post_end = min(len(effort), cg_round + 10)
                post = np.mean(effort[cg_round:post_end]) if cg_round < len(effort) else 0
                
                pre_efforts.append(pre)
                post_efforts.append(post)
        
        if len(pre_efforts) >= 2:
            t_stat, p_val = stats.ttest_rel(pre_efforts, post_efforts)
            reduction = (1 - np.mean(post_efforts) / np.mean(pre_efforts)) * 100 if np.mean(pre_efforts) > 0 else 0
            print(f"{cond:<18} {np.mean(pre_efforts):>14.2f} {np.mean(post_efforts):>14.2f} "
                  f"{reduction:>11.1f}% {t_stat:>10.3f} {p_val:>12.6f}")
        else:
            print(f"{cond:<18} {'Insufficient CG runs for test':>60}")
    
    print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant")

def create_publication_figures(results: Dict):
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    conditions = ['similar', 'different', 'very_different']
    cond_labels = ['Similar', 'Different', 'Very Different']
    
    TOM_COLOR = '#2E86AB'
    NO_TOM_COLOR = '#A23B72'
    CG_COLOR = '#FFB800'
    
    def _ci95(data_2d):
        """Compute mean and 95% CI band across axis 0."""
        mean = np.mean(data_2d, axis=0)
        se = np.std(data_2d, axis=0, ddof=1) / np.sqrt(data_2d.shape[0])
        return mean, mean - 1.96*se, mean + 1.96*se
    
    def _avg_cg_round(runs):
        """Average CG round across runs where both agents established CG."""
        cg = [max(r['cg_rounds'].values()) for r in runs if len(r['cg_rounds']) == 2]
        return np.mean(cg) if cg else None
    
    # =========================================================================
    # FIGURE 1: Learning Curves + Strategy Gap
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for col, (cond, label) in enumerate(zip(conditions, cond_labels)):
        # --- Top row: Performance ---
        ax = axes[0, col]
        for mode, color, name in [('tom', TOM_COLOR, 'ToM'), ('no_tom', NO_TOM_COLOR, 'No-ToM')]:
            scores = np.array([r['round_scores'] for r in results[mode][cond]])
            mean, lo, hi = _ci95(scores)
            ax.plot(mean, color=color, label=name, linewidth=1.5)
            ax.fill_between(range(len(mean)), lo, hi, color=color, alpha=0.15)
        
        # Mark CG transition
        cg_r = _avg_cg_round(results['tom'][cond])
        if cg_r is not None:
            ax.axvline(cg_r, color=CG_COLOR, linestyle='--', linewidth=1.5, label=f'CG (round {cg_r:.0f})')
        
        ax.set_title(f"Performance: {label}", fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Round")
        if col == 0: ax.set_ylabel("Success Rate")
        if col == 0: ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.2)
        
        # --- Bottom row: Strategy Gap ---
        ax = axes[1, col]
        for mode, color, name, ls in [('tom', TOM_COLOR, 'ToM', '-'), ('no_tom', NO_TOM_COLOR, 'No-ToM', '--')]:
            diffs = []
            for r in results[mode][cond]:
                wf0 = np.array([s['wait_factor'] for s in r['agent_states'][0]])
                wf1 = np.array([s['wait_factor'] for s in r['agent_states'][1]])
                diffs.append(np.abs(wf0 - wf1))
            diffs = np.array(diffs)
            mean, lo, hi = _ci95(diffs)
            ax.plot(mean, color=color, label=name, linewidth=1.5, linestyle=ls)
            ax.fill_between(range(len(mean)), lo, hi, color=color, alpha=0.1)
        
        if cg_r is not None:
            ax.axvline(cg_r, color=CG_COLOR, linestyle='--', linewidth=1.5)
        
        ax.set_title(f"Strategy Gap: {label}", fontweight='bold')
        ax.set_xlabel("Round")
        if col == 0: ax.set_ylabel("|WF_A \u2212 WF_B|")
        if col == 0: ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure1_complete.png'), dpi=150)
    plt.close()
    print("Saved figure1_complete.png")

    # =========================================================================
    # FIGURE 2: H1 Summary (Bar chart + CG rates)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(3)
    width = 0.35
    
    # Left: Final performance
    ax = axes[0]
    tom_means = [np.mean([r['round_scores'][-1] for r in results['tom'][c]]) for c in conditions]
    no_tom_means = [np.mean([r['round_scores'][-1] for r in results['no_tom'][c]]) for c in conditions]
    tom_se = [np.std([r['round_scores'][-1] for r in results['tom'][c]], ddof=1) / np.sqrt(len(results['tom'][c])) for c in conditions]
    no_tom_se = [np.std([r['round_scores'][-1] for r in results['no_tom'][c]], ddof=1) / np.sqrt(len(results['no_tom'][c])) for c in conditions]
    
    bars_tom = ax.bar(x - width/2, tom_means, width, yerr=tom_se, label='ToM', color=TOM_COLOR, capsize=5)
    bars_no = ax.bar(x + width/2, no_tom_means, width, yerr=no_tom_se, label='No-ToM', color=NO_TOM_COLOR, capsize=5)
    
    # Add significance stars
    for i, cond in enumerate(conditions):
        tom_vals = [r['round_scores'][-1] for r in results['tom'][cond]]
        no_tom_vals = [r['round_scores'][-1] for r in results['no_tom'][cond]]
        _, p = stats.ttest_rel(tom_vals, no_tom_vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if sig:
            y_max = max(tom_means[i] + tom_se[i], no_tom_means[i] + no_tom_se[i])
            ax.text(x[i], y_max + 0.03, sig, ha='center', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Final Success Rate')
    ax.set_title('H1: ToM Improves Final Performance', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2, axis='y')
    
    # Right: CG establishment rate
    ax = axes[1]
    n = len(results['tom'][conditions[0]])
    tom_cg = [sum(1 for r in results['tom'][c] if len(r['cg_rounds']) == 2)/n*100 for c in conditions]
    no_tom_cg = [sum(1 for r in results['no_tom'][c] if len(r['cg_rounds']) == 2)/n*100 for c in conditions]
    
    ax.bar(x - width/2, tom_cg, width, label='ToM', color=TOM_COLOR)
    ax.bar(x + width/2, no_tom_cg, width, label='No-ToM', color=NO_TOM_COLOR)
    
    ax.set_ylabel('CG Establishment Rate (%)')
    ax.set_title('Common Ground Success Rate', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure2_tom_comparison.png'), dpi=150)
    plt.close()
    print("Saved: figure2_tom_comparison.png")

    # =========================================================================
    # FIGURE 3: H2 — Cognitive Effort (with CG markers + CI bands)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ['#2E86AB', '#E8850C', '#2D936C']
    
    for condition, color in zip(conditions, colors):
        all_efforts = np.array([r['tom_effort'] for r in results['tom'][condition]])
        mean, lo, hi = _ci95(all_efforts)
        
        ax.plot(mean, label=condition.replace('_', ' ').title(), linewidth=2, color=color)
        ax.fill_between(range(len(mean)), lo, hi, color=color, alpha=0.15)
        
        # Mark average CG time
        cg_r = _avg_cg_round(results['tom'][condition])
        if cg_r is not None:
            ax.axvline(cg_r, linestyle='--', alpha=0.6, color=color, linewidth=1.5)
            ax.annotate(f'CG', xy=(cg_r, mean[min(int(cg_r), len(mean)-1)]),
                       xytext=(cg_r + 3, mean[min(int(cg_r), len(mean)-1)] + 10),
                       fontsize=9, color=color, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color=color, lw=1))
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Cognitive Effort (ToM Updates / Round)')
    ax.set_title('H2: Cognitive Effort Drops After Common Ground', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure3_cognitive_effort.png'), dpi=150)
    plt.close()
    print("Saved: figure3_cognitive_effort.png")

def save_results_json(results):
    """Save experiment results to JSON for reproducibility."""
    import json
    
    output = {}
    for mode in ['tom', 'no_tom']:
        output[mode] = {}
        for cond in ['similar', 'different', 'very_different']:
            runs = []
            for r in results[mode][cond]:
                run_data = {
                    'condition': r['condition'],
                    'use_tom': r['use_tom'],
                    'round_scores': [float(s) for s in r['round_scores']],
                    'tom_effort': [int(e) for e in r['tom_effort']],
                    'cg_rounds': {str(k): int(v) for k, v in r['cg_rounds'].items()},
                    'final_wait_factors': {
                        str(aid): float(r['agent_states'][aid][-1]['wait_factor'])
                        for aid in r['agent_states']
                    }
                }
                runs.append(run_data)
            output[mode][cond] = runs
    
    filepath = os.path.join(OUTPUT_DIR, 'experiment_results.json')
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {filepath}")