# The Mind: Common Ground & Theory of Mind Simulation

A simulation studying how Theory of Mind (ToM) and Common Ground (CG) affect coordination between agents in a simplified version of *The Mind* card game.

Inspired by: *Common Ground Provides a Mental Shortcut in Agent-Agent Interaction* (van der Meulen, Verbrugge & Van Duijn, 2024), https://doi.org/10.3233/FAIA240201.

---

## Research Questions

**RQ1 (Performance):** Does ToM improve coordination, especially when agents start with mismatched strategies?

**RQ2 (Mental shortcut):** Does cognitive effort decrease after Common Ground emerges, while performance remains stable?

## How It Works

Two agents repeatedly play a simplified *The Mind*-inspired card game. Each agent holds cards numbered 1–100 and must play them in globally ascending order, without seeing the partner's hand. The only coordination signal is *when* and *what* the partner plays.

Each agent has a **wait factor** controlling how long it waits before playing a card (higher = more cautious). Agents with **ToM enabled** observe partner plays, infer the partner's wait factor, and align their own strategy accordingly. Once strategies stabilize, a **Common Ground** detector locks in the shared convention and reduces further modeling effort.

The experiment compares ToM vs No-ToM agents across three initial mismatch regimes (similar, different, very different) to test whether ToM accelerates convergence and whether CG reduces cognitive cost.

## Project Structure

```
.
├── agent.py           # Agent with timing policy, ToM module, CG detector
├── game.py            # The Mind game engine (deals cards, evaluates plays)
├── experiment.py      # Experiment runner, statistical analysis, figure generation
├── main.py            # Entry point — runs the full experiment
├── visualizer.py      # Interactive Pygame visualization of live games
├── requirements.txt   # Python dependencies
└── README.md
```

## Setup

**Requirements:** Python 3.8+

```bash
pip install -r requirements.txt
```

## Usage

### Run the experiment

Runs 30 repetitions × 200 rounds across all conditions. Produces figures and statistical output.

```bash
python main.py
```

**Outputs:**
- `figure1_complete.png` — Learning curves and strategy gap over rounds
- `figure2_tom_comparison.png` — Final performance bars and CG establishment rates
- `figure3_cognitive_effort.png` — Cognitive effort (ToM updates) before/after CG
- `experiment_results.json` — Full results data for reproducibility

### Run the visualizer

Interactive Pygame window showing agents playing in real time.

```bash
python visualizer.py
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `cards_per_player` | 4 | Cards dealt to each agent per game |
| `learning_rate` | 0.05 | Base rate for strategy updates |
| `weber_fraction` | 0.15 | Noise scaling for human-like timing |
| `games_per_round` | 30 | Games aggregated per learning round |
| `n_runs` | 30 | Independent repetitions per condition |
| `num_rounds` | 200 | Rounds per run |
| CG window | 15 rounds | Stability window for CG detection |
| EMA α (pre-CG) | 0.15 | Partner model update rate |
| EMA α (post-CG) | 0.01 | Reduced update rate after CG |

### Mismatch Regimes

| Regime | Agent 0 WF | Agent 1 WF |
|--------|-----------|-----------|
| Similar | 1.1 | 0.9 |
| Different | 1.8 | 0.5 |
| Very Different | 3.0 | 0.4 |

## Agent Design

- **Timing policy:** Play time = `card_value × wait_factor + Weber noise`
- **ToM module:** Infers partner's wait factor from observed (card, tick) pairs via exponential moving average. Cards below 5 are filtered due to poor signal-to-noise ratio.
- **Gap heuristic:** Short-horizon reactive play when partner just played a nearby card (ToM agents only).
- **CG detection:** Triggered when own strategy, partner estimate, and performance all stabilize within tolerances over a 15-round window. CG can be revoked if performance drops significantly.
- **Post-CG behavior:** Learning rate reduced to 10% of base; partner model updates use α = 0.01 ("soft freeze").

## Results Summary

- **H1 supported:** ToM agents converge faster and reach higher final performance, with the largest advantage (~8.8 rounds faster CG) under very different starting strategies.
- **H2 supported:** After CG establishment, cognitive effort (ToM update counts) drops by ~95–98% while performance remains stable.
- No-ToM agents cannot establish CG under the current operational definition (which requires ToM-based partner estimation), but this is a design choice — alternative CG definitions could apply to any agent type.

## Author

**Dimitrios Sarris**

## License

This project was created for academic purposes.