import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import random

class Agent:
    """
    Agent that plays The Mind using timing-based coordination.
    
    Includes fixes for:
    1. Low-card signal noise (Variance Trap)
    2. Strategy oscillation (Damping)
    3. Absolute-time drift (Gap Heuristic)
    """

    def __init__(self, agent_id: int, wait_factor: float = 1.0, learning_rate: float = 0.05, use_tom: bool = True):
        self.agent_id = agent_id
        self.wait_factor = np.clip(wait_factor, 0.3, 5.0)
        self.learning_rate = learning_rate
        self.use_tom = use_tom 

        self.initial_wait_factor = self.wait_factor

        # ===== Theory of Mind: Partner Model =====
        # Initialize partner estimate to own wait_factor (egocentric projection:
        # "I assume my partner plays like me"). This must be updated through
        # actual observations, making CG detection dependent on real interaction.
        self.partner_wait_factor_estimate = self.wait_factor
        self.partner_observations = deque(maxlen=100) 
        self.tom_updates_count = 0          # Cumulative
        self.tom_updates_this_round = 0     # Per-round counter

        # ===== Learning History =====
        self.wait_factor_history = []
        self.partner_estimate_history = []
        self.score_history = deque(maxlen=30)
        self.mistake_history = deque(maxlen=30)

        # ===== Common Ground State =====
        self.cg_established = False
        self.cg_round = None
        self._cg_perf_at_establishment = 0.0
        
        # ===== Current Game State =====
        self.hand: List[int] = []
        self.planned_play_times: Dict[int, float] = {}
        
        # State tracking for Gap Heuristic
        self.last_partner_play_tick: Optional[int] = None
        self.last_partner_play_card: Optional[int] = None

    def receive_cards(self, cards: List[int]):
        """Receive cards and reset round-specific state."""
        self.hand = sorted(cards)
        self.last_partner_play_tick = None
        self.last_partner_play_card = None
        self._plan_play_times()

    def _plan_play_times(self):
        """Generate timing using Weber's Law."""
        self.planned_play_times = {}
        weber_fraction = 0.15 

        for card in self.hand:
            target_wait = card * self.wait_factor
            noise_std = max(0.5, target_wait * weber_fraction)
            actual_time = target_wait + np.random.normal(0, noise_std)
            self.planned_play_times[card] = max(0, actual_time)

    def decide(self, current_tick: int) -> Tuple[bool, int]:
        """Decide whether to play lowest card this tick."""
        if not self.hand:
            return False, -1

        my_lowest = min(self.hand)
        
        # Gap Heuristic (Relative Timing)
        if self.use_tom and self._check_gap_play(current_tick, my_lowest):
            return True, my_lowest

        # Standard Absolute Timing
        play_time = self.planned_play_times.get(my_lowest, float('inf'))
        if current_tick >= play_time:
            return True, my_lowest
            
        return False, -1

    def _check_gap_play(self, current_tick: int, my_card: int) -> bool:
        """Check if we should play immediately based on partner's last move."""
        if self.last_partner_play_tick is None:
            return False
            
        # Time since last play
        ticks_since_play = current_tick - self.last_partner_play_tick
        
        # Card gap
        card_gap = my_card - self.last_partner_play_card
        
        # Rule: If gap is small (<= 2) and time is short (< 15 ticks), 
        # assume Rapid Fire sequence and play now.
        if card_gap <= 2 and ticks_since_play < 15:
            return True
            
        return False

    def play_card(self, card: int):
        if card in self.hand:
            self.hand.remove(card)

    def observe_partner_play(self, partner_card: int, tick_played: int):
        """
        Observe partner play.
        
        No-ToM agents: This method returns immediately. No-ToM agents 
        are fully egocentric — they do not use partner actions in any 
        way (neither for short-horizon gap heuristic nor for long-term
        strategy alignment).
        
        ToM agents: Updates two subsystems:
        1. Gap Heuristic state: stores last partner card/tick for
           short-horizon reactive play (if my card is close, play now).
        2. Partner model (EMA): infers partner's wait_factor from 
           observed (card, tick) pairs. Filters cards < 5 due to 
           poor signal-to-noise ratio.
        """
        # Always update state for Gap Heuristic
        if self.use_tom:
            self.last_partner_play_tick = tick_played
            self.last_partner_play_card = partner_card
        else:
            # No-ToM agents ignore partner actions entirely
            return

        if partner_card < 5:
            return

        # Infer partner's wait_factor
        inferred_wf = tick_played / partner_card
        inferred_wf = np.clip(inferred_wf, 0.3, 5.0)

        self.partner_observations.append((partner_card, tick_played, inferred_wf))
        
        # Only increment effort if not in CG
        if not self.cg_established:
            self.tom_updates_count += 1
            self.tom_updates_this_round += 1

        # Update partner model (EMA)
        if len(self.partner_observations) >= 3:
            # If CG established, barely update
            alpha = 0.01 if self.cg_established else 0.15
            
            recent_wfs = [obs[2] for obs in list(self.partner_observations)[-10:]]
            recent_mean = np.mean(recent_wfs)
            
            self.partner_wait_factor_estimate = (
                (1 - alpha) * self.partner_wait_factor_estimate + 
                alpha * recent_mean
            )

    def update_after_round(self, success_rate: float, mistake_rate: float, round_num: int):
        """
        Update strategy.
        FIX: Uses Momentum to prevent oscillation.
        """
        # Record and reset per-round effort counter
        self._last_round_effort = self.tom_updates_this_round
        self.tom_updates_this_round = 0
        
        # Soft Freeze instead of Hard Freeze
        effective_lr = self.learning_rate * 0.1 if self.cg_established else self.learning_rate

        self.score_history.append(success_rate)
        self.mistake_history.append(mistake_rate)
        self.wait_factor_history.append(self.wait_factor)
        self.partner_estimate_history.append(self.partner_wait_factor_estimate)

        target_delta = 0.0

        # Rule 1: Crash Avoidance
        if mistake_rate > 0:
            target_delta += effective_lr * (1.0 + mistake_rate * 3.0)
        
        # Rule 2: Efficiency
        elif success_rate >= 0.95:
            target_delta -= effective_lr * 0.15

        # Rule 3: ToM Alignment
        if self.use_tom and len(self.partner_observations) >= 5:
            partner_wf = self.partner_wait_factor_estimate
            diff = partner_wf - self.wait_factor
            alignment_strength = 0.4 if mistake_rate > 0.1 else 0.2
            target_delta += diff * effective_lr * alignment_strength

        # Random exploration
        target_delta += np.random.normal(0, 0.02)

        # Momentum Update
        proposed_wf = self.wait_factor + target_delta
        self.wait_factor = (0.8 * self.wait_factor) + (0.2 * proposed_wf)
        
        self.wait_factor = np.clip(self.wait_factor, 0.3, 5.0)

        self._check_cg(round_num)

    def _check_cg(self, round_num: int):
        """
        Check for Common Ground emergence.
        
        CG is established when the agent has evidence that it and its partner
        have converged to a shared convention THROUGH mutual adaptation. This
        requires all of the following over a stability window:
        
        1. CONVERGENCE: The gap between own WF and estimated partner WF must
           have closed meaningfully relative to how far apart they started.
           This ensures CG reflects actual alignment through modeling, not 
           just starting close.
           
        2. OWN STABILITY: The agent's strategy has stopped changing.
           
        3. PARTNER STABILITY: The agent's estimate of its partner's strategy
           has stopped changing.
           
        4. PERFORMANCE CONSISTENCY: Joint performance is stable and above a
           minimum floor.
        
        Because each criterion depends on noisy observations (Weber-Law 
        timing, stochastic game outcomes), the round at which ALL criteria
        are simultaneously met varies naturally across random seeds.
        
        CG can be REVOKED if performance drops significantly after 
        establishment (van der Meulen et al.'s "false CG" / Table 1).
        """
        if not self.use_tom:
            return
        
        # --- CG Revocation ---
        if self.cg_established:
            recent_scores = list(self.score_history)[-10:]
            if len(recent_scores) >= 10:
                current_perf = np.mean(recent_scores)
                if current_perf < self._cg_perf_at_establishment - 0.15:
                    self.cg_established = False
                    self.cg_round = None
            return
        
        # Need enough history for meaningful stability assessment
        window = 15
        if len(self.wait_factor_history) < window:
            return
        
        # --- Criterion 1: CONVERGENCE ---
        initial_gap = abs(self.initial_wait_factor - 1.0)
        current_gap = abs(self.wait_factor - self.partner_wait_factor_estimate)
        
        # Must close at least 60% of initial gap, with absolute cap of 0.35
        # so very_different has a reachable target
        convergence_threshold = min(0.35, max(0.10, initial_gap * 0.40))
        if current_gap > convergence_threshold:
            return
        
        # --- Criterion 2: OWN STABILITY ---
        recent_wf = self.wait_factor_history[-window:]
        if np.std(recent_wf) > 0.10:
            return

        # --- Criterion 3: PARTNER MODEL STABILITY ---
        recent_partner = self.partner_estimate_history[-window:]
        if np.std(recent_partner) > 0.12:
            return

        # --- Criterion 4: PERFORMANCE CONSISTENCY ---
        recent_scores = list(self.score_history)[-window:]
        if len(recent_scores) < window:
            return
        if np.std(recent_scores) > 0.08:
            return
        if np.mean(recent_scores) < 0.55:
            return
        
        # All criteria met, establish CG
        self.cg_established = True
        self.cg_round = round_num
        self._cg_perf_at_establishment = np.mean(recent_scores)

    def get_state(self) -> Dict:
        return {
            'wait_factor': self.wait_factor,
            'partner_estimate': self.partner_wait_factor_estimate,
            'tom_updates': self.tom_updates_count,
            'tom_effort_this_round': getattr(self, '_last_round_effort', 0),
            'cg_established': self.cg_established,
            'cg_round': self.cg_round
        }