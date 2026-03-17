import pygame
import sys
import time
import numpy as np
from typing import List
import random
from agent import Agent
from game import TheMindGame

# =============================================================================
# CONFIGURATION & COLORS
# =============================================================================
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 30
TICKS_PER_FRAME = 1

# Colors
COLOR_BG = (30, 30, 30)
COLOR_TABLE = (34, 139, 34)  # Felt Green
COLOR_CARD = (240, 240, 240)
COLOR_CARD_TEXT = (20, 20, 20)
COLOR_TEXT = (255, 255, 255)
COLOR_AGENT_A = (46, 134, 171)  # Blue
COLOR_AGENT_B = (162, 59, 114)  # Magenta
COLOR_CG_GOLD = (255, 215, 0)
COLOR_MISTAKE = (200, 50, 50)
COLOR_SUCCESS = (50, 200, 50)

# =============================================================================
# VISUALIZER CLASS
# =============================================================================

class VisualTheMind(TheMindGame):
    """
    A visual wrapper for TheMindGame using Pygame.
    """
    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, cards_per_player=4):
        super().__init__(cards_per_player=cards_per_player)
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("The Mind: ToM & Common Ground Simulation")
        self.font_big = pygame.font.SysFont('Arial', 40, bold=True)
        self.font_med = pygame.font.SysFont('Arial', 24)
        self.font_small = pygame.font.SysFont('Arial', 18)
        self.clock = pygame.time.Clock()
        
        self.paused = False
        self.speed_multiplier = 1
        self.games_per_round = 30  # Match experiment

    def draw_card(self, x, y, value, color=COLOR_CARD, scale=1.0):
        w, h = 60 * scale, 90 * scale
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        pygame.draw.rect(self.screen, (0, 0, 0), rect, 2, border_radius=5)
        
        if value is not None:
            text = self.font_big.render(str(value), True, COLOR_CARD_TEXT)
            self.screen.blit(text, (x + w/2 - text.get_width()/2, y + h/2 - text.get_height()/2))

    def draw_agent_hud(self, agent: Agent, x, y, color):
        # Background panel
        rect = pygame.Rect(x, y, 350, 250)
        border_col = COLOR_CG_GOLD if agent.cg_established else (100, 100, 100)
        border_width = 4 if agent.cg_established else 2
        pygame.draw.rect(self.screen, (50, 50, 50), rect, border_radius=10)
        pygame.draw.rect(self.screen, border_col, rect, border_width, border_radius=10)

        # Name & Status
        tom_status = "ToM ON" if agent.use_tom else "ToM OFF"
        name = f"Agent {agent.agent_id} ({tom_status})"
        lbl = self.font_med.render(name, True, color)
        self.screen.blit(lbl, (x + 15, y + 15))

        if agent.cg_established:
            cg_lbl = self.font_small.render("★ COMMON GROUND ★", True, COLOR_CG_GOLD)
            self.screen.blit(cg_lbl, (x + 15, y + 45))

        # Stats
        stats = [
            f"Wait Factor: {agent.wait_factor:.3f}",
            f"Est. Partner WF: {agent.partner_wait_factor_estimate:.3f}",
            f"Cognitive Load: {agent.tom_updates_count}",
            f"Gap: {abs(agent.wait_factor - agent.partner_wait_factor_estimate):.3f}"
        ]
        
        for i, stat in enumerate(stats):
            txt = self.font_small.render(stat, True, COLOR_TEXT)
            self.screen.blit(txt, (x + 15, y + 68 + i * 22))

        # Hand (Cards)
        hand_x = x + 15
        hand_y = y + 160
        for card in agent.hand:
            self.draw_card(hand_x, hand_y, card, scale=0.7)
            hand_x += 50

        # "Thinking" Indicator
        if agent.hand:
            lowest = min(agent.hand)
            planned_time = agent.planned_play_times.get(lowest, 0)
            time_txt = self.font_small.render(f"Next Play: Tick {int(planned_time)}", True, (200, 200, 200))
            self.screen.blit(time_txt, (x + 15, y + 230))

    def play_single_game(self, agents: List[Agent], max_ticks: int = 3000):
        """
        Play one game with visuals disabled (background game for averaging).
        """
        total_cards = self.cards_per_player * len(agents)
        all_cards = random.sample(range(1, self.max_card + 1), total_cards)

        for i, agent in enumerate(agents):
            agent_cards = all_cards[i * self.cards_per_player:(i + 1) * self.cards_per_player]
            agent.receive_cards(agent_cards)

        cards_played_correctly = 0
        mistakes = 0
        lives = self.cards_per_player - 1

        for tick in range(max_ticks):
            if all(len(a.hand) == 0 for a in agents):
                break
            if lives <= 0:
                break

            decisions = {}
            for agent in agents:
                should_play, card = agent.decide(tick)
                if should_play and card > 0:
                    decisions[agent.agent_id] = card

            if not decisions:
                continue

            plays_this_tick = list(decisions.items())
            random.shuffle(plays_this_tick)

            for agent_id, card in plays_this_tick:
                if lives <= 0:
                    break

                agent = agents[agent_id]
                other = agents[1 - agent_id]

                lowest_unplayed = float('inf')
                for a in agents:
                    if a.hand:
                        lowest_unplayed = min(lowest_unplayed, min(a.hand))

                if card == lowest_unplayed:
                    agent.play_card(card)
                    cards_played_correctly += 1
                    other.observe_partner_play(card, tick)
                else:
                    mistakes += 1
                    lives -= 1
                    agent.play_card(card)
                    # Discard all skipped cards (matching game.py)
                    for a in agents:
                        skipped = [c for c in a.hand if c < card]
                        for sc in skipped:
                            a.play_card(sc)

        return {
            'success_rate': cards_played_correctly / total_cards,
            'mistake_rate': mistakes / total_cards,
        }

    def play_visual_round(self, agents: List[Agent], round_num: int):
        """
        Play one VISUAL game then run background games to get a proper averaged learning signal.
        """
        # ---- Visual game ----
        total_cards = self.cards_per_player * len(agents)
        all_cards = random.sample(range(1, self.max_card + 1), total_cards)

        for i, agent in enumerate(agents):
            agent_cards = all_cards[i * self.cards_per_player:(i + 1) * self.cards_per_player]
            agent.receive_cards(agent_cards)

        played_pile = []
        lives = self.cards_per_player - 1
        game_over = False
        visual_success_rate = 0.0
        visual_mistake_rate = 0.0
        cards_played_correctly = 0
        mistakes = 0
        mistake_flash = 0
        
        tick = 0
        max_ticks = 4000
        
        while tick < max_ticks and not game_over:
            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    if event.key == pygame.K_RIGHT:
                        self.speed_multiplier = min(20, self.speed_multiplier + 1)
                    if event.key == pygame.K_LEFT:
                        self.speed_multiplier = max(1, self.speed_multiplier - 1)

            if self.paused:
                # Still render while paused
                self._render_frame(agents, round_num, tick, lives, played_pile, mistake_flash)
                self.clock.tick(FPS)
                continue

            # --- Logic Update Loop ---
            for _ in range(self.speed_multiplier * TICKS_PER_FRAME):
                tick += 1

                if all(len(a.hand) == 0 for a in agents):
                    game_over = True
                    break
                if lives <= 0:
                    game_over = True
                    break

                # Collect all decisions this tick
                decisions = {}
                for agent in agents:
                    play, card = agent.decide(tick)
                    if play and card > 0:
                        decisions[agent.agent_id] = card
                
                if not decisions:
                    continue

                # Process ALL plays this tick in random order
                plays_this_tick = list(decisions.items())
                random.shuffle(plays_this_tick)

                for agent_id, card in plays_this_tick:
                    if lives <= 0:
                        break

                    playing_agent = agents[agent_id]
                    other_agent = agents[1 - agent_id]

                    # Find global lowest unplayed card
                    lowest_unplayed = float('inf')
                    for a in agents:
                        if a.hand:
                            lowest_unplayed = min(lowest_unplayed, min(a.hand))

                    if card == lowest_unplayed:
                        # Correct play
                        playing_agent.play_card(card)
                        cards_played_correctly += 1
                        played_pile.append((card, playing_agent, False))
                        other_agent.observe_partner_play(card, tick)
                    else:
                        # Mistake — discard skipped cards
                        mistakes += 1
                        lives -= 1
                        playing_agent.play_card(card)
                        
                        for a in agents:
                            skipped = [c for c in a.hand if c < card]
                            for sc in skipped:
                                a.play_card(sc)

                        played_pile.append((card, playing_agent, True))
                        mistake_flash = 15

            # Render
            if mistake_flash > 0:
                mistake_flash -= 1
            self._render_frame(agents, round_num, tick, lives, played_pile, mistake_flash)
            self.clock.tick(FPS)

        visual_success_rate = cards_played_correctly / total_cards
        visual_mistake_rate = mistakes / total_cards

        # ---- Background games for averaged learning signal ----
        round_scores = [visual_success_rate]
        round_mistakes = [visual_mistake_rate]

        for _ in range(self.games_per_round - 1):
            result = self.play_single_game(agents)
            round_scores.append(result['success_rate'])
            round_mistakes.append(result['mistake_rate'])

        avg_score = np.mean(round_scores)
        avg_mistakes = np.mean(round_mistakes)

        return avg_score, avg_mistakes

    def _render_frame(self, agents, round_num, tick, lives, played_pile, mistake_flash):
        """All rendering logic extracted to avoid duplication."""
        self.screen.fill(COLOR_BG)

        # Mistake Flash overlay
        if mistake_flash > 0:
            flash_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(180 * (mistake_flash / 15))
            flash_surface.fill((*COLOR_MISTAKE, alpha))
            self.screen.blit(flash_surface, (0, 0))

        # Info Bar
        pause_str = " [PAUSED]" if self.paused else ""
        info = f"Round: {round_num}/50 | Tick: {tick} | Lives: {lives} | Speed: {self.speed_multiplier}x{pause_str}"
        info_surf = self.font_med.render(info, True, COLOR_TEXT)
        self.screen.blit(info_surf, (20, 20))

        # Help Text
        help_txt = self.font_small.render("Controls: SPACE=Pause | LEFT/RIGHT=Speed", True, (150, 150, 150))
        self.screen.blit(help_txt, (SCREEN_WIDTH - 370, 25))

        # Draw Table
        table_rect = pygame.Rect(100, 130, 800, 300)
        pygame.draw.ellipse(self.screen, COLOR_TABLE, table_rect)

        # Draw Pile
        pile_x, pile_y = SCREEN_WIDTH // 2 - 30, 230
        if played_pile:
            last_card, owner, is_mistake = played_pile[-1]
            col = COLOR_MISTAKE if is_mistake else COLOR_CARD
            self.draw_card(pile_x, pile_y, last_card, col)
            owner_txt = self.font_small.render(f"Agent {owner.agent_id}", True, COLOR_TEXT)
            self.screen.blit(owner_txt, (pile_x, pile_y + 100))
            
            # Show pile count
            pile_count_txt = self.font_small.render(f"Played: {len(played_pile)}", True, (200, 200, 200))
            self.screen.blit(pile_count_txt, (pile_x - 10, pile_y - 25))
        else:
            pygame.draw.rect(self.screen, (30, 100, 30), (pile_x, pile_y, 60, 90), 2)

        # Draw Agents
        self.draw_agent_hud(agents[0], 30, 440, COLOR_AGENT_A)
        self.draw_agent_hud(agents[1], 620, 440, COLOR_AGENT_B)

        # Draw Strategy Convergence Graph
        self.draw_mini_graph(agents, 350, 50)

        pygame.display.flip()

    def draw_mini_graph(self, agents, x, y):
        """Visualize wait factor positions with correct normalization."""
        w, h = 300, 70
        pygame.draw.rect(self.screen, (20, 20, 20), (x, y, w, h))
        pygame.draw.rect(self.screen, (100, 100, 100), (x, y, w, h), 1)

        mid_y = y + h // 2

        # Normalization: cover 0.3 to 5.0 (full agent WF range)
        wf_min, wf_max = 0.3, 5.0

        def get_x(wf):
            norm = (wf - wf_min) / (wf_max - wf_min)
            norm = max(0.0, min(1.0, norm))
            return int(x + 10 + norm * (w - 20))

        wf0 = agents[0].wait_factor
        wf1 = agents[1].wait_factor

        pos0 = get_x(wf0)
        pos1 = get_x(wf1)

        # Draw scale markers
        for marker_wf in [0.5, 1.0, 2.0, 3.0]:
            mx = get_x(marker_wf)
            pygame.draw.line(self.screen, (60, 60, 60), (mx, y + 5), (mx, y + h - 5), 1)

        pygame.draw.circle(self.screen, COLOR_AGENT_A, (pos0, mid_y - 10), 8)
        pygame.draw.circle(self.screen, COLOR_AGENT_B, (pos1, mid_y + 10), 8)
        pygame.draw.line(self.screen, COLOR_TEXT, (pos0, mid_y - 10), (pos1, mid_y + 10), 1)

        dist = abs(wf0 - wf1)
        txt = self.font_small.render(f"Gap: {dist:.3f}", True, COLOR_TEXT)
        self.screen.blit(txt, (x + 5, y + 3))

        # Labels
        lbl0 = self.font_small.render(f"A:{wf0:.2f}", True, COLOR_AGENT_A)
        lbl1 = self.font_small.render(f"B:{wf1:.2f}", True, COLOR_AGENT_B)
        self.screen.blit(lbl0, (x + 5, y + h - 18))
        self.screen.blit(lbl1, (x + w - 55, y + h - 18))


# =============================================================================
# MAIN RUNNER WITH MENU
# =============================================================================

def get_user_config():
    print("\n" + "=" * 50)
    print("VISUALIZER CONFIGURATION")
    print("=" * 50)

    while True:
        mode = input("Enable Theory of Mind? (y/n): ").lower().strip()
        if mode in ['y', 'yes']:
            use_tom = True
            break
        elif mode in ['n', 'no']:
            use_tom = False
            break

    print("\nChoose Start Strategy (Wait Factors):")
    print("1. Similar        (1.1 vs 0.9) - Easy")
    print("2. Different      (1.8 vs 0.5) - Medium")
    print("3. Very Different (3.0 vs 0.4) - Hard")

    configs = {
        '1': {'name': 'similar', '0': 1.1, '1': 0.9},
        '2': {'name': 'different', '0': 1.8, '1': 0.5},
        '3': {'name': 'very_different', '0': 3.0, '1': 0.4}
    }

    while True:
        choice = input("Select (1-3): ").strip()
        if choice in configs:
            cfg = configs[choice]
            break

    print("\n" + "=" * 50)
    print(f"Launching: {cfg['name'].upper()} | ToM: {use_tom}")
    print("=" * 50 + "\n")
    return cfg, use_tom


def run_visual_demo():
    cfg, use_tom = get_user_config()

    # Match experiment: 4 cards per player
    viz = VisualTheMind(cards_per_player=4)

    agents = [
        Agent(0, wait_factor=cfg['0'], learning_rate=0.05, use_tom=use_tom),
        Agent(1, wait_factor=cfg['1'], learning_rate=0.05, use_tom=use_tom)
    ]

    print("Starting Visual Simulation...")

    for round_num in range(1, 51):
        avg_score, avg_mistakes = viz.play_visual_round(agents, round_num)

        # Learning update with averaged signal (matches experiment)
        for agent in agents:
            agent.update_after_round(avg_score, avg_mistakes, round_num)

        if not pygame.get_init():
            break
        time.sleep(0.3)

    print("Simulation Complete.")
    pygame.quit()


if __name__ == "__main__":
    run_visual_demo()