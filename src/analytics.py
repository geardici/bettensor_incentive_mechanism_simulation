import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from utils import logger

logger = logging.getLogger(__name__)

class SimulationAnalytics:
    def __init__(self, simulation):
        self.simulation = simulation
        self.scoring_system = simulation.scoring_system
        self.miner_stats = simulation.miner_stats
        self.output_dir = "analytics_output"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Analytics output directory created: {self.output_dir}")

    def run_all_analytics(self):
        print("Starting analytics...")
        logger.info("Running all analytics...")
        self.analyze_tier_progression()
        self.analyze_tier_transition_times()
        self.analyze_swap_statistics()
        self.analyze_overall_performance()
        self.analyze_miner_retention()
        self.detect_anomalies()
        self.plot_miner_scores(num_miners=20)
        self.plot_tier_changes(num_miners=20)
        self.plot_weights_over_time()
        self.plot_cumulative_weights()
        self.plot_all_miners_cumulative_weights()
        print("Analytics completed. Plots should be displayed in your browser.")
        logger.info("All analytics completed. Plots should be displayed in your browser.")

    def analyze_tier_progression(self):
        print("Analyzing tier progression...")
        milestone_days = [0, 100, 300, 500, 700]
        skill_levels = ['conservative', 'balanced', 'aggressive']
        
        logger.info("Tier progression analysis:")
        
        for start_day in milestone_days:
            logger.info(f"\nMiners registered after day {start_day}:")
            
            all_miners = [m for m, data in self.miner_stats.items() if data['registration_day'] > start_day]
            total_miners = len(all_miners)
            
            if total_miners == 0:
                logger.info(f"  No miners registered after day {start_day}")
                continue
            
            tier_counts = [0] * 5
            for m in all_miners:
                tier_history = self.scoring_system.get_miner_tier_history(m)
                highest_tier = max(tier_history)
                tier_counts[highest_tier - 1] += 1
            
            logger.info(f"  Total miners: {total_miners}")
            for tier, count in enumerate(tier_counts, 1):
                probability = count / total_miners
                logger.info(f"  Reached Tier {tier}: Count={count}, Probability={probability:.2f}")
            
            # Calculate average time to reach each tier
            for tier in range(2, 6):  # Tiers 2 to 5
                times_to_tier = []
                for m in all_miners:
                    tier_history = self.scoring_system.get_miner_tier_history(m)
                    registration_day = self.miner_stats[m]['registration_day']
                    time_to_tier = next((i for i, t in enumerate(tier_history) if t >= tier), None)
                    if time_to_tier is not None:
                        times_to_tier.append(time_to_tier - registration_day)
                
                if times_to_tier:
                    avg_time = sum(times_to_tier) / len(times_to_tier)
                    logger.info(f"  Average time to reach Tier {tier}: {avg_time:.2f} days")
            
            # Analyze by skill level
            for skill in skill_levels:
                skill_miners = [m for m in all_miners if self.miner_stats[m]['skill_level'] == skill]
                skill_total = len(skill_miners)
                
                if skill_total == 0:
                    continue
                
                logger.info(f"\n  {skill.capitalize()} miners:")
                logger.info(f"    Total: {skill_total}")
                
                skill_tier_counts = [0] * 5
                for m in skill_miners:
                    tier_history = self.scoring_system.get_miner_tier_history(m)
                    highest_tier = max(tier_history)
                    skill_tier_counts[highest_tier - 1] += 1
                
                for tier, count in enumerate(skill_tier_counts, 1):
                    probability = count / skill_total
                    logger.info(f"    Reached Tier {tier}: Count={count}, Probability={probability:.2f}")

    def analyze_tier_transition_times(self):
        logger.info("Analyzing tier transition times...")
        transition_times = {(i, j): [] for i in range(1, 5) for j in range(i+1, 6)}
        demotion_times = {(i, j): [] for i in range(2, 6) for j in range(1, i)}
        
        for miner, data in self.miner_stats.items():
            tier_history = self.scoring_system.get_miner_tier_history(miner)
            for t in range(1, len(tier_history)):
                from_tier = tier_history[t-1].item()
                to_tier = tier_history[t].item()
                if to_tier > from_tier:
                    if (from_tier, to_tier) in transition_times:
                        transition_times[(from_tier, to_tier)].append(t)
                elif to_tier < from_tier:
                    if (from_tier, to_tier) in demotion_times:
                        demotion_times[(from_tier, to_tier)].append(t)
        
        logger.info("Promotion transition times:")
        for (start, end), times in transition_times.items():
            if times:
                logger.info(f"Tier {start} -> {end} transition times: "
                            f"Min={min(times)}, Max={max(times)}, "
                            f"Mean={sum(times)/len(times):.2f}, Median={sorted(times)[len(times)//2]}")
        
        logger.info("\nDemotion transition times:")
        for (start, end), times in demotion_times.items():
            if times:
                logger.info(f"Tier {start} -> {end} demotion times: "
                            f"Min={min(times)}, Max={max(times)}, "
                            f"Mean={sum(times)/len(times):.2f}, Median={sorted(times)[len(times)//2]}")

    def analyze_swap_statistics(self):
        # Implementation of analyze_swap_statistics
        ...

    def analyze_overall_performance(self):
        # Implementation of analyze_overall_performance
        ...

    def analyze_miner_retention(self):
        # Implementation of analyze_miner_retention
        ...

    def detect_anomalies(self):
        # Implementation of detect_anomalies
        ...

    def plot_miner_scores(self, miner_ids=None, num_miners=5, score_types=None):
        print("Plotting miner scores...")
        if score_types is None:
            score_types = ['composite', 'clv', 'roi', 'sharpe', 'sortino']
        
        if miner_ids is None:
            # If no specific miners are requested, choose top performers
            latest_scores = self.scoring_system.composite_scores[:, -1]
            top_miners = latest_scores.argsort(descending=True)[:num_miners]
            miner_ids = top_miners.tolist()
        
        num_rows = len(score_types)
        fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=[f'{score_type.capitalize()} Scores' for score_type in score_types])
        
        for i, score_type in enumerate(score_types, start=1):
            for miner_id in miner_ids:
                if score_type == 'composite':
                    scores = self.scoring_system.composite_scores[miner_id, :].cpu().numpy()
                elif score_type == 'clv':
                    scores = self.scoring_system.clv_scores[miner_id, :].cpu().numpy()
                elif score_type == 'roi':
                    scores = self.scoring_system.roi_scores[miner_id, :].cpu().numpy()
                elif score_type == 'sharpe':
                    scores = self.scoring_system.sharpe_scores[miner_id, :].cpu().numpy()
                elif score_type == 'sortino':
                    scores = self.scoring_system.sortino_scores[miner_id, :].cpu().numpy()
                else:
                    raise ValueError(f"Unknown score type: {score_type}")
                
                fig.add_trace(
                    go.Scatter(x=list(range(len(scores))), y=scores, name=f'Miner {miner_id}', mode='lines'),
                    row=i, col=1
                )
        
        fig.update_layout(height=300*num_rows, title_text="Miner Scores Over Time", showlegend=True)
        fig.update_xaxes(title_text="Day", row=num_rows, col=1)
        
        for i, score_type in enumerate(score_types, start=1):
            fig.update_yaxes(title_text=f'{score_type.capitalize()} Score', row=i, col=1)
        
        fig.show()
        print("Miner scores plot displayed.")
        logger.info("Miner scores plot displayed.")

    def plot_tier_changes(self, num_miners=5):
        # Implementation of plot_tier_changes
        ...

    def plot_weights_over_time(self):
        # Implementation of plot_weights_over_time
        ...

    def plot_cumulative_weights(self):
        # Implementation of plot_cumulative_weights
        ...

    def plot_all_miners_cumulative_weights(self):
        print("Plotting all miners' cumulative weights...")
        logger.info("Plotting all miners' cumulative weights...")

        fig = go.Figure()

        # Get all unique miners that have ever been registered
        all_miners = set(self.miner_stats.keys())

        for miner in all_miners:
            # Get miner's registration and deregistration days
            registration_day = self.miner_stats[miner]['registration_day']
            deregistration_day = self.miner_stats[miner].get('deregistration_day', self.simulation.num_days - 1)

            # Get cumulative weights for this miner
            cumulative_weights = self.get_miner_cumulative_weights(miner)

            # Create x-axis (days) and y-axis (cumulative weights)
            days = np.arange(registration_day, deregistration_day + 1)
            weights = cumulative_weights[registration_day:deregistration_day + 1]

            # Plot the cumulative weights line
            fig.add_trace(go.Scatter(
                x=days,
                y=weights,
                mode='lines',
                name=f'Miner {miner}',
                showlegend=False
            ))

            # Add registration day marker (green)
            fig.add_trace(go.Scatter(
                x=[registration_day],
                y=[weights[0]],
                mode='markers',
                marker=dict(color='green', size=10),
                name=f'Miner {miner} Registration',
                showlegend=False
            ))

            # Add deregistration day marker (red) if applicable
            if deregistration_day < self.simulation.num_days - 1:
                fig.add_trace(go.Scatter(
                    x=[deregistration_day],
                    y=[weights[-1]],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name=f'Miner {miner} Deregistration',
                    showlegend=False
                ))

        fig.update_layout(
            title='All Miners Cumulative Weights',
            xaxis_title='Day',
            yaxis_title='Cumulative Weight',
            showlegend=False
        )

        fig.show()
        print("All miners cumulative weights plot displayed.")
        logger.info("All miners cumulative weights plot displayed.")

    def get_miner_cumulative_weights(self, miner):
        # Initialize an array to store cumulative weights
        cumulative_weights = np.zeros(self.simulation.num_days)

        # Get the miner's unique ID
        unique_id = self.simulation.miner_registry.get_unique_id(miner)

        if unique_id in self.scoring_system.cumulative_weights:
            miner_data = self.scoring_system.cumulative_weights[unique_id]
            start_day = miner_data['start_day']
            weights = miner_data['weights']

            # Calculate cumulative weights
            for day, weight in weights:
                if day < self.simulation.num_days:
                    cumulative_weights[day:] += weight

        return cumulative_weights