import torch as t
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from utils import logger
from miner_registry import MinerRegistry
import traceback

class ScoringSystem:
    def __init__(self, max_capacity, max_days, temperature=1.0):
        self.max_capacity = max_capacity
        self.max_days = max_days
        
        # Fixed-size tensors for active miners
        self.clv_scores = t.zeros(max_capacity, max_days)
        self.sortino_scores = t.zeros(max_capacity, max_days)
        self.sharpe_scores = t.zeros(max_capacity, max_days)
        self.roi_scores = t.zeros(max_capacity, max_days)
        self.num_predictions = t.zeros(max_capacity, max_days, dtype=t.int64)
        self.tiers = t.ones(max_capacity, dtype=t.long)
        self.composite_scores = t.zeros((max_capacity, max_days))
        self.tier_history = t.ones((max_capacity, max_days), dtype=t.long)
        self.profits = t.zeros((max_capacity, max_days))

        # Dynamic data structure for all miners
        self.all_miners_data = {}

        self.clv_weight = 0.25
        self.roi_weight = 0.25
        self.ssi_weight = 0.25
        
        self.tier_configs = [
            {'capacity': int(max_capacity * 0.4), 'window': 3, 'min_predictions': 0, 'incentive': 0.1},  # Tier 1: 40%
            {'capacity': int(max_capacity * 0.25), 'window': 7, 'min_predictions': 15, 'incentive': 0.15},  # Tier 2: 25%
            {'capacity': int(max_capacity * 0.15), 'window': 15, 'min_predictions': 40, 'incentive': 0.2},  # Tier 3: 15%
            {'capacity': int(max_capacity * 0.1), 'window': 30, 'min_predictions': 75, 'incentive': 0.25},  # Tier 4: 10%
            {'capacity': int(max_capacity * 0.1), 'window': 45, 'min_predictions': 110, 'incentive': 0.3}  # Tier 5: 10%
        ]

        self.miner_registry = MinerRegistry(max_capacity)
        self.cumulative_predictions = t.zeros(max_capacity, dtype=t.int64)
        self.temperature = temperature

        self.miner_stats = {}  # To store miner statistics for analysis
        self.swap_history = []  # Add this to track swaps

    def update_cumulative_predictions(self, miner_id, daily_predictions):
        miner_id = self._ensure_capacity(miner_id)
        self.cumulative_predictions[miner_id] += daily_predictions

    def update_profits(self, miner, day, profit):
        miner_id = miner if isinstance(miner, int) else miner.item()
        miner_id = self._ensure_capacity(miner_id)
        if miner_id < self.max_capacity and day < self.max_days:
            self.profits[miner_id, day] = profit
        else:
            logger.warning(f"Attempted to update profits for invalid miner {miner_id} or day {day}")

    def get_profits(self):
        return self.profits

    def get_cumulative_predictions(self):
        return self.cumulative_predictions

    def update_miner_daily_scores(self, miner, day, predictions, closing_line_odds, results, debug: bool = False):
        miner_id = miner.item() if isinstance(miner, t.Tensor) else miner
        miner_id = self._ensure_capacity(miner_id)
        if self.miner_registry.is_miner_registered(miner_id):
            if debug:
                logger.debug(f"Miner {miner_id} - Day {day}")
                logger.debug(f"Predictions shape: {predictions.shape}")
                logger.debug(f"Closing line odds shape: {closing_line_odds.shape}")
                logger.debug(f"Results shape: {results.shape}")

            try:
                clv = self.calculate_clv(predictions, closing_line_odds, debug=debug)
                roi = self.calculate_roi(predictions, results, debug=debug)
                sharpe = self.calculate_sharpe(predictions, results, debug=debug)
                sortino = self.calculate_sortino(predictions, results, debug=debug)

                logger.debug(f"Miner {miner_id} - Day {day} scores: CLV={clv:.4f}, ROI={roi:.4f}, Sharpe={sharpe:.4f}, Sortino={sortino:.4f}")

                self.clv_scores[miner_id, day] = clv
                self.roi_scores[miner_id, day] = roi
                self.sharpe_scores[miner_id, day] = sharpe
                self.sortino_scores[miner_id, day] = sortino
                self.composite_scores[miner_id, day] = (clv + roi + sharpe + sortino) / 4
                self.num_predictions[miner_id, day] = len(predictions)

                logger.debug(f"Miner {miner_id} - Day {day} composite score: {self.composite_scores[miner_id, day]:.4f}")

            except Exception as e:
                logger.error(f"Error calculating scores for Miner {miner_id} on Day {day}: {str(e)}")
                logger.error(traceback.format_exc())

        # Update dynamic data structure
        if miner_id not in self.all_miners_data:
            self.all_miners_data[miner_id] = {'scores': [], 'predictions': []}
        self.all_miners_data[miner_id]['scores'].append({
            'day': day,
            'clv': self.clv_scores[miner_id, day].item(),
            'roi': self.roi_scores[miner_id, day].item(),
            'sharpe': self.sharpe_scores[miner_id, day].item(),
            'sortino': self.sortino_scores[miner_id, day].item(),
            'composite': self.composite_scores[miner_id, day].item()
        })
        self.all_miners_data[miner_id]['predictions'].append(len(predictions))

    def calculate_clv(self, predictions: t.Tensor, closing_line_odds: t.Tensor, debug: bool = False) -> float:
        try:
            # Convert American odds to European (decimal) odds
            predicted_probs = 1 / (predictions[:, 2].float().abs() / 100 + 1)  # Use abs to avoid division by zero
            
            # Ensure we're using the correct number of closing line odds
            num_predictions = predictions.shape[0]
            closing_probs = 1 / (closing_line_odds[:num_predictions, 0].float().abs() / 100 + 1)
            
            # Ensure the tensors have the same shape
            predicted_probs = predicted_probs[:num_predictions]
            closing_probs = closing_probs[:num_predictions]
            
            clv = (predicted_probs - closing_probs).mean().item()
            
            if debug:
                logger.debug(f"CLV: {clv:.4f}")
                logger.debug(f"Predictions shape: {predictions.shape}, Closing odds shape: {closing_line_odds.shape}")
                logger.debug(f"Predicted probs shape: {predicted_probs.shape}, Closing probs shape: {closing_probs.shape}")
            
            return clv
        except Exception as e:
            logger.error(f"Error in calculate_clv: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0

    def calculate_roi(self, predictions: t.Tensor, results: t.Tensor, debug: bool = False) -> float:
        try:
            # Convert American odds to decimal odds
            decimal_odds = t.where(predictions[:, 2] > 0, 
                                   predictions[:, 2].float() / 100 + 1, 
                                   100 / (-predictions[:, 2].float()) + 1)
            
            # Calculate returns: (odds - 1) for wins, -1 for losses
            returns = t.where(predictions[:, 1] == results, 
                              decimal_odds - 1, 
                              t.tensor(-1.0))
            
            # Calculate ROI
            roi = returns.mean().item()
            
            if debug:
                logger.debug(f"ROI: {roi:.4f}, Max return: {returns.max().item():.4f}, Min return: {returns.min().item():.4f}")
                logger.debug(f"Correct predictions: {(predictions[:, 1] == results).sum().item()} out of {len(results)}")
            
            return roi
        except Exception as e:
            logger.error(f"Error in calculate_roi: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0

    def calculate_sharpe(self, predictions: t.Tensor, results: t.Tensor, debug: bool = False) -> float:
        try:
            returns = t.where(predictions[:, 1] == results, predictions[:, 2].float() - 1, t.tensor(-1.0))
            returns_mean = returns.mean()
            returns_std = returns.std()
            risk_free_rate = 0.02 / 365  # Assuming a 2% annual risk-free rate

            if returns_std == 0:
                if returns_mean > risk_free_rate:
                    return 10.0  # Max Sharpe ratio for positive returns with no volatility
                elif returns_mean < risk_free_rate:
                    return -10.0  # Min Sharpe ratio for negative returns with no volatility
                else:
                    return 0.0  # Neutral Sharpe ratio when return equals risk-free rate

            sharpe = (returns_mean - risk_free_rate) / returns_std

            # Clip Sharpe ratio to a reasonable range
            sharpe = t.clamp(sharpe, -10, 10)

            if debug:
                logger.debug(f"Sharpe calculation - Mean return: {returns_mean:.6f}, Std dev: {returns_std:.6f}")
                logger.debug(f"Sharpe ratio (before clamping): {sharpe.item():.6f}")

            return sharpe.item()
        except Exception as e:
            logger.error(f"Error in calculate_sharpe: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0

    def calculate_sortino(self, predictions: t.Tensor, results: t.Tensor, debug: bool = False) -> float:
        try:
            roi = self.calculate_roi(predictions, results, debug)
            returns = t.where(predictions[:, 1] == results, predictions[:, 2].float() - 1, t.tensor(-1.0))
            avg_return = returns.mean()
            downside_returns = returns[returns < avg_return] - avg_return
            
            risk_free_rate = 0.02 / 365  # Assuming a 2% annual risk-free rate
            
            if debug:
                logger.debug(f"Sortino calculation - ROI: {roi:.6f}, Risk-free rate: {risk_free_rate:.6f}")
                logger.debug(f"Total returns: {len(returns)}, Downside returns: {len(downside_returns)}")
            
            if len(downside_returns) > 0:
                downside_std = t.sqrt(t.mean(downside_returns**2))
                
                if downside_std > 0:
                    sortino = (roi - risk_free_rate) / downside_std
                    clamped_sortino = t.clamp(sortino, -10, 10).item()
                    
                    if debug:
                        logger.debug(f"Downside STD: {downside_std:.6f}")
                        logger.debug(f"Sortino ratio: {clamped_sortino:.6f}")
                    
                    return clamped_sortino
                else:
                    logger.warning("Downside STD is zero, but there are downside returns. This should not happen.")
            
            # Handle edge cases
            if roi > risk_free_rate:
                return 10.0
            elif roi < risk_free_rate:
                return -10.0
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Error in calculate_sortino: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0

    def count_predictions(self, predictions: t.Tensor, debug: bool = False) -> t.Tensor:
        if debug:
            if not isinstance(predictions, t.Tensor):
                raise TypeError("predictions must be a PyTorch tensor")
            
            if predictions.dim() != 3 or predictions.size(2) != 3:
                raise ValueError("predictions must have shape (num_miners, num_predictions, 3)")        
        return (predictions[:, :, 2] > 0).sum(dim=1)

    def log_tier_counts_and_capacities(self):
        # Ensure tiers are non-negative integers
        valid_tiers = t.clamp(self.tiers, min=0, max=5).long()
        tier_counts = t.bincount(valid_tiers, minlength=6)[1:]  # Start from index 1 and ignore index 0
        logger.info("Tier counts and capacities:")
        for tier, count in enumerate(tier_counts, start=1):
            capacity = int(self.tier_configs[tier - 1]['capacity'])
            logger.info(f"  Tier {tier}: {count.item()}/{capacity}")

    def manage_tiers(self, day):
        logger.info(f"Managing tiers for day {day}")
        active_miners = self.miner_registry.get_active_miners()
        adjusted_miners = [self._ensure_capacity(miner) for miner in active_miners]
        current_tiers = self.tiers[adjusted_miners]
        
        immune_miners = sum(1 for miner in adjusted_miners if day - self.miner_registry.get_registration_day(miner) <= 3)
        logger.info(f"Miners in immunity period: {immune_miners}")
        
        next_tiers = self.check_demotion_eligibility(day, current_tiers)
        next_tiers = self.check_promotion_eligibility(day, next_tiers)
        
        # Ensure miners aren't promoted too quickly
        for miner in adjusted_miners:
            registration_day = self.miner_registry.get_registration_day(miner)
            days_since_registration = day - registration_day
            max_tier = min(5, 1 + days_since_registration // 9)  # 45 days minimum to reach tier 5
            next_tiers[adjusted_miners.index(miner)] = min(next_tiers[adjusted_miners.index(miner)], max_tier)
        
        # Enforce tier capacities and implement demotion cascade
        for tier in range(5, 1, -1):  # Change this line to start from 5 and go down to 2
            tier_capacity = int(self.tier_configs[tier - 1]['capacity'])
            tier_miners = [m for m, t in zip(adjusted_miners, next_tiers) if t == tier]
            
            if len(tier_miners) > tier_capacity:
                # Sort miners by their composite scores in descending order
                sorted_miners = sorted(tier_miners, key=lambda m: self.composite_scores[m, max(0, day-30):day+1].mean(), reverse=True)
                
                # Keep top performers in the tier, demote others
                for miner in sorted_miners[tier_capacity:]:
                    old_tier = next_tiers[adjusted_miners.index(miner)]
                    new_tier = max(tier - 1, 1)  # Ensure we don't demote below Tier 1
                    miner_score = self.composite_scores[miner, max(0, day-30):day+1].mean().item()
                    
                    # Check if the miner's score is better than the lowest score in the new tier
                    new_tier_miners = [m for m, t in zip(adjusted_miners, next_tiers) if t == new_tier]
                    if new_tier_miners:
                        lowest_score_new_tier = min(self.composite_scores[m, max(0, day-30):day+1].mean().item() for m in new_tier_miners)
                        if miner_score > lowest_score_new_tier:
                            # Demote the lowest-scoring miner in the new tier instead
                            lowest_miner = min(new_tier_miners, key=lambda m: self.composite_scores[m, max(0, day-30):day+1].mean().item())
                            next_tiers[adjusted_miners.index(lowest_miner)] = new_tier - 1
                            logger.info(f"Day {day}: Demoting miner {lowest_miner} from Tier {new_tier} to Tier {new_tier-1}. Score: {self.composite_scores[lowest_miner, max(0, day-30):day+1].mean().item():.4f}")
                            self.swap_history.append((day, lowest_miner, new_tier, new_tier-1))
                        else:
                            next_tiers[adjusted_miners.index(miner)] = new_tier
                            logger.info(f"Day {day}: Demoting miner {miner} from Tier {old_tier} to Tier {new_tier}. Score: {miner_score:.4f}")
                            self.swap_history.append((day, miner, old_tier, new_tier))
                    else:
                        next_tiers[adjusted_miners.index(miner)] = new_tier
                        logger.info(f"Day {day}: Demoting miner {miner} from Tier {old_tier} to Tier {new_tier}. Score: {miner_score:.4f}")
                        self.swap_history.append((day, miner, old_tier, new_tier))
        
        # Update tiers for active miners
        self.tiers[adjusted_miners] = t.clamp(next_tiers, min=1, max=5)
        self.tier_history[adjusted_miners, day] = t.clamp(next_tiers, min=1, max=5)
        
        self.log_tier_counts_and_capacities()
        logger.info(f"Tier distribution after management: {self.get_tier_distribution()}")

    def check_demotion_eligibility(self, day, current_tiers):
        logger.info("Checking demotion eligibility")
        next_tiers = current_tiers.clone()
        active_miners = self.miner_registry.get_active_miners()
        adjusted_miners = [self._ensure_capacity(miner) for miner in active_miners]
        
        for tier, config in enumerate(self.tier_configs[1:], 2):  # Start from tier 2
            window = config['window']
            min_predictions = config['min_predictions']
            
            tier_mask = current_tiers == tier
            if not tier_mask.any():
                continue
            
            start_day = max(0, day - window + 1)
            prediction_counts = t.zeros(len(adjusted_miners))
            for i, miner in enumerate(adjusted_miners):
                if tier_mask[i]:
                    prediction_counts[i] = self.num_predictions[miner, start_day:day + 1].sum()
            
            demotion_mask = (prediction_counts < min_predictions) & tier_mask
            if demotion_mask.any():
                next_tiers[demotion_mask] = tier - 1
                logger.info(f"Demoted {demotion_mask.sum().item()} miners from Tier {tier} to Tier {tier-1}")
                logger.info(f"  Due to low predictions: {demotion_mask.sum().item()}")
        
        logger.info(f"Tiers after demotion check: {next_tiers.tolist()}")
        return next_tiers

    def check_promotion_eligibility(self, day, current_tiers):
        logger.info("Checking promotion eligibility")
        next_tiers = current_tiers.clone()
        active_miners = self.miner_registry.get_active_miners()
        adjusted_miners = [self._ensure_capacity(miner) for miner in active_miners]
        
        logger.info(f"Current tier distribution: {t.bincount(current_tiers)}")
        
        for tier, config in enumerate(self.tier_configs[:-1], 1):  # Exclude the highest tier
            next_tier = tier + 1
            next_tier_config = self.tier_configs[next_tier - 1]
            window = next_tier_config['window']
            min_predictions = next_tier_config['min_predictions']
            next_tier_capacity = int(next_tier_config['capacity'])
            
            tier_mask = current_tiers == tier
            next_tier_mask = current_tiers == next_tier
            
            start_day = max(0, day - window + 1)
            window_scores = t.zeros(len(adjusted_miners))
            prediction_counts = t.zeros(len(adjusted_miners))
            
            for i, miner in enumerate(adjusted_miners):
                if tier_mask[i] or next_tier_mask[i]:
                    window_scores[i] = self.composite_scores[miner, start_day:day + 1].mean()
                    prediction_counts[i] = self.num_predictions[miner, start_day:day + 1].sum()
            
            promotion_mask = (prediction_counts >= min_predictions) & (window_scores > 0) & tier_mask
            
            current_next_tier_count = (next_tiers == next_tier).sum()
            available_spots = max(0, next_tier_capacity - current_next_tier_count)
            
            logger.info(f"Tier {tier} to {next_tier}:")
            logger.info(f"  Miners in current tier: {tier_mask.sum().item()}")
            logger.info(f"  Miners in next tier: {next_tier_mask.sum().item()}")
            logger.info(f"  Miners eligible for promotion: {promotion_mask.sum().item()}")
            logger.info(f"  Available spots in next tier: {available_spots}")
            
            if promotion_mask.sum() > 0:
                if promotion_mask.sum() > available_spots:
                    # Sort eligible miners by their scores
                    eligible_miners = t.tensor(adjusted_miners)[promotion_mask]
                    eligible_scores = window_scores[promotion_mask]
                    sorted_indices = t.argsort(eligible_scores, descending=True)
                    promoted_miners = eligible_miners[sorted_indices[:available_spots]]
                    
                    # Update next_tiers for promoted miners
                    for miner in promoted_miners:
                        next_tiers[adjusted_miners.index(miner)] = next_tier
                    
                    logger.info(f"  Promoted {len(promoted_miners)} miners from Tier {tier} to Tier {next_tier}")
                else:
                    next_tiers[promotion_mask] = next_tier
                    logger.info(f"  Promoted {promotion_mask.sum().item()} miners from Tier {tier} to Tier {next_tier}")
            else:
                logger.info(f"  No miners eligible for promotion from Tier {tier} to Tier {next_tier}")
            
            # Check for potential swaps
            if current_next_tier_count >= next_tier_capacity:
                current_tier_scores = window_scores[tier_mask]
                next_tier_scores = window_scores[next_tier_mask]
                
                potential_swaps = (current_tier_scores > next_tier_scores.min()).sum()
                logger.info(f"  Potential swaps: {potential_swaps}")
                
                if potential_swaps > 0:
                    num_swaps = min(potential_swaps, next_tier_capacity // 10)  # Swap up to 10% of the next tier capacity
                    logger.info(f"  Swapping {num_swaps} miners between Tier {tier} and Tier {next_tier}")
                    
                    _, top_current_indices = t.topk(current_tier_scores, k=num_swaps)
                    _, bottom_next_indices = t.topk(next_tier_scores, k=num_swaps, largest=False)
                    
                    current_miners_to_promote = t.tensor(adjusted_miners)[tier_mask][top_current_indices]
                    next_tier_miners_to_demote = t.tensor(adjusted_miners)[next_tier_mask][bottom_next_indices]
                    
                    for promote, demote in zip(current_miners_to_promote, next_tier_miners_to_demote):
                        logger.info(f"    Swap: Promoting miner {promote} (score: {window_scores[promote]:.4f}) "
                                    f"and demoting miner {demote} (score: {window_scores[demote]:.4f})")
                    
                    next_tiers[current_miners_to_promote] = next_tier
                    next_tiers[next_tier_miners_to_demote] = tier
                else:
                    logger.info(f"  No swaps performed between Tier {tier} and Tier {next_tier}")
            else:
                logger.info(f"  No swaps possible: next tier not at capacity")
        
        logger.info(f"Tiers after promotion check: {next_tiers.tolist()}")
        return next_tiers

    def verify_tier_capacities(self):
        for tier, config in enumerate(self.tier_configs, 1):
            capacity = int(config['capacity'])
            tier_miners = (self.tiers == tier).nonzero().squeeze()
            if tier_miners.dim() == 0:
                tier_count = 1 if tier_miners.item() else 0
            else:
                tier_count = tier_miners.size(0)
            if tier_count > capacity:
                logger.warning(f"Tier {tier} has {tier_count} miners, exceeding capacity of {capacity}")

    def get_tier_distribution(self):
        return t.bincount(self.tiers, minlength=6)[1:]  # Start from index 1 and ignore index 0

    def get_tier_history(self):
        return self.tier_history

    def get_miner_scores(self, miner_id):
        return self.all_miners_data.get(miner_id, {})

    def get_miner_predictions(self, miner_id):
        return self.all_miners_data.get(miner_id, {}).get('predictions', [])

    def get_miner_tier_history(self, miner_id):
        return self.tier_history[miner_id].long()  # Ensure we return long (integer) values

    def get_miner_tier(self, miner_id):
        return self.tiers[miner_id].item()

    def get_miner_profits(self, miner_id):
        return self.profits[miner_id]

    def get_miner_cumulative_predictions(self, miner_id):
        return self.cumulative_predictions[miner_id]

    def get_miner_clv_scores(self, miner_id):
        return self.clv_scores[miner_id]

    def get_miner_roi_scores(self, miner_id):
        return self.roi_scores[miner_id]

    def get_miner_sharpe_scores(self, miner_id):
        return self.sharpe_scores[miner_id]

    def get_miner_sortino_scores(self, miner_id):
        return self.sortino_scores[miner_id]

    def get_miner_composite_scores(self, miner_id):
        return self.composite_scores[miner_id]

    def get_miner_num_predictions(self, miner_id):
        return self.num_predictions[miner_id]

    def get_miner_registry(self):
        return self.miner_registry

    def get_tier_configs(self):
        return self.tier_configs

    def get_temperature(self):
        return self.temperature

    def get_clv_weight(self):
        return self.clv_weight

    def get_roi_weight(self):
        return self.roi_weight

    def get_ssi_weight(self):
        return self.ssi_weight

    def get_max_capacity(self):
        return self.max_capacity

    def get_max_days(self):
        return self.max_days

    def get_all_miners_data(self):
        return self.all_miners_data

    def set_tier_configs(self, tier_configs):
        self.tier_configs = tier_configs

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_clv_weight(self, clv_weight):
        self.clv_weight = clv_weight

    def set_roi_weight(self, roi_weight):
        self.roi_weight = roi_weight

    def set_ssi_weight(self, ssi_weight):
        self.ssi_weight = ssi_weight

    def set_max_capacity(self, max_capacity):
        self.max_capacity = max_capacity

    def set_max_days(self, max_days):
        self.max_days = max_days

    def set_all_miners_data(self, all_miners_data):
        self.all_miners_data = all_miners_data

    def _ensure_capacity(self, miner_id):
        if miner_id >= self.max_capacity:
            logger.warning(f"Miner ID {miner_id} exceeds maximum capacity of {self.max_capacity}. Adjusting to fit.")
            return miner_id % self.max_capacity
        return miner_id

    def calculate_final_weights(self, day):
        active_miners = self.miner_registry.get_active_miners()
        weights = t.zeros(self.max_capacity)
        valid_miners = [miner for miner in active_miners if miner < self.max_capacity]
        
        logger.info(f"Calculating weights for day {day}")
        logger.info(f"Number of active miners: {len(active_miners)}")
        logger.info(f"Number of valid miners: {len(valid_miners)}")
        
        total_incentive = sum(config['incentive'] for config in self.tier_configs)
        
        for tier in range(5, 0, -1):
            tier_miners = [m for m in valid_miners if self.tiers[m] == tier]
            tier_config = self.tier_configs[tier - 1]
            tier_capacity = int(tier_config['capacity'])
            tier_incentive = tier_config['incentive'] / total_incentive
            
            logger.info(f"Tier {tier}: {len(tier_miners)}/{tier_capacity} miners")
            
            if tier_miners:
                window = tier_config['window']
                start_day = max(0, day - window + 1)
                tier_scores = self.composite_scores[tier_miners, start_day:day+1].mean(dim=1)
                tier_scores = t.nan_to_num(tier_scores, nan=0.0)
                logger.info(f"Tier {tier} scores: min={tier_scores.min().item():.6f}, max={tier_scores.max().item():.6f}, mean={tier_scores.mean().item():.6f}")
                
                # Normalize scores within the tier
                normalized_scores = F.softmax(tier_scores, dim=0)
                
                # Allocate weights within the tier
                tier_total_weight = tier_incentive * min(len(tier_miners), tier_capacity) / tier_capacity
                for i, miner in enumerate(tier_miners):
                    weights[miner] = tier_total_weight * normalized_scores[i].item()
        
        # Ensure weights sum to 1
        weights /= weights.sum()
        
        # Update cumulative weights
        self.update_cumulative_weights(day, weights)
        
        # Log weight distribution by tier
        logger.info(f"Weight distribution by tier (Day {day}):")
        for tier in range(5, 0, -1):
            tier_miners = [m for m in valid_miners if self.tiers[m] == tier]
            tier_capacity = int(self.tier_configs[tier - 1]['capacity'])
            if tier_miners:
                tier_weights = weights[tier_miners]
                min_weight = tier_weights.min().item()
                max_weight = tier_weights.max().item()
                mean_weight = tier_weights.mean().item()
                total_tier_weight = tier_weights.sum().item()
                logger.info(f"  Tier {tier}: Min = {min_weight:.6f}, Max = {max_weight:.6f}, Mean = {mean_weight:.6f}, "
                            f"Total = {total_tier_weight:.6f}, Miners = {len(tier_miners)}/{tier_capacity}")
            else:
                logger.info(f"  Tier {tier}: No miners (Capacity: {tier_capacity})")
        
        return weights

    def update_cumulative_weights(self, day, weights):
        if not hasattr(self, 'cumulative_weights'):
            self.cumulative_weights = {}
        
        for network_uid, weight in enumerate(weights):
            unique_id = self.miner_registry.get_unique_id(network_uid)
            if unique_id is not None:
                if unique_id not in self.cumulative_weights:
                    registration_day = self.miner_registry.get_registration_day(network_uid)
                    self.cumulative_weights[unique_id] = {'start_day': registration_day, 'weights': []}
                
                # Only add weight if the current day is not before the registration day
                if day >= self.cumulative_weights[unique_id]['start_day']:
                    self.cumulative_weights[unique_id]['weights'].append((day, weight.item()))

    def run_analysis(self, start_day, end_day):
        analysis_start_day = max(start_day, 30)  # Start analysis from day 30 or later
        
        # Perform analysis only for the final day
        self.analyze_performance_metrics(end_day)
        self.analyze_tier_movement(analysis_start_day, end_day)
        self.analyze_score_distribution(end_day)
        self.analyze_prediction_count_distribution(end_day)
        self.analyze_tier_stability(analysis_start_day, end_day)
        self.plot_miner_scores(num_miners=20)
        self.plot_tier_changes(num_miners=20)
        self.plot_weights_over_time()
        self.plot_cumulative_weights()

    def analyze_performance_metrics(self, day):
        metrics = {
            'CLV': self.clv_scores[:, day].numpy(),
            'ROI': self.roi_scores[:, day].numpy(),
            'Sharpe': self.sharpe_scores[:, day].numpy(),
            'Sortino': self.sortino_scores[:, day].numpy()
        }
        tiers = self.tier_history[:, day].numpy()

        for metric, values in metrics.items():
            non_zero_values = values[values != 0]
            if len(non_zero_values) > 0:
                logger.info(f"{metric} stats - Mean: {np.mean(non_zero_values):.4f}, "
                                 f"Median: {np.median(non_zero_values):.4f}, "
                                 f"Min: {np.min(non_zero_values):.4f}, "
                                 f"Max: {np.max(non_zero_values):.4f}, "
                                 f"Non-zero count: {len(non_zero_values)}")
            else:
                logger.warning(f"No non-zero values for {metric}.")

    def log_score_summary(self, day):
        logger.info(f"Score summary for day {day}:")
        for score_type, scores in [
            ("CLV", self.clv_scores[:, day]),
            ("ROI", self.roi_scores[:, day]),
            ("Sharpe", self.sharpe_scores[:, day]),
            ("Sortino", self.sortino_scores[:, day])
        ]:
            finite_scores = scores[t.isfinite(scores)]
            if finite_scores.numel() > 0:
                logger.info(f"{score_type} scores - Mean: {finite_scores.mean():.4f}, Max: {finite_scores.max():.4f}")
            else:
                logger.info(f"{score_type} scores - No finite values")

    def analyze_tier_movement(self, start_day, end_day):
        tier_history = self.tier_history[:, start_day:end_day]
        
        # Count transitions
        transitions = {}
        for i in range(tier_history.shape[1] - 1):
            for from_tier in range(1, len(self.tier_configs) + 1):
                for to_tier in range(1, len(self.tier_configs) + 1):
                    key = (from_tier, to_tier)
                    count = ((tier_history[:, i] == from_tier) & (tier_history[:, i+1] == to_tier)).sum().item()
                    transitions[key] = transitions.get(key, 0) + count

        # Create transition matrix
        matrix = np.zeros((len(self.tier_configs), len(self.tier_configs)))
        for (from_tier, to_tier), count in transitions.items():
            matrix[from_tier-1, to_tier-1] = count

        # Normalize
        row_sums = matrix.sum(axis=1)
        epsilon = 1e-8  # Small value to avoid division by zero
        matrix_norm = matrix / (row_sums[:, np.newaxis] + epsilon)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix_norm,
            x=[f'Tier {i+1}' for i in range(len(self.tier_configs))],
            y=[f'Tier {i+1}' for i in range(len(self.tier_configs))],
            colorscale='YlGnBu',
            text=matrix_norm,
            texttemplate="%{text:.2f}",
            textfont={"size":10},
        ))
        fig.update_layout(
            title='Tier Transition Probabilities',
            xaxis_title='To Tier',
            yaxis_title='From Tier'
        )
        pio.show(fig)

    def analyze_score_distribution(self, day):
        scores = self.composite_scores[:, day].numpy()
        tiers = self.tier_history[:, day].numpy()
        
        df = pd.DataFrame({'Tier': tiers, 'Score': scores})
        fig = px.box(df, x='Tier', y='Score', title=f'Score Distribution per Tier (Day {day})')
        pio.show(fig)

    def analyze_prediction_count_distribution(self, day):
        prediction_counts = self.num_predictions[:, max(0, day - 30):day].sum(dim=1).numpy()
        tiers = self.tier_history[:, day].numpy()
        
        df = pd.DataFrame({'Tier': tiers, 'Prediction Count': prediction_counts})
        fig = px.box(df, x='Tier', y='Prediction Count', 
                     title=f'30-Day Prediction Count Distribution per Tier (Day {day})')
        pio.show(fig)

    def analyze_tier_stability(self, start_day, end_day):
        tier_history = self.tier_history[:, start_day:end_day].numpy()
        
        stability_scores = []
        for miner in range(tier_history.shape[0]):
            changes = (tier_history[miner, 1:] != tier_history[miner, :-1]).sum()
            stability_scores.append(1 - (changes / (end_day - start_day)))

        fig = px.histogram(x=stability_scores, nbins=30, 
                           title='Tier Stability Scores',
                           labels={'x': 'Stability Score (1 = No Changes, 0 = Changes Every Day)', 'y': 'Count'})
        pio.show(fig)

    def plot_miner_scores(self, miner_ids=None, num_miners=5, score_types=None):
        if score_types is None:
            score_types = ['composite', 'clv', 'roi', 'sharpe', 'sortino']
        
        if miner_ids is None:
            # If no specific miners are requested, choose top performers
            latest_scores = self.composite_scores[:, -1]
            top_miners = latest_scores.argsort(descending=True)[:num_miners]
            miner_ids = top_miners.tolist()
        
        num_rows = len(score_types)
        fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=[f'{score_type.capitalize()} Scores' for score_type in score_types])
        
        for i, score_type in enumerate(score_types, start=1):
            for miner_id in miner_ids:
                if score_type == 'composite':
                    scores = self.composite_scores[miner_id, :].cpu().numpy()
                elif score_type == 'clv':
                    scores = self.clv_scores[miner_id, :].cpu().numpy()
                elif score_type == 'roi':
                    scores = self.roi_scores[miner_id, :].cpu().numpy()
                elif score_type == 'sharpe':
                    scores = self.sharpe_scores[miner_id, :].cpu().numpy()
                elif score_type == 'sortino':
                    scores = self.sortino_scores[miner_id, :].cpu().numpy()
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
        
        pio.show(fig)

    def plot_tier_changes(self, miner_ids=None, num_miners=5):
        if miner_ids is None:
            # If no specific miners are requested, choose miners with most tier changes
            tier_changes = (self.tier_history[:, 1:] != self.tier_history[:, :-1]).sum(dim=1)
            top_changers = tier_changes.argsort(descending=True)[:num_miners]
            miner_ids = top_changers.tolist()
        
        fig = go.Figure()
        
        for miner_id in miner_ids:
            tier_history = self.tier_history[miner_id, :].cpu().numpy()
            fig.add_trace(go.Scatter(x=list(range(len(tier_history))), y=tier_history, 
                                     name=f'Miner {miner_id}', mode='lines+markers'))
        
        fig.update_layout(title_text="Miner Tier Changes Over Time",
                          xaxis_title="Day",
                          yaxis_title="Tier",
                          yaxis_range=[0.5, 5.5],  # Assuming 5 tiers
                          yaxis_tickvals=[1, 2, 3, 4, 5])
        
        pio.show(fig)

    def plot_weights_over_time(self):
        start_day = 30  # Start from day 30
        weights_over_time = []
        for day in range(start_day, self.max_days):
            weights = self.calculate_final_weights(day)
            weights_over_time.append(weights.numpy())
        
        weights_array = np.array(weights_over_time)
        
        fig = go.Figure()
        for miner in range(weights_array.shape[1]):
            fig.add_trace(go.Scatter(y=weights_array[:, miner], mode='lines', name=f'Miner {miner}'))
        
        fig.update_layout(title='Miner Weights Over Time',
                          xaxis_title='Day',
                          yaxis_title='Weight')
        pio.show(fig)

    def plot_cumulative_weights(self):
        fig = go.Figure()
        
        for unique_id, data in self.cumulative_weights.items():
            start_day = data['start_day']
            days, weights = zip(*data['weights'])
            
            # Ensure we don't have negative days
            adjusted_days = [max(0, day - start_day) for day in days]
            
            cumulative_weights = np.cumsum(weights)
            fig.add_trace(go.Scatter(x=adjusted_days, y=cumulative_weights, mode='lines', name=f'Miner {unique_id}'))
        
        fig.update_layout(
            title='Cumulative Incentives (Weights) per Unique Miner',
            xaxis_title='Days Since Registration',
            yaxis_title='Cumulative Weight',
            legend_title='Unique Miner ID'
        )
        
        pio.show(fig)

    def reset_miner_scores(self, network_uid):
        """
        Reset all scores and data for a given network_uid.
        This is called when a new miner replaces an existing one.
        """
        network_uid = self._ensure_capacity(network_uid)
        current_tier = self.tiers[network_uid].item()
        logger.info(f"Resetting scores for miner: network_uid={network_uid}, current_tier={current_tier}")
        
        self.clv_scores[network_uid, :] = 0
        self.sortino_scores[network_uid, :] = 0
        self.sharpe_scores[network_uid, :] = 0
        self.roi_scores[network_uid, :] = 0
        self.num_predictions[network_uid, :] = 0
        self.composite_scores[network_uid, :] = 0
        self.tier_history[network_uid, :] = 1  # Reset to Tier 1
        self.profits[network_uid, :] = 0
        self.cumulative_predictions[network_uid] = 0
        if network_uid in self.all_miners_data:
            del self.all_miners_data[network_uid]
        
        # Reset the tier to 1 (assuming 1 is the lowest tier)
        self.tiers[network_uid] = 1

        # Give new miners a slight initial boost
        self.composite_scores[network_uid, -1] = 0.1

        logger.info(f"Reset scores for network_uid {network_uid}")