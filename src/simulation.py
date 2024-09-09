import random
import traceback
from tqdm.auto import tqdm
import torch as t
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from miner_profile import MinerProfile
from scoring_system import ScoringSystem
from registration_fee_manager import RegistrationFeeManager
from tau_simulator import TauSimulator
from utils import logger
from analytics import SimulationAnalytics

class Simulation:
    def __init__(self, max_capacity, num_days, return_generator, min_burn, max_burn, initial_fee, alpha=0.1, temperature=1.0):
        self.max_capacity = max_capacity
        self.num_days = num_days
        self.return_generator = return_generator
        self.min_burn = min_burn
        self.max_burn = max_burn
        self.initial_fee = initial_fee
        self.scoring_system = ScoringSystem(max_capacity=max_capacity, max_days=num_days, temperature=temperature)
        self.next_miner_id = 0
        self.all_miners = {}
        self.registration_fee_manager = RegistrationFeeManager(min_burn=min_burn, max_burn=max_burn, initial_fee=initial_fee, alpha=alpha)
        self.tau_simulator = TauSimulator(num_days=num_days)
        self.miner_registration_days = {}  # network_uid -> registration day
        self.unique_id_to_profile = {}  # unique_id -> MinerProfile
        self.miner_registry = self.scoring_system.miner_registry  # Add this line
        self.current_day = 0  # Add this line to keep track of the current day
        self.miner_stats = {}  # Add this to store miner statistics

    def run_simulation(self):
        for day in tqdm(range(self.num_days)):
            try:
                self.run_day(day)
            except Exception as e:
                logger.error(f"Error on day {day}: {e}")
                logger.error(traceback.format_exc())
                break
        
        print("Simulation completed. Running analytics...")
        logger.info("Simulation completed. Running analytics...")
        try:
            analytics = SimulationAnalytics(self)
            analytics.run_all_analytics()
        except Exception as e:
            print(f"Error during analytics: {str(e)}")
            logger.error(f"Error during analytics: {str(e)}")
        print("Analytics process finished. Check your browser for plots.")
        logger.info("Analytics process finished. Check your browser for plots.")

    def run_day(self, day):
        self.current_day = day  # Update the current day
        self.miner_registry.update_current_day(day)  # Update the day in MinerRegistry
        current_tau = self.tau_simulator.get_tau(day)
        current_fee = self.registration_fee_manager.get_current_fee()
        
        # Register new miners
        new_miners = self.register_new_miners(day, current_tau, current_fee)
        
        # Generate predictions
        predictions = self.generate_predictions(day)
        
        # Update miner profiles
        self.update_miner_profiles(day, predictions)
        
        # Update registration fee
        self.registration_fee_manager.update_fee(current_tau, len(new_miners), self.scoring_system.miner_registry.get_total_registered_miners(), day)
        
        # Update miner scores
        self.update_miner_scores(day, predictions)
        
        # Manage tiers
        self.scoring_system.manage_tiers(day)
        
        # Log tier distribution
        tier_distribution = self.scoring_system.get_tier_distribution()
        logger.info(f"Day {day} tier distribution: {tier_distribution}")
        
        # Update miner weights using ScoringSystem's method
        self.scoring_system.calculate_final_weights(day)
        self.update_miner_stats(day)  # Add this line

    def register_new_miners(self, day, current_tau, current_fee):
        new_miners = []
        attempts = 0
        max_attempts = self.max_capacity * 2
        total_registered = self.scoring_system.miner_registry.get_total_registered_miners()
        
        logger.info(f"Day {day}: Starting registration process. Current miners: {total_registered}/{self.max_capacity}")
        
        while len(new_miners) < self.max_capacity * 0.1 and attempts < max_attempts:
            miner_profile = MinerProfile.get_random_profile(self.max_capacity, self.num_days)
            prob_register = miner_profile.calculate_registration_probability(current_tau, current_fee, day)
            
            logger.debug(f"Registration probability: {prob_register:.4f}")
            
            if random.random() < prob_register:
                if total_registered < self.max_capacity:
                    network_uid, unique_id = self.scoring_system.miner_registry.register_miner()
                    if network_uid is not None:
                        self.unique_id_to_profile[unique_id] = miner_profile
                        self.miner_registration_days[network_uid] = day
                        new_miners.append(network_uid)
                        total_registered += 1
                        logger.info(f"New miner registered: network_uid={network_uid}, unique_id={unique_id}")
                        self.miner_stats[network_uid] = {
                            'registration_day': day,
                            'skill_level': miner_profile.profile_type,  # Use profile_type instead of skill_level
                            'deregistration_day': None
                        }
                else:
                    low_performer = self.find_low_performing_miner(day)
                    if low_performer is not None:
                        self.scoring_system.miner_registry.deregister_miner(low_performer)
                        network_uid, unique_id = self.scoring_system.miner_registry.register_miner()
                        if network_uid is not None:
                            self.unique_id_to_profile[unique_id] = miner_profile
                            self.miner_registration_days[network_uid] = day
                            new_miners.append(network_uid)
                            logger.info(f"Miner {low_performer} replaced by new miner: network_uid={network_uid}, unique_id={unique_id}")
                            # Reset scores for this network_uid
                            self.scoring_system.reset_miner_scores(network_uid)
                            self.miner_stats[network_uid] = {
                                'registration_day': day,
                                'skill_level': miner_profile.profile_type,  # Use profile_type instead of skill_level
                                'deregistration_day': None
                            }
                        else:
                            logger.warning("Failed to register new miner after deregistering low performer")
                    else:
                        logger.info("No low-performing miner found for replacement")
            attempts += 1
        
        logger.info(f"Day {day}: Registered {len(new_miners)} new miners after {attempts} attempts. Total miners: {total_registered}")
        if len(new_miners) == 0:
            logger.info(f"No new miners registered. Current tau: {current_tau:.4f}, Current fee: {current_fee:.4f}")
        return new_miners

    def find_low_performing_miner(self, day):
        active_miners = self.scoring_system.miner_registry.get_active_miners()
        if not active_miners:
            return None
        
        # Get the final weights for the current day
        weights = self.scoring_system.calculate_final_weights(day)
        
        # Filter out miners in their immunity period and ensure they are in valid tiers
        eligible_miners = [
            miner for miner in active_miners
            if day - self.miner_registration_days.get(miner, 0) > 3 and 1 <= self.scoring_system.tiers[miner] <= 5
        ]
        
        if not eligible_miners:
            return None
        
        # Find the miner with the lowest weight among eligible miners
        lowest_weight_miner = min(eligible_miners, key=lambda m: weights[m])
        
        logger.info(f"Lowest performing miner: {lowest_weight_miner}, "
                    f"tier: {self.scoring_system.tiers[lowest_weight_miner]}, "
                    f"weight: {weights[lowest_weight_miner]:.6f}")
        
        if lowest_weight_miner is not None:
            self.miner_stats[lowest_weight_miner]['deregistration_day'] = day
        
        return lowest_weight_miner

    def generate_predictions(self, day):
        predictions = []
        for network_uid, miner_profile in self.unique_id_to_profile.items():
            if self.scoring_system.miner_registry.is_miner_registered(network_uid):
                num_predictions = self.generate_num_predictions_for_miner(network_uid)
                miner_predictions = self.generate_predictions_for_miner(num_predictions)
                predictions.append((network_uid, miner_predictions))
        return predictions

    def generate_num_predictions_for_miner(self, network_uid):
        """
        Generate the number of predictions a miner will make for a day.
        """
        miner_profile = self.unique_id_to_profile[self.scoring_system.miner_registry.get_unique_id(network_uid)]
        base_predictions = 5  # Ensure a minimum number of predictions
        max_additional = 25   # Allow for more variation
        skill_factor = miner_profile.skill_factor
        
        # Use the skill factor to influence the number of predictions
        num_predictions = base_predictions + int(random.randint(0, int(max_additional * skill_factor)))
        
        return num_predictions

    def generate_predictions_for_miner(self, num_predictions):
        predictions = t.zeros((num_predictions, 3), dtype=t.int64)
        predictions[:, 0] = t.randint(0, 100, (num_predictions,))  # Game IDs
        predictions[:, 1] = t.randint(0, 2, (num_predictions,))  # Predicted outcomes (0 or 1)
        
        # Generate more realistic odds
        odds = t.randint(-200, 200, (num_predictions,))
        odds = t.where(odds == 0, t.tensor(100), odds)  # Avoid 0 odds
        predictions[:, 2] = odds
        
        return predictions

    def update_miner_profiles(self, day, predictions):
        for network_uid, miner_profile in self.unique_id_to_profile.items():
            if self.scoring_system.miner_registry.is_miner_registered(network_uid):
                miner_predictions = next((pred for uid, pred in predictions if uid == network_uid), None)
                if miner_predictions is not None:
                    num_predictions = miner_predictions.shape[0]
                    self.scoring_system.update_cumulative_predictions(network_uid, num_predictions)
                    profit = self.scoring_system.profits[network_uid, day].item() if day < self.scoring_system.profits.shape[1] else 0
                    miner_profile.update_strategy(profit)

    def update_miner_scores(self, day, predictions):
        debug = day < 5 or day % 10 == 0  # Log every 10 days and the first 5 days
        for network_uid, miner_predictions in predictions:
            num_predictions = miner_predictions.shape[0]
            closing_line_odds = t.randint(-200, 200, (num_predictions, 1))
            closing_line_odds = t.where(closing_line_odds == 0, t.tensor(100), closing_line_odds)
            
            # Generate results based on the odds, with a slight advantage for miners
            probabilities = 1 / (1 + t.abs(closing_line_odds.float()) / 100)
            probabilities = t.clamp(probabilities * 1.05, 0, 1)  # Give miners a 5% edge, but clamp to [0, 1]
            
            if debug:
                logger.debug(f"Miner {network_uid} - Day {day}")
                logger.debug(f"Probabilities min: {probabilities.min().item():.4f}, max: {probabilities.max().item():.4f}")
            
            results = t.bernoulli(probabilities).long().squeeze()
            
            self.scoring_system.update_miner_daily_scores(network_uid, day, miner_predictions, closing_line_odds, results, debug=debug)
            
            if debug:
                composite_score = self.scoring_system.composite_scores[network_uid, day].item()
                logger.info(f"Day {day}, Miner {network_uid}: Composite Score = {composite_score:.6f}")
            
            miner_profile = self.unique_id_to_profile[self.scoring_system.miner_registry.get_unique_id(network_uid)]
            miner_profile.update_strategy(self.scoring_system.profits[network_uid, day])

    def update_miner_stats(self, day):
        for network_uid in self.miner_registry.get_active_miners():
            if network_uid not in self.miner_stats:
                unique_id = self.miner_registry.get_unique_id(network_uid)
                miner_profile = self.unique_id_to_profile[unique_id]
                self.miner_stats[network_uid] = {
                    'registration_day': self.miner_registration_days[network_uid],
                    'skill_level': miner_profile.profile_type,
                    'deregistration_day': None
                }
            self.miner_stats[network_uid]['current_tier'] = self.scoring_system.get_miner_tier(network_uid)
            self.miner_stats[network_uid]['current_score'] = self.scoring_system.get_miner_composite_scores(network_uid)[day]