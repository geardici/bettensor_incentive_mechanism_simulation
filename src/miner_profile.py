import torch as t

class MinerProfile:
    def __init__(self, num_miners: int, max_days: int, profile_type: str = None, base_price: float = None, 
                 risk_tolerance: float = None, fee_sensitivity: float = None, skill_factor: float = None):
        self.num_miners = num_miners
        self.max_days = max_days
        self.profile_type = profile_type
        self.base_price = base_price
        self.risk_tolerance = risk_tolerance
        self.fee_sensitivity = fee_sensitivity
        self.skill_factor = skill_factor
        self.learning_rate = 0.01  # Default value, can be adjusted
        self.num_predictions = t.zeros(num_miners, max_days, dtype=t.int64)
        self.weights = t.zeros(num_miners,max_days, dtype=t.float32)  # Add this line to store weights
        self.performance_history = t.tensor([])  # Initialize performance_history

    @classmethod
    def get_random_profile(cls, num_miners: int, max_days: int):
        profiles = [
            cls(num_miners, max_days, 'conservative', 1.02, 0.3, 0.7, 0.7),
            cls(num_miners, max_days, 'balanced', 1.05, 0.5, 0.5, 0.8),
            cls(num_miners, max_days, 'aggressive', 1.1, 0.7, 0.3, 1.0)
        ]
        weights = t.tensor([0.3, 0.5, 0.2])
        return profiles[t.multinomial(weights, 1).item()]

    def calculate_registration_probability(self, current_tau, current_fee, day):
        earning_potential = self.base_price * current_tau * (self.skill_factor ** 2)  # Make skill factor more impactful
        relative_expense = current_fee / earning_potential
        base_prob = max(0, 1 - relative_expense) * self.risk_tolerance
        day_factor = max(1, 30 / (day + 1))  # This will decrease over time
        scaling_factor = 0.05  # Increased from 0.01 to 0.05
        prob_register = base_prob * day_factor * scaling_factor
        return max(min(prob_register, 1), 0.01)  # Ensure a minimum 1% chance of registration

    def decide_prediction_volume(self, current_tau):
        base_volume = int(10 * self.risk_tolerance)
        adjusted_volume = int(base_volume * (1 + (current_tau - 1) * 0.2))
        return max(min(adjusted_volume, 20), 1)

    def update_strategy(self, performance):
        self.performance_history = t.cat([self.performance_history, t.tensor([performance])])
        if len(self.performance_history) > 10:
            recent_performance = t.mean(self.performance_history[-10:])
            if recent_performance > 0:
                self.risk_tolerance = min(1.0, self.risk_tolerance + self.learning_rate * self.skill_factor)
            else:
                self.risk_tolerance = max(0.1, self.risk_tolerance - self.learning_rate * (1 - self.skill_factor))