class RegistrationFeeManager:
    def __init__(self, min_burn=0.05, max_burn=5, initial_fee=0.1, target_registrations=24, alpha=0.1):
        self.min_burn = min_burn
        self.max_burn = max_burn
        self.current_fee = initial_fee
        self.target_registrations = target_registrations
        self.alpha = alpha

    def update_fee(self, tau, daily_registrations, active_miners, _):
        registration_ratio = daily_registrations / self.target_registrations

        if registration_ratio > 1:
            self.current_fee *= (1 + self.alpha * (registration_ratio - 1))
        else:
            self.current_fee *= (1 - self.alpha * (1 - registration_ratio))

        self.current_fee = max(min(self.current_fee, self.max_burn, tau / 2), self.min_burn)

    def get_current_fee(self):
        return self.current_fee