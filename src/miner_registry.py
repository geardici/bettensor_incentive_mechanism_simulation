class MinerRegistry:
    def __init__(self, max_capacity):
        self.max_capacity = max_capacity
        self.registered_miners = {}  # network_uid -> unique_id
        self.next_unique_id = 0
        self.available_network_uids = set(range(max_capacity))
        self.registration_days = {}  # network_uid -> registration day
        self.current_day = 0  # Add this to keep track of the current day

    def register_miner(self):
        if self.available_network_uids:
            network_uid = min(self.available_network_uids)
            unique_id = self.next_unique_id
            self.registered_miners[network_uid] = unique_id
            self.registration_days[network_uid] = self.current_day  # Record registration day
            self.available_network_uids.remove(network_uid)
            self.next_unique_id += 1
            return network_uid, unique_id
        return None, None

    def deregister_miner(self, network_uid):
        if network_uid in self.registered_miners:
            del self.registered_miners[network_uid]
            del self.registration_days[network_uid]  # Remove registration day
            self.available_network_uids.add(network_uid)
            return True
        return False

    def is_miner_registered(self, network_uid):
        return network_uid in self.registered_miners

    def get_active_miners(self):
        return list(self.registered_miners.keys())

    def get_total_registered_miners(self):
        return len(self.registered_miners)

    def get_unique_id(self, network_uid):
        return self.registered_miners.get(network_uid)

    def get_registration_day(self, network_uid):
        return self.registration_days.get(network_uid, 0)

    def update_current_day(self, day):
        self.current_day = day