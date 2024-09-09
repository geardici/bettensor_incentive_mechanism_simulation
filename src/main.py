import logging
import traceback
from simulation import Simulation
from return_generator import ReturnGenerator
from utils import setup_logger

# Define constants
MAX_MINERS = 230
NUM_DAYS = 365

# Create a single logger instance to be used throughout the file
logger = setup_logger()

if __name__ == "__main__":
    try:
        print("Starting simulation...")
        logger.info("Starting simulation...")
        return_generator = ReturnGenerator()
        simulation = Simulation(max_capacity=MAX_MINERS, num_days=NUM_DAYS, return_generator=return_generator,
                                min_burn=0.05, max_burn=5, initial_fee=0.1)
        simulation.run_simulation()
        print("Simulation and analytics completed.")
        logger.info("Simulation and analytics completed.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())