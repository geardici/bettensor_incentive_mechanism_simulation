import logging

def setup_logger():
    """
    Sets up and returns a centralized logger for the entire application.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger('simulation_logger')
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler which logs even debug messages
    fh = logging.FileHandler('simulation.log', mode='w')
    fh.setLevel(logging.DEBUG)
    
    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False
    
    return logger

# Create a single logger instance to be used throughout the application
logger = setup_logger()