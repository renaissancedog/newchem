import os
import logging

# Get a logger for this module
logger = logging.getLogger(__name__)


def get_api_key(env_variable_name: str = "MP_API_KEY") -> str | None:
    """
    Retrieves an API key from an environment variable.

    Args:
        env_variable_name (str): The name of the environment variable
                                 holding the API key. Defaults to "MP_API_KEY".

    Returns:
        str | None: The API key if found, otherwise None.
    """
    api_key = os.getenv(env_variable_name)
    if not api_key:
        logger.warning(
            f"Environment variable '{env_variable_name}' not found. "
            "This is required for accessing data from Materials Project."
        )
    return api_key


def setup_logging(level=logging.INFO, log_file: str | None = None) -> None:
    """
    Configures basic logging for the application.
    Logs to console by default. Can also log to a file.

    Args:
        level: The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file (str | None): Optional path to a file where logs should be written.
    """
    handlers = [logging.StreamHandler()]  # Log to console
    if log_file:
        # Ensure directory for log file exists if specified with a path
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='a'))  # Append to log file

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True
    )
    logger.info(f"Logging configured. Level: {logging.getLevelName(level)}. Output to console" +
                (f" and file: {log_file}" if log_file else "."))


# Define constants that might be used across modules, e.g., HDF5 keys
HDF5_KEY_ELECTRODE_PAIRS = "data"  # Key for initial pairs and structures
HDF5_KEY_WITH_ENERGIES = "data_with_energies"  # Key for data after energy calculations
HDF5_KEY_INTERCALATION = "intercalation_data"  # Key for the interstitial intercalation pipeline
