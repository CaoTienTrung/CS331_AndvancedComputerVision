import yaml
import argparse

def load_config(config_path: str) -> dict:
    """Load a YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_parser() -> argparse.ArgumentParser:
    """Create an argument parser for command-line options."""
    parser = argparse.ArgumentParser(description="Dataset Utilities")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    return parser.parse_args()