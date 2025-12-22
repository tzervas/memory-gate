"""Command-line interface for MemoryGate API server."""

import argparse
import logging
from pathlib import Path
import sys

import uvicorn

from memory_gate.api.app import create_app
from memory_gate.config import MemoryGateConfig


def setup_logging(verbose: bool = False) -> None:
    """Configure logging.

    Args:
        verbose: Enable verbose logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    """Main entry point for the API server CLI."""
    parser = argparse.ArgumentParser(
        description="MemoryGate API Server",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to configuration file (YAML or TOML)",
    )
    parser.add_argument(
        "--host",
        help="Host to bind to (overrides config file)",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to bind to (overrides config file)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development only)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        if args.config:
            config = MemoryGateConfig.load(args.config)
        else:
            config = MemoryGateConfig.load()
    except FileNotFoundError as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # Override config with CLI arguments
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    if args.reload:
        config.server.reload = args.reload
    if args.verbose:
        config.server.log_level = "debug"

    setup_logging(verbose=args.verbose or config.server.log_level == "debug")

    logger = logging.getLogger(__name__)
    logger.info(f"Starting MemoryGate API on {config.server.host}:{config.server.port}")
    if args.config:
        logger.info(f"Loaded configuration from: {args.config}")

    # Create app instance
    app = create_app()

    # Run server
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        log_level=config.server.log_level,
    )


if __name__ == "__main__":
    sys.exit(main())
