"""Command-line interface for MemoryGate API server."""

import argparse
import logging
import sys

import uvicorn

from memory_gate.api.app import create_app


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
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
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
    setup_logging(verbose=args.verbose)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting MemoryGate API on {args.host}:{args.port}")

    # Create app instance
    app = create_app()

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="debug" if args.verbose else "info",
    )


if __name__ == "__main__":
    sys.exit(main())
