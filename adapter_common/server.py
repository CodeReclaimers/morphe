"""
Common server utilities for sketch adapters.

This module provides shared functionality used across all CAD adapter servers.
"""

from xmlrpc.server import SimpleXMLRPCRequestHandler

DEFAULT_HOST = "localhost"
SERVER_VERSION = "1.0.0"


class QuietRequestHandler(SimpleXMLRPCRequestHandler):
    """Request handler that suppresses logging."""

    def log_message(self, format: str, *args: object) -> None:
        pass  # Suppress default logging
