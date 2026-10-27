"""Compatibility imports for legacy Streamlit code.

New SaaS code should import analytics.config_loader directly.
"""

from analytics.config_loader import load_client_config, validate_client_config

__all__ = ["load_client_config", "validate_client_config"]
