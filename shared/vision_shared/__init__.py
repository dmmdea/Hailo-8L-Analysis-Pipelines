"""
vision_shared — shared primitives across Python MCP servers.

Lightweight exports only. Subpackages with heavier dependencies are imported explicitly:

  # quota (always lightweight):
  from vision_shared import QuotaBudget, QuotaExceeded, KNOWN_SERVICES

  # metrics (needs pandas):
  from vision_shared.metrics.outlier import channel_outlier_scores

  # oauth (needs google-auth / google-auth-oauthlib):
  from vision_shared.oauth_manager import load_credentials, run_consent_flow
"""
from vision_shared.quota_budget import (
    QuotaBudget,
    QuotaExceeded,
    ServiceConfig,
    KNOWN_SERVICES,
    state_dir,
)

__all__ = [
    "QuotaBudget",
    "QuotaExceeded",
    "ServiceConfig",
    "KNOWN_SERVICES",
    "state_dir",
]
