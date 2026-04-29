"""
Content-addressed caches for Hailo-derived vision features.

Public surface:
    from openclaw_shared.cache.vision_cache import (
        VisionCache,
        sha256_file,
        fingerprint_hef_dir,
        fingerprint_pipeline,
    )
"""
from openclaw_shared.cache.vision_cache import (
    VisionCache,
    sha256_file,
    fingerprint_hef_dir,
    fingerprint_pipeline,
)

__all__ = [
    "VisionCache",
    "sha256_file",
    "fingerprint_hef_dir",
    "fingerprint_pipeline",
]
