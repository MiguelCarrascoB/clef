"""Sync/async Python client for the clef-flash server.

    from clef_client import ClefClient
    c = ClefClient()                       # CLEF_URL / CLEF_API_KEY env fallbacks
    r = c.classify("Checkout is down", ["billing", "technical", "account"])
    r.label, r.confidence, r.scores

Multi-label results: ``labels`` = every label >= threshold (best first), ``label`` = the top one (or None).
"""

from __future__ import annotations

from ._async import AsyncClassifier, AsyncClefClient
from ._common import DEFAULT_URL, ClefError, JobNotFinished, image_to_data_url, video_to_data_url
from ._sync import Classifier, ClefClient
from .types import Classification, Job, JobItem, JobList, ScoreResult

__all__ = [
    "AsyncClassifier",
    "AsyncClefClient",
    "DEFAULT_URL",
    "Classification",
    "Classifier",
    "ClefClient",
    "ClefError",
    "Job",
    "JobItem",
    "JobList",
    "JobNotFinished",
    "ScoreResult",
    "image_to_data_url",
    "video_to_data_url",
]
