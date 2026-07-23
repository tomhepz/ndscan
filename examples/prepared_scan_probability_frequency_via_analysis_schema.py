"""Schema-compiled entrypoint for the probability/frequency via-analysis example.

This reuses the same fragment hierarchy as
``prepared_scan_probability_frequency_via_analysis.py`` but exercises the new
dict-based scan submission schema compilation path instead of the code-first
``lambda fragment: ScanRequest...`` form.
"""

from __future__ import annotations

from prepared_scan_probability_frequency_via_analysis import (
    FrequencyFromProbabilityViaAnalysisFragment,
)

from ndscan.runtime.api import make_fragment_prepared_scan_exp
from ndscan.submission.scan_submission_schema import compile_scan_submission_schema

PreparedScanProbabilityFrequencyViaAnalysisSchema = make_fragment_prepared_scan_exp(
    FrequencyFromProbabilityViaAnalysisFragment,
    lambda fragment: compile_scan_submission_schema(
        fragment,
        {
            "version": 1,
            "mode": {"type": "grid"},
            "entries": [],
            "execution": {},
            "metadata": {
                "demo_name": "prepared_scan_probability_frequency_via_analysis_schema"
            },
        },
    ),
)
