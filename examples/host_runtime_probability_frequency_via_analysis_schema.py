"""Dict-schema entrypoint for the probability/frequency via-analysis example.

This reuses the same fragment hierarchy as
``host_runtime_probability_frequency_via_analysis.py`` but exercises the new
dict-based host scan schema compilation path instead of the code-first
``lambda fragment: ScanRequest...`` form.
"""

from __future__ import annotations

from ndscan.runtime.api import make_fragment_prepared_scan_exp

from host_runtime_probability_frequency_via_analysis import (
    FrequencyFromProbabilityViaAnalysisFragment,
)


HostRuntimeProbabilityFrequencyViaAnalysisSchema = make_fragment_prepared_scan_exp(
    FrequencyFromProbabilityViaAnalysisFragment,
    {
        "version": 1,
        "mode": {"type": "grid"},
        "entries": [],
        "execution": {},
        "metadata": {
            "demo_name": "host_runtime_probability_frequency_via_analysis_schema"
        },
    },
)
