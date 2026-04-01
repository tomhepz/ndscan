# ndscan Agent Notes

## Python Environment

- Do **not** use `uv run --active ...` in this repository.
- It can rebuild or swap environments in a way that breaks the ARTIQ setup used by the
  dashboard, worker examine path, and local development flow.
- Use the project venv directly instead:
  - Python: `/home/lab/artiq-files/install/ndscan/.venv/bin/python`
  - Typical unittest pattern:
    - `/home/lab/artiq-files/install/ndscan/.venv/bin/python -m unittest ...`

## ARTIQ / Dashboard

- If a file suddenly fails to open in the dashboard with an ARTIQ worker/examine error
  after environment changes, first suspect the Python environment.
- Keep the runtime used for local checks aligned with the runtime used by ARTIQ
  examine/worker processes.

## Runtime / Plotting Context

- The prepared runtime is the active path.
- Legacy code still exists, but new work should generally target:
  - `ndscan.define`
  - `ndscan.scan`
  - `ndscan.submission`
  - `ndscan.runtime`
  - `ndscan.plots.runtime`

## Fitting

- `third_party/sensible-fitting` is vendored and wrapped through:
  - `ndscan.fits.sensible`
- Prefer using the ndscan wrapper, not the vendored package directly from multiple
  places.

