"""Constants shared by persisted ndscan schemas and submission arguments."""

#: Name of the ``artiq.language.HasEnvironment`` argument carrying scan parameters.
PARAMS_ARG_KEY = "ndscan_params"

#: Revision of the legacy experiment result schema.
SCHEMA_REVISION = 2

#: Dataset key used to identify an ndscan result tree and its schema revision.
SCHEMA_REVISION_KEY = "ndscan_schema_revision"

__all__ = ["PARAMS_ARG_KEY", "SCHEMA_REVISION", "SCHEMA_REVISION_KEY"]
