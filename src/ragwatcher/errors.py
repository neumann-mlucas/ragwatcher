"""Custom exception hierarchy. Mapped to CLI exit codes in cli.py."""


class RagwatcherError(Exception):
    """Base."""


class LockHeld(RagwatcherError):
    """Another process holds the persist-dir lock."""


class SchemaMismatch(RagwatcherError):
    """Manifest fingerprint disagrees with current config."""


class ReadError(RagwatcherError):
    """A reader failed on a specific file. Non-fatal."""


class EmbedError(RagwatcherError):
    """Embedding backend failed."""


class StoreError(RagwatcherError):
    """Vector store backend failed."""
