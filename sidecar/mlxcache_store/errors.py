"""Protocol error taxonomy (connector protocol §3)."""


class CorruptCache(Exception):  # noqa: N818 — protocol §3 taxonomy
    """A cache entry is deterministically invalid: digest | version | header.

    The caller falls back to scratch; the entry must be treated as gone.
    """

    def __init__(self, kind: str, detail: str = ""):
        self.kind = kind
        super().__init__(f"corrupt cache entry ({kind}): {detail}")


class Unavailable(Exception):  # noqa: N818 — protocol §3 taxonomy
    """Transient/IO failure — the store cannot service the operation now.

    Not proof of corruption; retrying later is legitimate.
    """


class Refused(Exception):  # noqa: N818 — protocol §3 taxonomy
    """The store declined the operation (budget/recovery policy)."""
