"""Byte-exact stdin/stdout adapters.

Text is decoded as UTF-8 with ``surrogateescape``: bytes that are not valid UTF-8
become lone surrogates on the way in and the very same bytes on the way out.
"""

ENCODING = "utf-8"
ERRORS = "surrogateescape"
