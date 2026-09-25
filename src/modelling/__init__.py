"""Feature-engineering ladder: shared logic for notebooks/01, 02 and 03.

This package is additive (v3 of the build). It does not import from, and is
never imported by, src/extract, src/transform, src/model or src/quality --
the ladder reads the already-built, already-verified gold layer read-only.
"""
