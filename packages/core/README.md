# @tern/core

> Shared types, errors, and utilities used across `@tern` packages. Not a standalone library — install `@tern/semantic` (or other downstream packages) instead.

## What's here

- **Type definitions** common to all `@tern` packages (`Embedding`, `SimilarityResult`, etc.)
- **Error classes** for consistent error handling across packages
- **Internal utilities** that don't belong to any single package

This package exists to avoid duplicating type definitions and helper code as the `@tern` package family grows. End users rarely depend on it directly.

## Status

v0.1, pre-alpha. Bootstrap minimal — adds members as packages need them.