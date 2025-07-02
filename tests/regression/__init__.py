"""Regression testing framework for memory-gate.

This module provides a comprehensive regression testing framework that ensures
core functionality remains stable across releases.

Key Principles:
- Idempotent tests that can run multiple times with same results
- No for loops in test logic (use parametrize instead)
- Simple, focused test scenarios
- Comprehensive coverage of critical paths
"""
