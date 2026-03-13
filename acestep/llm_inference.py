"""Backward-compatibility shim — imports LLMHandler from lm_wrapper."""

from acestep.lm_wrapper import AceStepLMWrapper, LLMHandler  # noqa: F401

__all__ = ["AceStepLMWrapper", "LLMHandler"]
