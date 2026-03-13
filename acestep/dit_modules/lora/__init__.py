"""Handler-bound LoRA management methods (dit_modules layer).

These modules provide ``self``-bound methods that are assigned onto
``AceStepDiTWrapper`` at class definition time. They delegate to the
pure/stateless ``acestep.core.lora.LoraService`` via ``self._lora_service``.
"""
