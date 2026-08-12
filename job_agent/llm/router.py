"""Which model runs which step.

The routing rule is the proposal's cost strategy in one place: mechanical steps
(parsing, summarising) go to the cheap LIGHT tier, judgement steps (scoring,
writing) go to the HEAVY tier. Agents ask for a tier, never for a vendor.

Clients are built lazily and cached, so a run that never touches the heavy tier
never needs an Anthropic key - which keeps light-only tests and demos runnable.
"""

from ..config import HEAVY, LIGHT, ModelConfig, Settings
from .base import ModelClient
from .providers import build_client

# Step name -> tier. Extend as steps are added; unknown steps default to HEAVY
# so a new step is never silently run on the weaker model.
STEP_TIERS: dict[str, str] = {
    "parse_cv": LIGHT,
    "parse_jd": LIGHT,
    "summarise": LIGHT,
    "score_fit": HEAVY,
    "tailor_cv": HEAVY,
    "draft_cover_letter": HEAVY,
}


class ModelRouter:
    def __init__(self, settings: Settings, client_factory=build_client) -> None:
        self._configs: dict[str, ModelConfig] = {
            HEAVY: settings.heavy,
            LIGHT: settings.light,
        }
        self._client_factory = client_factory
        self._cache: dict[str, ModelClient] = {}

    def tier_for(self, step: str) -> str:
        return STEP_TIERS.get(step, HEAVY)

    def client(self, tier: str) -> ModelClient:
        if tier not in self._configs:
            raise ValueError(f"Unknown model tier '{tier}' (expected 'heavy'/'light').")
        if tier not in self._cache:
            self._cache[tier] = self._client_factory(self._configs[tier])
        return self._cache[tier]

    def for_step(self, step: str) -> ModelClient:
        """The client that should run `step` - the call agents actually make."""
        return self.client(self.tier_for(step))
