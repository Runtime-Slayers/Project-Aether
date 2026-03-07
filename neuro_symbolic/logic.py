from typing import List

class DoctrineRules:
    """Simple neuro-symbolic rule engine for action filtering and explainability.

    For demo: rules are modulation-level constraints. Expand to a full symbolic
    reasoner or logic programming layer as needed for publications.
    """
    def __init__(self, forbidden_modulations: List[str] = None):
        if forbidden_modulations is None:
            forbidden_modulations = []
        self.forbidden = set(forbidden_modulations)

    def allowed_mask(self, class_names: List[str]):
        """Return a boolean mask for allowed classes given `class_names`."""
        return [name not in self.forbidden for name in class_names]

    def explain_filter(self, class_name: str):
        if class_name in self.forbidden:
            return f"{class_name} filtered by doctrine (forbidden)."
        return f"{class_name} allowed."
