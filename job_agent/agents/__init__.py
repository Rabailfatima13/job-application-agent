"""Supervisor + workers.

Responsibilities are split exactly as the proposal specifies (S8), and the
split is the point: the supervisor coordinates but does not do the workers'
jobs, and no worker reaches outside its own.

  supervisor.py  Week 6 - parses inputs, runs the graph, assembles the result,
                 writes the tracker record. Owns routing, not reasoning.
  research.py    Week 6 - web-searches the company, returns a sourced brief.
  scoring.py     done - requirements vs. CV evidence -> FitReport.
  writing.py     Week 7 - re-emphasised CV + cover letter, grounded only.

All three workers inherit BaseAgent for their traced model/tool calls.
"""

from .base import BaseAgent
from .scoring import ScoringAgent, compute_fit_score

__all__ = ["BaseAgent", "ScoringAgent", "compute_fit_score"]
