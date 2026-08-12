"""Supervisor + workers.

Responsibilities are split exactly as the proposal specifies (S8), and the
split is the point: the supervisor coordinates but does not do the workers'
jobs, and no worker reaches outside its own.

  supervisor.py  done - runs the graph, assembles the result, writes the
                 tracker record. Owns orchestration, not reasoning.
  research.py    done - web-searches the company, returns a sourced brief.
  scoring.py     done - requirements vs. CV evidence -> FitReport.
  writing.py     Week 7 - re-emphasised CV + cover letter, grounded only.

All three workers inherit BaseAgent for their traced model/tool calls.
"""

from .base import BaseAgent
from .research import ResearchAgent
from .scoring import ScoringAgent, compute_fit_score
from .supervisor import Supervisor, build_supervisor

__all__ = [
    "BaseAgent",
    "ResearchAgent",
    "ScoringAgent",
    "Supervisor",
    "build_supervisor",
    "compute_fit_score",
]
