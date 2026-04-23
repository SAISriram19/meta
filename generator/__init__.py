"""Scenario generator — self-improving curriculum.

CoEvolve-inspired: extracts failure signals from agent rollouts and synthesizes
new scenarios that target those weak spots. GenEnv-inspired feasibility gate
keeps synthesized scenarios in the "zone of proximal development" (solvable ≥ ε
of the time by the current policy).
"""
