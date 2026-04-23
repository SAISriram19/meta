"""Stakeholder Management Gym — env package."""

# Auto-load .env (no-op if file absent). Safe because it only SETS keys that
# aren't already in the environment — external shell exports always win.
from env._dotenv import load_dotenv  # noqa: F401,E402

