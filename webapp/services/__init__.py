"""Service layer: the only place the web app touches ``pickleball_phase2``.

Each service is a small, testable unit with no Flask dependency, so routes stay
thin and the algorithm wiring is easy to swap (mock -> trained model).
"""
