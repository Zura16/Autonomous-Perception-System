"""APS -- monocular perception and forward-collision-warning stack.

Scope, stated identically everywhere: a single forward camera. No LiDAR in the
estimation path, no radar, no fusion, no vehicle actuation. LiDAR appears only
as ground truth for evaluation. The system emits warnings and an AEB *request*
in an offline, open-loop evaluation. It does not brake anything.
"""

__version__ = "0.1.0"
