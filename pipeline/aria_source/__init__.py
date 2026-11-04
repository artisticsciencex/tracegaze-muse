"""aria_source — source adapters for Project Aria public datasets.

Adds Aria as a SECOND source system beside NEON. Aria recordings are not
disguised as NEON recordings: they get their own artifact types
(`prepared_aria_recording`, `trimmed_aria_recording`) whose manifests
state their own timestamp domain (Aria DEVICE_TIME) and provenance. Both
sources converge on one logical recording interface consumed downstream:

    scene video · one authoritative timestamp per scene frame ·
    timestamped 2D gaze in scene-video pixels · source/provenance metadata

Native Project Aria code (VRS, calibration, gaze reprojection) NEVER runs
in a long-lived server process. It lives exclusively in `helper_cli`,
executed as an isolated subprocess by the dedicated Aria interpreter, so
a VRS/native crash cannot take a GUI server down.
"""

__version__ = "0.2.0"

PREPARED_ARIA_TYPE = "prepared_aria_recording"
TRIMMED_ARIA_TYPE = "trimmed_aria_recording"
ARIA_PREP_MANIFEST = "prep_manifest.json"
ARIA_TRIM_MANIFEST = "trim_manifest.json"
SOURCE_SYSTEM = "project_aria"
