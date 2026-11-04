"""pipeline.common — services shared by neon_trim, object_tracking, and
future pipeline stages.

Promotion history: fsbrowse/config/media grew inside neon_trim (where they
are fully tested); the canonical implementations now live here and neon_trim
re-exports them, so existing imports and all existing tests keep working.

Modules:
    fsbrowse   PathAuthorizer, list_dir, RecentsStore (filesystem authz + browse)
    config     shared root/state configuration helpers
    media      video probing/decoding (PyAV) + ffmpeg helpers
    jobs       background job manager (threads, progress, cancel, shutdown)
    artifacts  generic manifest/loader helpers shared by artifact types
"""
