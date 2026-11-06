"""One physical namespace for an owned run under a mounted Volume alias."""
from pathlib import Path

from .hn2_safe_runtime import owned_prefixes


def checked_owned_paths(data_root, run_id, asset_root):
    assets, output = owned_prefixes(data_root, run_id)
    supplied = Path(asset_root)
    if not supplied.is_absolute() or supplied.resolve() != assets:
        raise ValueError('Private asset prefix differs from owned run')
    # Only the mount root was resolved in owned_prefixes. A symlink in the
    # run/parent that redirects outside this exact owned directory is rejected.
    return assets, output
