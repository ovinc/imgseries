"""Misc. tools for compatibility with older code versions"""


from pathlib import Path

from .fileio import FileIO


def update_analysis_metadata(old_file, new_file=None):
    """Update json of analysis metadata with formatting compatible with v>=0.10

    The main difference is that transform metadata is now stored with the
    key "transforms" in analysis metadata, contrary to being at the root
    of the json/dict as before.

    Parameters
    ----------
    old_file : {str, pathlib.Path}
        path to the old file containing the metadata

    new_file : {str, pathlib.Path}, optional
        if not supplied, use same name but with '_new' added in the name
    """
    TRANSFORMS = (   # transforms used by default prior to v0.10
        'grayscale',
        'rotation',
        'crop',
        'filter',
        'subtraction',
        'threshold',
    )
    file = Path(old_file)
    data = FileIO.from_json(file)

    data['transforms'] = {}
    for transform in TRANSFORMS:
        data['transforms'][transform] = data.pop(transform)
        # This is to put back time info and code versions at the end of the file
        data['time (utc)'] = data.pop('time (utc)')
        data['code version'] = data.pop('code version')

    if new_file is None:
        new_file = file.with_name(f"{file.stem}_new{file.suffix}")

    FileIO.to_json(data=data, filepath=new_file)
