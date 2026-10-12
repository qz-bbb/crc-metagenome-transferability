"""Protect source and repository directories before report generation."""
from pathlib import Path

def output_directory(target, sources):
    target=Path(target).resolve()
    repository=Path(__file__).resolve().parents[1]
    for source in [repository,*map(lambda p:Path(p).resolve(),sources)]:
        if target==source or target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError('Output must be disjoint from source and repository directories')
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError('Output directory must be new or empty')
    target.mkdir(parents=True,exist_ok=True)
    return target
