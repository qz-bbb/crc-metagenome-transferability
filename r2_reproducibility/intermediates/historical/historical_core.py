"""Bounded historical computations; source main routines are never invoked."""
import ast
import csv
import gzip
import hashlib
import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


def checkpoint(frame, index=False):
    """Reproduce a historical default to_csv/read_csv TSV stage boundary."""
    return pd.read_csv(StringIO(frame.to_csv(sep='\t', index=index)), sep='\t', index_col=0 if index else None)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fresh_output(output, inputs):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Output already exists; overwrite refused')
    for path in inputs:
        source = Path(path).resolve()
        parent = source if source.is_dir() else source.parent
        if output == parent or parent in output.parents or output in source.parents:
            raise ValueError('Output overlaps input root')
    output.mkdir(parents=True)
    return output


def read_matrix(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as handle:
        header = next(csv.reader(handle, delimiter='\t'))[1:]
    if len(header) != len(set(header)) or any(not name.strip() for name in header):
        raise ValueError('Duplicate or empty matrix columns')
    # Historical main() used the default C parser. Retain that parsed baseline.
    return pd.read_csv(path, sep='\t', index_col=0)


def transform(raw, metadata):
    required = {'sample_uid', 'study_name', 'study_condition'}
    if not required.issubset(metadata):
        raise ValueError('Missing metadata labels')
    if raw.index.has_duplicates or raw.columns.has_duplicates or metadata.sample_uid.duplicated().any():
        raise ValueError('Duplicate axes or sample labels')
    if raw.empty or metadata.empty or raw.index.isna().any() or raw.columns.isna().any():
        raise ValueError('Empty or missing axes')
    if metadata[list(required)].isna().any().any() or not metadata.study_condition.isin(['CRC', 'control']).all():
        raise ValueError('Invalid metadata labels')
    labels = list(raw.index) + list(raw.columns)
    labels += metadata.sample_uid.tolist() + metadata.study_name.tolist()
    if any(not isinstance(value, str) or not value.strip() for value in labels):
        raise ValueError('Blank or non-text axis/metadata labels')
    if list(raw.columns) != metadata.sample_uid.tolist():
        raise ValueError('Metadata/matrix sample order mismatch')
    values = raw.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Nonfinite or negative abundance')
    abundance = raw.T.copy()
    abundance.index.name = 'sample_uid'
    study = metadata.set_index('sample_uid').study_name
    presence = pd.DataFrame(index=abundance.columns)
    for cohort, ids in study.groupby(study).groups.items():
        presence[cohort] = (abundance.loc[ids] > 0).any(axis=0).astype(int)
    keep = ((abundance > 0).mean(axis=0) >= .05) & (presence.sum(axis=1) >= 3)
    abundance = abundance.loc[:, keep].copy()
    if abundance.empty:
        raise ValueError('No species retained')
    pseudocount = float(abundance.where(abundance > 0).min().min() / 2)
    logged = np.log(abundance + pseudocount)
    return abundance, logged.sub(logged.mean(axis=1), axis=0), pseudocount


def safe_functions(path, names, constants=()):
    """Load exact named functions and imports, omitting executable global setup."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            nodes.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets):
            ast.literal_eval(node.value)  # Reject executable constants.
            nodes.append(node)
    found = {node.name for node in nodes if isinstance(node, ast.FunctionDef)}
    if set(names) - found:
        raise ValueError(f'Missing source functions: {set(names) - found}')
    context = {'__name__': 'historical_algorithms'}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), context)
    return context


class NoFileWrites(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr in {'to_csv', 'write_text', 'write_bytes', 'mkdir'}:
            return None
        return self.generic_visit(node)


def fragment(path, start, end, context):
    """Execute an explicit main() statement interval, without historical writes."""
    text = Path(path).read_text(encoding='utf-8-sig')
    lines = text.splitlines()
    first = [i + 1 for i, line in enumerate(lines) if line.lstrip().startswith(start)]
    last = [i + 1 for i, line in enumerate(lines) if line.lstrip().startswith(end)]
    if len(first) != 1 or len(last) != 1 or first[0] > last[0]:
        raise ValueError(f'Ambiguous historical interval: {start!r} .. {end!r}')
    main = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    chosen = [node for node in main.body if first[0] <= node.lineno <= last[0]]
    module = NoFileWrites().visit(ast.Module(body=chosen, type_ignores=[]))
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), 'exec'), context)


def compare_frames(generated, target, tolerance):
    result = {'pass': True, 'shape_generated': list(generated.shape), 'shape_target': list(target.shape),
              'columns_exact': generated.columns.tolist() == target.columns.tolist(),
              'index_exact': generated.index.tolist() == target.index.tolist(), 'columns': {},
              'atol': tolerance, 'rtol': tolerance}
    if generated.shape != target.shape or not result['columns_exact'] or not result['index_exact']:
        result['pass'] = False
        return result
    for column in generated:
        left, right = generated[column], target[column]
        missing = left.isna().equals(right.isna())
        valid = ~(left.isna() | right.isna())
        numeric = (pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right)
                   and not pd.api.types.is_bool_dtype(left) and not pd.api.types.is_bool_dtype(right))
        if numeric:
            a, b = left[valid].to_numpy(float), right[valid].to_numpy(float)
            exact = bool(np.array_equal(a, b))
            close = bool(np.allclose(a, b, atol=tolerance, rtol=tolerance))
            maximum = float(np.max(np.abs(a-b))) if len(a) else 0.
        else:
            exact = left[valid].tolist() == right[valid].tolist()
            close, maximum = exact, None
        result['columns'][str(column)] = {'numeric': numeric, 'exact': exact, 'missingness_exact': missing,
                                         'within_tolerance': close, 'max_abs_difference': maximum}
        result['pass'] &= missing and close
    result['pass'] = bool(result['pass'])
    return result


def snapshot_source(path, names, intervals=(), constants=()):
    """Mechanically export selected AST nodes, with no private global setup."""
    text = Path(path).read_text(encoding='utf-8-sig')
    tree = ast.parse(text)
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets):
            ast.literal_eval(node.value)
            nodes.append(node)
    if intervals:
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        keep = []
        lines = text.splitlines()
        for start, end in intervals:
            first = [i+1 for i,line in enumerate(lines) if line.lstrip().startswith(start)]
            last = [i+1 for i,line in enumerate(lines) if line.lstrip().startswith(end)]
            if len(first) != 1 or len(last) != 1:
                raise ValueError('Ambiguous snapshot interval')
            keep.extend(node for node in main.body if first[0] <= node.lineno <= last[0])
        main.body = sorted({node.lineno: node for node in keep}.values(), key=lambda node: node.lineno)
        nodes.append(NoFileWrites().visit(main))
    output = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(output)
    return '# Mechanically selected historical AST; main is not an entrypoint.\n' + ast.unparse(output) + '\n'
