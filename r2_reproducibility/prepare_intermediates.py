"""Generate one explicitly selected missing intermediate family at a time.

No data download, model-wide refit, or public release is performed implicitly.
Each stage has its own documented JSON configuration and new output directory.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
STAGES={
    'historical-matrices':('historical/historical_intermediates.py','matrices'),
    'historical-panels':('historical/historical_intermediates.py','panels'),
    'historical-full':('historical/historical_intermediates.py','full'),
    'identity-enrich':('identity/identity_sources.py','enrich'),
    'identity-audit':('identity/identity_sources.py','audit'),
    'sparsity':('sparsity/sparsity_generator.py',None),
}

def command(stage,config):
    if stage not in STAGES:raise ValueError('Select one explicit stage; no implicit all-stage run')
    script,substage=STAGES[stage]
    args=[sys.executable,'-B',str(ROOT/'intermediates'/script),'--config',str(Path(config).resolve())]
    if substage:args.append(substage)
    return args

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=STAGES);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();args=command(a.stage,a.config)
    if not a.config.is_file():p.error('Missing explicit configuration file')
    if not Path(args[2]).is_file():p.error('Incomplete generator package')
    raise SystemExit(subprocess.run(args,check=False).returncode)

if __name__=='__main__':main()
