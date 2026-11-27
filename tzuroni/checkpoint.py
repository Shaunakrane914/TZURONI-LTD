import json
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import pandas as pd


class LiveCheckpoint:
    def __init__(self, path, end=None):
        self.path = path
        if end is None:
            self.data = json.loads(path.read_text(encoding='utf-8'))
            if self.data.get('version') != 1:
                raise ValueError('This checkpoint version is unsupported. Use a new output directory.')
        else:
            self.data = {'version': 1, 'end': end.isoformat(), 'stages': {}}
            self.save()
        self.end = pd.Timestamp(self.data['end'])
        self.start = self.end - timedelta(days=10)

    def get(self, stage):
        return self.data['stages'].get(stage)

    def put(self, stage, value):
        self.data['stages'][stage] = value
        self.save()

    def save(self):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.data, ensure_ascii=False), encoding='utf-8')
        temporary.replace(self.path)


@contextmanager
def open_live_checkpoint(output: Path, resume: bool, end: pd.Timestamp):
    output.mkdir(parents=True, exist_ok=True)
    path = output / '.live-checkpoint.json'
    lock = output / '.live.lock'
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        raise ValueError(f'A live run already holds {lock}.') from None
    try:
        if resume and not path.exists():
            raise ValueError(f'No live checkpoint in {output}.')
        if not resume and (path.exists() or (output / 'snapshot').exists()):
            raise ValueError(f'{output} already holds a run. Use --resume or choose a new --out directory.')
        yield LiveCheckpoint(path, None if resume else end)
    finally:
        lock.unlink(missing_ok=True)
