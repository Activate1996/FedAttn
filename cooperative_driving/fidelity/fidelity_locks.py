"""Process locks for writer isolation, crash recovery and sequential model loading."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import socket
import time

PROTOCOL='guard-flock-v1'

@contextmanager
def result_lock(root, reader=False):
    root=Path(root)
    if reader and not root.is_dir():
        raise FileNotFoundError(root)
    root.mkdir(parents=True,exist_ok=True)
    marker=root/'v2v_fidelity_running.lock'
    with (root/'.v2v_fidelity_output.guard').open('a+') as guard:
        try:
            fcntl.flock(guard.fileno(),(fcntl.LOCK_SH if reader else fcntl.LOCK_EX)|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'{root}: a writer or conflicting export is active; wait for it to stop') from exc
        try:
            if marker.exists():
                try: old=json.loads(marker.read_text())
                except (OSError,ValueError): old={}
                if old.get('lock_protocol')!=PROTOCOL:
                    raise ValueError(f'{marker}: legacy/unknown lock; confirm its process has stopped before removing this marker')
                # A held writer would prevent acquiring guard above. Therefore a
                # marker with this protocol is stale after a killed process.
            if not reader:
                temp=marker.with_name(marker.name+'.tmp')
                temp.write_text(json.dumps({'lock_protocol':PROTOCOL,'pid':os.getpid(),'host':socket.gethostname()}))
                temp.replace(marker)
            try:
                yield
            finally:
                if not reader: marker.unlink(missing_ok=True)
        finally:
            fcntl.flock(guard.fileno(),fcntl.LOCK_UN)

@contextmanager
def model_load_lock(path, check_cancel=lambda:None):
    if path is None:
        check_cancel();yield;return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as guard:
        print(f'Waiting for model loading slot: {path}',flush=True)
        while True:
            check_cancel()
            try:
                fcntl.flock(guard.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(.2)
        try:
            check_cancel()
            guard.seek(0);guard.truncate();guard.write(str(os.getpid()));guard.flush()
            print('Model loading slot acquired.',flush=True)
            yield
        finally:
            fcntl.flock(guard.fileno(),fcntl.LOCK_UN)
