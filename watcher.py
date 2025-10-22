from pathlib import Path
from threading import Thread

import time

def watch(folder="runs", pattern="*.csv", interval=100e-3):
    folder = Path(folder)
    seen = {p.resolve() for p in folder.glob(pattern)}
    def _run():
        nonlocal seen
        while True:
            current_files = {p.resolve() for p in folder.glob(pattern)}
            new_files = current_files - seen

            for new_file in new_files:
                print(f"[NEW FILE]: {new_file.name}")

            seen = current_files.copy()
            time.sleep(interval)
    Thread(target=_run, daemon=True).start()