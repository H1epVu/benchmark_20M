"""
colab_drive_sync.py — Lưu checkpoint lên Google Drive trong quá trình training.

Cách dùng trong Colab cell:
    from colab_drive_sync import DriveSync
    sync = DriveSync('/content/drive/MyDrive/benchmark_checkpoints')
    sync.save()           # lưu thủ công
    sync.save(tag='best') # lưu với tag riêng
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path


class DriveSync:
    """Đồng bộ thư mục checkpoints & results lên Google Drive."""

    def __init__(
        self,
        drive_dir: str = '/content/drive/MyDrive/benchmark_checkpoints',
        benchmark_dir: str | None = None,
    ):
        self.drive_dir = Path(drive_dir)
        self.benchmark_dir = Path(benchmark_dir) if benchmark_dir else Path('/content/code/benchmark')
        self.drive_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.drive_dir / 'sync_log.json'
        self._log: list[dict] = self._load_log()

    # ------------------------------------------------------------------
    def save(self, tag: str = '') -> None:
        """Sao chép checkpoints + results từ benchmark lên Drive."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        label = f'{timestamp}_{tag}' if tag else timestamp

        copied: list[str] = []
        for folder in ('checkpoints', 'results'):
            src = self.benchmark_dir / folder
            if src.exists():
                dst = self.drive_dir / folder
                shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
                copied.append(folder)

        entry = {'time': label, 'copied': copied, 'tag': tag}
        self._log.append(entry)
        self._write_log()

        print(f'[DriveSync] Saved → {self.drive_dir}')
        print(f'  tag      : {tag or "(none)"}')
        print(f'  folders  : {", ".join(copied) if copied else "nothing found"}')

    def save_best(self, metric: float, minimize: bool = False) -> bool:
        """
        Lưu nếu metric tốt hơn lần trước.
        Trả về True nếu đã lưu, False nếu bỏ qua.
        """
        best = self._log[-1].get('best_metric') if self._log else None
        is_better = (
            best is None
            or (minimize and metric < best)
            or (not minimize and metric > best)
        )
        if is_better:
            self.save(tag='best')
            self._log[-1]['best_metric'] = metric
            self._write_log()
        return is_better

    # ------------------------------------------------------------------
    def restore(self, tag: str = '') -> None:
        """Khôi phục checkpoint từ Drive về benchmark_dir."""
        for folder in ('checkpoints', 'results'):
            src = self.drive_dir / folder
            if src.exists():
                dst = self.benchmark_dir / folder
                shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
                print(f'[DriveSync] Restored {folder} ← {src}')

    def status(self) -> None:
        """In lịch sử các lần lưu."""
        if not self._log:
            print('[DriveSync] Chưa có lần lưu nào.')
            return
        print(f'[DriveSync] {len(self._log)} lần lưu — thư mục: {self.drive_dir}')
        for e in self._log[-5:]:
            print(f"  {e['time']:22s}  tag={e.get('tag') or '-':12s}  {e['copied']}")

    # ------------------------------------------------------------------  (private)
    def _load_log(self) -> list[dict]:
        if self.log_path.exists():
            try:
                return json.loads(self.log_path.read_text())
            except Exception:
                return []
        return []

    def _write_log(self) -> None:
        self.log_path.write_text(json.dumps(self._log, indent=2))
