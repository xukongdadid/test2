import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable


@dataclass
class LogConfig:
    save_dof: bool = False
    save_hydro: bool = False
    save_wind_aero: bool = False
    save_moor: bool = False


class RunDataLogger:
    def __init__(self, base_dir: str, config: LogConfig, timestamp: datetime | None = None):
        self.config = config
        self.timestamp = timestamp or datetime.now()
        self.run_stamp = self.timestamp.strftime("%Y%m%d%H%M")
        self.run_dir = os.path.join(base_dir, "data", f"Run_{self.run_stamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self._files: Dict[str, object] = {}
        self._writers: Dict[str, csv.DictWriter] = {}

    def _open_writer(self, key: str, filename: str, fieldnames: Iterable[str]) -> csv.DictWriter:
        if key in self._writers:
            return self._writers[key]
        path = os.path.join(self.run_dir, filename)
        f = open(path, "w", newline="")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        self._files[key] = f
        self._writers[key] = writer
        return writer

    def log(self, data: dict, logs: dict) -> None:
        stamp = self.run_stamp
        if self.config.save_dof:
            dof_data = logs.get("dof", {})
            writer = self._open_writer("dof", f"dof{stamp}.csv", dof_data.keys())
            writer.writerow(dof_data)

        if self.config.save_hydro:
            hydro_data = logs.get("hydro", {})
            writer = self._open_writer("hydro", f"hydro{stamp}.csv", hydro_data.keys())
            writer.writerow(hydro_data)

        if self.config.save_wind_aero:
            wind_data = logs.get("wind_aero", {})
            writer = self._open_writer("wind", f"wind_aero{stamp}.csv", wind_data.keys())
            writer.writerow(wind_data)

        if self.config.save_moor:
            moor_data = logs.get("moor", {})
            writer = self._open_writer("moor", f"moor{stamp}.csv", moor_data.keys())
            writer.writerow(moor_data)

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files = {}
        self._writers = {}
