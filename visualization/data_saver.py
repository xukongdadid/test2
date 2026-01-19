import csv
import datetime
import os


class DataSaver:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.folder = ""
        self.save_config = {}
        self.csv_files = {}
        self.csv_writers = {}

    def start(self, save_config):
        self.close()
        self.save_config = save_config or {}
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
        folder = os.path.join(self.base_dir, f"Run_{timestamp}")
        suffix = 1
        while os.path.exists(folder):
            folder = os.path.join(self.base_dir, f"Run_{timestamp}_{suffix:02d}")
            suffix += 1
        os.makedirs(folder, exist_ok=True)
        self.folder = folder

        filenames = {
            "Structure Response": f"dof{timestamp}.csv",
            "Wave/Current Loads": f"hydro{timestamp}.csv",
            "Wind/Aero Loads": f"wind_aero{timestamp}.csv",
            "Mooring Loads": f"moor{timestamp}.csv",
        }

        for group, fn in filenames.items():
            if not self.save_config.get(group):
                self.csv_files[group] = None
                self.csv_writers[group] = None
                continue
            path = os.path.join(folder, fn)
            try:
                self.csv_files[group] = open(path, "w", newline="")
                self.csv_writers[group] = None
            except Exception:
                self.csv_files[group] = None
                self.csv_writers[group] = None

        return self.folder

    def write(self, data_frame):
        for group in list(self.csv_files.keys()):
            csv_file = self.csv_files.get(group)
            if not csv_file:
                continue
            if self.csv_writers.get(group) is None:
                fields = ["time"]
                for k in self.save_config.get(group, []):
                    if k not in fields:
                        fields.append(k)
                writer = csv.DictWriter(csv_file, fieldnames=fields)
                writer.writeheader()
                self.csv_writers[group] = writer
            else:
                writer = self.csv_writers[group]

            row = {"time": data_frame.get("time", data_frame.get("Time", 0.0))}
            for k in self.save_config.get(group, []):
                if k in data_frame:
                    row[k] = data_frame[k]
            try:
                writer.writerow(row)
            except Exception:
                pass

    def close(self):
        for f in self.csv_files.values():
            try:
                if f:
                    f.close()
            except Exception:
                pass
        self.csv_files = {}
        self.csv_writers = {}
        self.folder = ""
