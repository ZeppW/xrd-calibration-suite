from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from view_map_with_1d import launch_viewer


class MapViewerLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("XRD Map Viewer Launcher")
        self.geometry("860x520")

        self.root_dir = tk.StringVar()
        self.out_dir = tk.StringVar()
        self.sp1d_dir = tk.StringVar()
        self.map_file = tk.StringVar()
        self.metric = tk.StringVar(value="ni_score")
        self.metrics_filename = tk.StringVar(value="xrd_metrics_ni.csv")
        self.maps_filename = tk.StringVar(value="xrd_maps_ni.npz")
        self.right_mode = tk.StringVar(value="spectrum")
        self.dataset_name = tk.StringVar(value="data")
        self.col_shift = tk.StringVar(value="0")
        self.yscale = tk.StringVar(value="linear")
        self.x_min = tk.StringVar(value="")
        self.x_max = tk.StringVar(value="")
        self.show_downsample = tk.StringVar(value="1")
        self.flip_ud = tk.BooleanVar(value=False)
        self.flip_lr = tk.BooleanVar(value=False)
        self.flip_right_ud = tk.BooleanVar(value=False)
        self.flip_right_lr = tk.BooleanVar(value=False)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}
        row = 0

        ttk.Label(self, text="Root Dataset Folder").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.root_dir, width=80).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self, text="Browse", command=self._pick_root).grid(row=row, column=2, **pad)
        row += 1

        ttk.Label(self, text="Output Folder").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.out_dir, width=80).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self, text="Browse", command=self._pick_out_dir).grid(row=row, column=2, **pad)
        row += 1

        ttk.Label(self, text="1D XY Folder").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.sp1d_dir, width=80).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self, text="Browse", command=self._pick_sp1d_dir).grid(row=row, column=2, **pad)
        row += 1

        ttk.Label(self, text="Custom Map File (optional)").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.map_file, width=80).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self, text="Browse", command=self._pick_map_file).grid(row=row, column=2, **pad)
        row += 1

        ttk.Label(self, text="Metric Name").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.metric, width=18).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(self, text="Right Panel").grid(row=row, column=1, sticky="e", **pad)
        ttk.Combobox(self, textvariable=self.right_mode, values=["spectrum", "image"], width=12, state="readonly").grid(
            row=row, column=2, sticky="w", **pad
        )
        row += 1

        ttk.Label(self, text="Metrics CSV Filename").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.metrics_filename, width=30).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(self, text="Maps NPZ Filename").grid(row=row, column=1, sticky="e", **pad)
        ttk.Entry(self, textvariable=self.maps_filename, width=18).grid(row=row, column=2, sticky="w", **pad)
        row += 1

        ttk.Label(self, text="Dataset Name (right image mode)").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.dataset_name, width=18).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(self, text="Downsample").grid(row=row, column=1, sticky="e", **pad)
        ttk.Entry(self, textvariable=self.show_downsample, width=8).grid(row=row, column=2, sticky="w", **pad)
        row += 1

        ttk.Label(self, text="Y Scale").grid(row=row, column=0, sticky="w", **pad)
        ttk.Combobox(self, textvariable=self.yscale, values=["linear", "log"], width=10, state="readonly").grid(
            row=row, column=1, sticky="w", **pad
        )
        ttk.Label(self, text="Col Shift").grid(row=row, column=1, sticky="e", **pad)
        ttk.Entry(self, textvariable=self.col_shift, width=8).grid(row=row, column=2, sticky="w", **pad)
        row += 1

        ttk.Label(self, text="X Min (optional)").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.x_min, width=12).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(self, text="X Max (optional)").grid(row=row, column=1, sticky="e", **pad)
        ttk.Entry(self, textvariable=self.x_max, width=12).grid(row=row, column=2, sticky="w", **pad)
        row += 1

        flips = ttk.LabelFrame(self, text="Display Flips")
        flips.grid(row=row, column=0, columnspan=3, sticky="we", padx=8, pady=8)
        ttk.Checkbutton(flips, text="Flip left map UD", variable=self.flip_ud).grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(flips, text="Flip left map LR", variable=self.flip_lr).grid(row=0, column=1, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(flips, text="Flip right image UD", variable=self.flip_right_ud).grid(row=0, column=2, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(flips, text="Flip right image LR", variable=self.flip_right_lr).grid(row=0, column=3, sticky="w", padx=8, pady=4)
        row += 1

        actions = ttk.Frame(self)
        actions.grid(row=row, column=0, columnspan=3, pady=12)
        ttk.Button(actions, text="Run Viewer", command=self._run).grid(row=0, column=0, padx=10)
        ttk.Button(actions, text="Close", command=self.destroy).grid(row=0, column=1, padx=10)

        self.grid_columnconfigure(1, weight=1)

    def _pick_root(self):
        p = filedialog.askdirectory(title="Select Root Dataset Folder")
        if p:
            self.root_dir.set(p)
            out = Path(p) / "hdf5_images_output"
            if not self.out_dir.get():
                self.out_dir.set(str(out))
            if not self.sp1d_dir.get():
                self.sp1d_dir.set(str(out / "linecut" / "allsp"))

    def _pick_out_dir(self):
        p = filedialog.askdirectory(title="Select Output Folder")
        if p:
            self.out_dir.set(p)
            if not self.sp1d_dir.get():
                self.sp1d_dir.set(str(Path(p) / "linecut" / "allsp"))

    def _pick_sp1d_dir(self):
        p = filedialog.askdirectory(title="Select 1D XY Folder")
        if p:
            self.sp1d_dir.set(p)

    def _pick_map_file(self):
        p = filedialog.askopenfilename(
            title="Select Map File",
            filetypes=[
                ("NumPy / image", "*.npy *.npz *.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if p:
            self.map_file.set(p)

    def _float_or_none(self, text: str):
        t = text.strip()
        return None if t == "" else float(t)

    def _run(self):
        try:
            root_dir = Path(self.root_dir.get().strip())
            out_dir = Path(self.out_dir.get().strip())
            sp1d_dir = Path(self.sp1d_dir.get().strip())
            if not root_dir.exists():
                raise FileNotFoundError(f"Root dataset folder not found: {root_dir}")
            if not out_dir.exists():
                raise FileNotFoundError(f"Output folder not found: {out_dir}")
            if self.right_mode.get() == "spectrum" and (not sp1d_dir.exists()):
                raise FileNotFoundError(f"1D XY folder not found: {sp1d_dir}")

            map_file = self.map_file.get().strip()
            map_path = Path(map_file) if map_file else None
            if map_path is not None and (not map_path.exists()):
                raise FileNotFoundError(f"Map file not found: {map_path}")

            launch_viewer(
                root_dir=root_dir,
                out_dir=out_dir,
                sp1d_dir=sp1d_dir,
                metric=self.metric.get().strip(),
                metrics_filename=self.metrics_filename.get().strip(),
                maps_filename=self.maps_filename.get().strip(),
                map_file=map_path,
                ny=31,
                nx=33,
                col_shift=int(self.col_shift.get().strip()),
                yscale=self.yscale.get().strip(),
                x_min=self._float_or_none(self.x_min.get()),
                x_max=self._float_or_none(self.x_max.get()),
                right_mode=self.right_mode.get().strip(),
                dataset_name=self.dataset_name.get().strip(),
                show_downsample=max(1, int(self.show_downsample.get().strip())),
                flip_ud=bool(self.flip_ud.get()),
                flip_lr=bool(self.flip_lr.get()),
                flip_right_ud=bool(self.flip_right_ud.get()),
                flip_right_lr=bool(self.flip_right_lr.get()),
            )
        except Exception as e:
            messagebox.showerror("Run failed", str(e))


def main():
    app = MapViewerLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
