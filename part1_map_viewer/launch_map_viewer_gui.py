from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from view_map_with_1d import launch_simple_viewer


class MapViewerLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("XRD Map Viewer (Simple Launcher)")
        self.geometry("860x220")

        self.root_dir = tk.StringVar()
        self.map_npy = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}
        row = 0

        ttk.Label(self, text="Root Dataset Folder").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.root_dir, width=80).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self, text="Browse", command=self._pick_root).grid(row=row, column=2, **pad)
        row += 1

        ttk.Label(self, text="Custom Map NPY").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.map_npy, width=80).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self, text="Browse", command=self._pick_map_npy).grid(row=row, column=2, **pad)
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

    def _pick_map_npy(self):
        p = filedialog.askopenfilename(
            title="Select Custom Map NPY",
            filetypes=[("NumPy", "*.npy"), ("All files", "*.*")],
        )
        if p:
            self.map_npy.set(p)

    def _run(self):
        try:
            root_dir = Path(self.root_dir.get().strip())
            if not root_dir.exists():
                raise FileNotFoundError(f"Root dataset folder not found: {root_dir}")

            map_npy = Path(self.map_npy.get().strip())
            if not map_npy.exists():
                raise FileNotFoundError(f"Map NPY not found: {map_npy}")
            if map_npy.suffix.lower() != ".npy":
                raise ValueError("Custom map file must be a .npy file.")

            launch_simple_viewer(
                root_dir=root_dir,
                map_npy=map_npy,
            )
        except Exception as e:
            messagebox.showerror("Run failed", str(e))


def main():
    app = MapViewerLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
