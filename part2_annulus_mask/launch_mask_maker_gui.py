from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


class MaskMakerLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mask Maker Launcher (Simple)")
        self.geometry("880x210")

        self.poni = tk.StringVar()
        self.h5 = tk.StringVar()
        self.save_dir = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}
        row = 0

        ttk.Label(self, text="PONI File").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.poni, width=82).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self, text="Browse", command=self._pick_poni).grid(row=row, column=2, **pad)
        row += 1

        ttk.Label(self, text="HDF5 File").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.h5, width=82).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self, text="Browse", command=self._pick_h5).grid(row=row, column=2, **pad)
        row += 1

        ttk.Label(self, text="Save Directory").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.save_dir, width=82).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self, text="Browse", command=self._pick_save_dir).grid(row=row, column=2, **pad)
        row += 1

        help_text = (
            "Defaults: mode=calibrated, frame=0, dataset=auto. "
            "Outputs are written to the selected save directory."
        )
        ttk.Label(self, text=help_text).grid(row=row, column=0, columnspan=3, sticky="w", **pad)
        row += 1

        actions = ttk.Frame(self)
        actions.grid(row=row, column=0, columnspan=3, pady=12)
        ttk.Button(actions, text="Run Mask Maker", command=self._run).grid(row=0, column=0, padx=10)
        ttk.Button(actions, text="Close", command=self.destroy).grid(row=0, column=1, padx=10)

        self.grid_columnconfigure(1, weight=1)

    def _pick_poni(self):
        p = filedialog.askopenfilename(title="Select .poni file", filetypes=[("PONI", "*.poni"), ("All files", "*.*")])
        if p:
            self.poni.set(p)

    def _pick_h5(self):
        p = filedialog.askopenfilename(
            title="Select HDF5 file",
            filetypes=[("HDF5", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
        if p:
            self.h5.set(p)
            if not self.save_dir.get().strip():
                self.save_dir.set(str(Path(p).parent))

    def _pick_save_dir(self):
        p = filedialog.askdirectory(title="Select Output Directory for Mask/JSON")
        if p:
            self.save_dir.set(p)

    def _run(self):
        try:
            poni = Path(self.poni.get().strip())
            h5 = Path(self.h5.get().strip())
            if not poni.exists():
                raise FileNotFoundError(f"PONI file not found: {poni}")
            if not h5.exists():
                raise FileNotFoundError(f"HDF5 file not found: {h5}")

            save_dir_raw = self.save_dir.get().strip()
            save_dir = Path(save_dir_raw) if save_dir_raw else h5.parent
            save_dir.mkdir(parents=True, exist_ok=True)

            out_mask = save_dir / f"{h5.stem}.mask"
            out_spec = save_dir / f"{h5.stem}.json"

            script_path = Path(__file__).with_name("interactive_annulus_mask_builder.py")
            cmd = [
                sys.executable,
                str(script_path),
                "--mode",
                "calibrated",
                "--poni",
                str(poni),
                "--h5",
                str(h5),
                "--frame",
                "0",
                "--out",
                str(out_mask),
                "--spec-out",
                str(out_spec),
            ]
            subprocess.run(cmd, check=True)
            messagebox.showinfo("Mask maker finished", f"Saved:\n{out_mask}\n{out_spec}")
        except Exception as e:
            messagebox.showerror("Run failed", str(e))


def main():
    app = MaskMakerLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
