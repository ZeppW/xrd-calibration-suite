from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


class AnnulusLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Annulus Mask Builder Launcher")
        self.geometry("860x420")

        self.mode = tk.StringVar(value="calibrated")
        self.poni = tk.StringVar()
        self.h5 = tk.StringVar()
        self.dataset = tk.StringVar(value="")
        self.frame = tk.StringVar(value="0")
        self.out_mask = tk.StringVar()
        self.spec_out = tk.StringVar(value="")
        self.vmin = tk.StringVar(value="")
        self.vmax = tk.StringVar(value="")

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}
        row = 0

        ttk.Label(self, text="Mode").grid(row=row, column=0, sticky="w", **pad)
        ttk.Combobox(self, textvariable=self.mode, values=["calibrated", "pixel"], width=16, state="readonly").grid(
            row=row, column=1, sticky="w", **pad
        )
        row += 1

        ttk.Label(self, text="PONI File").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.poni, width=80).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self, text="Browse", command=self._pick_poni).grid(row=row, column=2, **pad)
        row += 1

        ttk.Label(self, text="HDF5 File").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.h5, width=80).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self, text="Browse", command=self._pick_h5).grid(row=row, column=2, **pad)
        row += 1

        ttk.Label(self, text="Output Mask (.mask)").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.out_mask, width=80).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self, text="Save As", command=self._pick_out_mask).grid(row=row, column=2, **pad)
        row += 1

        ttk.Label(self, text="Spec JSON (optional)").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.spec_out, width=80).grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self, text="Save As", command=self._pick_spec).grid(row=row, column=2, **pad)
        row += 1

        ttk.Label(self, text="HDF5 Dataset Path (optional)").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.dataset, width=24).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(self, text="Frame").grid(row=row, column=1, sticky="e", **pad)
        ttk.Entry(self, textvariable=self.frame, width=8).grid(row=row, column=2, sticky="w", **pad)
        row += 1

        ttk.Label(self, text="Display Vmin (optional)").grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(self, textvariable=self.vmin, width=12).grid(row=row, column=1, sticky="w", **pad)
        ttk.Label(self, text="Display Vmax (optional)").grid(row=row, column=1, sticky="e", **pad)
        ttk.Entry(self, textvariable=self.vmax, width=12).grid(row=row, column=2, sticky="w", **pad)
        row += 1

        actions = ttk.Frame(self)
        actions.grid(row=row, column=0, columnspan=3, pady=14)
        ttk.Button(actions, text="Run Builder", command=self._run).grid(row=0, column=0, padx=10)
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
            if not self.out_mask.get():
                self.out_mask.set(str(Path(p).with_suffix(".mask")))

    def _pick_out_mask(self):
        p = filedialog.asksaveasfilename(
            title="Save mask as",
            defaultextension=".mask",
            filetypes=[("Mask/TIFF", "*.mask *.tif *.tiff"), ("All files", "*.*")],
        )
        if p:
            self.out_mask.set(p)

    def _pick_spec(self):
        p = filedialog.asksaveasfilename(
            title="Save annulus specs JSON as",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if p:
            self.spec_out.set(p)

    def _run(self):
        try:
            poni = Path(self.poni.get().strip())
            h5 = Path(self.h5.get().strip())
            out = Path(self.out_mask.get().strip())
            if not poni.exists():
                raise FileNotFoundError(f"PONI file not found: {poni}")
            if not h5.exists():
                raise FileNotFoundError(f"HDF5 file not found: {h5}")
            if out.parent and (not out.parent.exists()):
                out.parent.mkdir(parents=True, exist_ok=True)

            script_path = Path(__file__).with_name("interactive_annulus_mask_builder.py")
            cmd = [
                sys.executable,
                str(script_path),
                "--mode",
                self.mode.get().strip(),
                "--poni",
                str(poni),
                "--h5",
                str(h5),
                "--frame",
                str(int(self.frame.get().strip())),
                "--out",
                str(out),
            ]
            ds = self.dataset.get().strip()
            if ds:
                cmd.extend(["--dataset", ds])
            spec = self.spec_out.get().strip()
            if spec:
                cmd.extend(["--spec-out", spec])
            vmin = self.vmin.get().strip()
            vmax = self.vmax.get().strip()
            if vmin:
                cmd.extend(["--vmin", vmin])
            if vmax:
                cmd.extend(["--vmax", vmax])

            subprocess.run(cmd, check=True)
        except Exception as e:
            messagebox.showerror("Run failed", str(e))


def main():
    app = AnnulusLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
