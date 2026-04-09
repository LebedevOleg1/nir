"""
Анимация неустойчивости Кельвина-Гельмгольца из VTK-файлов решателя.

Использование:
    python3 animate_kh.py                        # все output_*.vtk → kh_animation.gif
    python3 animate_kh.py --field VelocityX      # другое поле
    python3 animate_kh.py --out kh.mp4           # MP4 вместо GIF (нужен ffmpeg)
    python3 animate_kh.py --fps 15 --dpi 120     # настройки качества
    python3 animate_kh.py --vmin 0.8 --vmax 2.2  # фиксированный диапазон цветов

Доступные поля для --field:
    Density    (плотность)        — лучше всего видно вихри КГ
    VelocityX  (скорость по X)
    VelocityY  (скорость по Y)
    Pressure   (давление)
    Mach       (число Маха)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import glob
import os
import sys
import argparse
import struct


# =============================================================================
# Парсер бинарных VTK-файлов
# =============================================================================

def _read_line(f):
    line = b""
    while True:
        ch = f.read(1)
        if not ch or ch == b"\n":
            return line.decode("ascii", errors="replace").strip()
        line += ch


def read_vtk(filename):
    with open(filename, "rb") as f:
        nx = ny = None
        for _ in range(20):
            line = _read_line(f)
            if line.startswith("DIMENSIONS"):
                parts = line.split()
                nx, ny = int(parts[1]), int(parts[2])
                break
        if nx is None:
            raise ValueError(f"Не найден заголовок DIMENSIONS в {filename}")

        npts = nx * ny

        line = _read_line(f)
        raw = f.read(nx * 4)
        x_coords = np.frombuffer(raw, dtype=">f4").astype(np.float32)
        f.read(1)

        line = _read_line(f)
        raw = f.read(ny * 4)
        y_coords = np.frombuffer(raw, dtype=">f4").astype(np.float32)
        f.read(1)

        line = _read_line(f)
        f.read(4)
        f.read(1)

        line = _read_line(f)  # POINT_DATA npts

        fields = {}
        while True:
            line = _read_line(f)
            if not line:
                break
            if line.startswith("SCALARS"):
                parts = line.split()
                field_name = parts[1]
                _read_line(f)  # LOOKUP_TABLE default
                raw = f.read(npts * 4)
                if len(raw) < npts * 4:
                    break
                data = np.frombuffer(raw, dtype=">f4").astype(np.float32)
                fields[field_name] = data.reshape(ny, nx)
                f.read(1)

    return x_coords, y_coords, fields


# =============================================================================
# Анимация
# =============================================================================

def make_animation(vtk_files, field, out_file, fps, dpi, vmin_arg, vmax_arg):
    print(f"Поле: {field}, файлов: {len(vtk_files)}")

    x, y, flds = read_vtk(vtk_files[0])
    if field not in flds:
        print(f"Ошибка: поле '{field}' не найдено. Доступные: {list(flds.keys())}")
        sys.exit(1)

    nx, ny = len(x), len(y)
    print(f"Сетка: {nx} x {ny}")

    if vmin_arg is not None and vmax_arg is not None:
        vmin, vmax = vmin_arg, vmax_arg
    else:
        print("Сканирую файлы для диапазона цветов...")
        vmin, vmax = np.inf, -np.inf
        for i, fname in enumerate(vtk_files):
            _, _, flds = read_vtk(fname)
            d = flds[field]
            if d.min() < vmin: vmin = float(d.min())
            if d.max() > vmax: vmax = float(d.max())
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(vtk_files)}: [{vmin:.4f}, {vmax:.4f}]")
        print(f"Диапазон: [{vmin:.4f}, {vmax:.4f}]")

    aspect = nx / ny
    fig_w  = 8.0
    fig_h  = fig_w / aspect + 0.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    colormap = {
        "Density":   "plasma",
        "VelocityX": "RdBu_r",
        "VelocityY": "RdBu_r",
        "Pressure":  "inferno",
        "Mach":      "viridis",
    }.get(field, "viridis")

    if field in ("VelocityX", "VelocityY"):
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

    norm = Normalize(vmin=vmin, vmax=vmax)

    _, _, flds = read_vtk(vtk_files[0])
    im = ax.imshow(flds[field], origin="lower",
                   extent=[x[0], x[-1], y[0], y[-1]],
                   cmap=colormap, norm=norm, aspect="auto",
                   interpolation="bilinear")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    cbar.outline.set_edgecolor("white")
    labels = {"Density": "Плотность ρ", "VelocityX": "Скорость u",
              "VelocityY": "Скорость v", "Pressure": "Давление p",
              "Mach": "Число Маха"}
    cbar.set_label(labels.get(field, field), color="white")

    title = ax.set_title("", color="white", fontsize=11)
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("y", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    plt.tight_layout()

    def update(frame_idx):
        fname = vtk_files[frame_idx]
        _, _, flds = read_vtk(fname)
        im.set_data(flds[field])
        step_num = os.path.splitext(os.path.basename(fname))[0].replace("output_", "")
        title.set_text(f"Неустойчивость Кельвина–Гельмгольца  |  {field}  |  кадр {step_num}")
        return [im, title]

    print(f"Собираю анимацию ({len(vtk_files)} кадров)...")
    ani = animation.FuncAnimation(fig, update, frames=len(vtk_files),
                                  interval=1000 // fps, blit=True)

    ext = os.path.splitext(out_file)[1].lower()
    if ext == ".gif":
        ani.save(out_file, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    elif ext in (".mp4", ".mkv", ".avi"):
        ani.save(out_file,
                 writer=animation.FFMpegWriter(fps=fps, bitrate=3000,
                                               extra_args=["-vcodec", "libx264"]),
                 dpi=dpi)
    else:
        ani.save(out_file, writer=animation.PillowWriter(fps=fps), dpi=dpi)

    plt.close(fig)
    print(f"Готово: {out_file}  ({os.path.getsize(out_file)/1024/1024:.1f} МБ)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", default="Density",
                        choices=["Density", "VelocityX", "VelocityY", "Pressure", "Mach"])
    parser.add_argument("--out",  default="kh_animation.gif")
    parser.add_argument("--fps",  type=int,   default=15)
    parser.add_argument("--dpi",  type=int,   default=100)
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--dir",  default=".")
    args = parser.parse_args()

    vtk_files = sorted(glob.glob(os.path.join(args.dir, "output_*.vtk")))
    if not vtk_files:
        print(f"Ошибка: output_*.vtk не найдены в '{args.dir}'")
        sys.exit(1)

    print(f"Найдено {len(vtk_files)} VTK-файлов")
    make_animation(vtk_files, args.field, args.out, args.fps, args.dpi,
                   args.vmin, args.vmax)


if __name__ == "__main__":
    main()
