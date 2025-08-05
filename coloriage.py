#!/usr/bin/env python3
"""
color_by_number.py – Fiche « coloriage magique » + aperçu, **avec fusion** des zones
adjacentes partageant la même couleur de légende.

Nouveauté
---------
Les super-pixels (SLIC) qui se touchent *et* qui reçoivent le même numéro de
légende sont automatiquement fusionnés ; la fiche n’affiche donc qu’une seule
zone numérotée pour cet ensemble contigu. Cela évite les frontières inutiles et
facilite le coloriage.

Usage
-----
```bash
python color_by_number.py input.jpg sheet.png \
       --segments 350 --palette 10 \
       --preview preview.png --legend legend.png
```
La syntaxe reste identique, toute fusion étant effectuée automatiquement en
interne.

Dépendances
-----------
```bash
pip install numpy scikit-image scikit-learn pillow opencv-python
```

"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from skimage import io, segmentation, color, morphology
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont

# OpenCV (optionnel) pour dilatation rapide.
try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # Fallback sur skimage.
    cv2 = None  # type: ignore


# ╭─────────────────────────── Aide générale ───────────────────────────╮

class UnionFind:
    """Structure d’union-trouve simple (compression de chemin + union par rang)."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # union par rang
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def resize_if_needed(img: np.ndarray, max_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    if cv2 is not None:
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    from skimage.transform import resize  # fallback
    img_resized = resize(img, (new_size[1], new_size[0]), anti_aliasing=True)
    return (img_resized * 255).astype(np.uint8)


def slic_segmentation(img: np.ndarray, n_segments: int) -> np.ndarray:
    img_lab = color.rgb2lab(img)
    return segmentation.slic(img_lab, n_segments=n_segments,
                              compactness=10, start_label=0)


def region_means(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n_labels = labels.max() + 1
    means = np.zeros((n_labels, 3), dtype=float)
    for lbl in range(n_labels):
        means[lbl] = img[labels == lbl].mean(axis=0)
    return means


def quantize_palette(means: np.ndarray, k: int, random_state: int = 42):
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    palette_idx = km.fit_predict(means)
    palette_rgb = km.cluster_centers_.astype(int)
    return palette_idx, palette_rgb


def assign_palette(labels: np.ndarray, palette_idx: np.ndarray):
    return palette_idx[labels] + 1  # 1-base pour la légende


# ╭────────────────── Fusion des régions adjacentes même couleur ──────────────────╮

def merge_same_color_regions(labels: np.ndarray, numbers: np.ndarray):
    """Retourne (merged_labels, mapping merged_label → number)."""
    h, w = labels.shape
    uf = UnionFind(labels.max() + 1)

    # Union horizontal puis vertical si numéro identique.
    #  ─ horiz ─
    left = labels[:, :-1]
    right = labels[:, 1:]
    same_num_h = numbers[:, :-1] == numbers[:, 1:]
    mask_h = (left != right) & same_num_h
    for (y, x) in zip(*np.where(mask_h)):
        uf.union(left[y, x], right[y, x])
    #  ─ vert ─
    up = labels[:-1, :]
    down = labels[1:, :]
    same_num_v = numbers[:-1, :] == numbers[1:, :]
    mask_v = (up != down) & same_num_v
    for (y, x) in zip(*np.where(mask_v)):
        uf.union(up[y, x], down[y, x])

    # Construction des labels fusionnés.
    rep = np.array([uf.find(i) for i in range(labels.max() + 1)])
    merged_labels = rep[labels]

    # Mapping merged_label → number (suffit d’en récupérer un par label).
    region_to_number = np.zeros(rep.max() + 1, dtype=int)
    for lbl_old in range(rep.size):
        root = rep[lbl_old]
        region_to_number[root] = numbers.flat[np.argmax(labels == lbl_old)]

    return merged_labels, region_to_number


# ╭─────────────────────── Frontières & dessin ───────────────────────────╮

def generate_boundary_image(labels: np.ndarray, thickness: int) -> np.ndarray:
    boundaries = segmentation.find_boundaries(labels, mode="outer").astype(np.uint8) * 255
    if thickness > 1:
        if cv2 is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
            boundaries = cv2.dilate(boundaries, kernel, iterations=1)
        else:
            boundaries = morphology.dilation(boundaries, morphology.square(thickness))
    return boundaries


def draw_numbers(sheet: Image.Image, merged_labels: np.ndarray, region_to_number: np.ndarray):
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for lbl in np.unique(merged_labels):
        y, x = np.argwhere(merged_labels == lbl).mean(axis=0)
        draw.text((x, y), str(int(region_to_number[lbl])), fill=0, font=font, anchor="mm")


def save_legend(path: Path, palette_rgb: np.ndarray):
    n = len(palette_rgb)
    swatch = 50
    margin = 10
    w = swatch * n + margin * (n + 1)
    h = swatch + margin * 2 + 20
    legend = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(legend)
    font = ImageFont.load_default()
    for i, rgb in enumerate(palette_rgb):
        x = margin + i * (swatch + margin)
        y = margin
        d.rectangle([x, y, x + swatch, y + swatch], fill=tuple(map(int, rgb)), outline="black")
        d.text((x + swatch / 2, y + swatch + 2), str(i + 1), fill="black", font=font, anchor="ma")
    legend.save(path)


# ╭──────────────────────── Aperçu colorié ───────────────────────────────╮

def generate_preview_image(number_map: np.ndarray, palette_rgb: np.ndarray,
                           boundaries: np.ndarray | None = None) -> Image.Image:
    preview_rgb = palette_rgb[number_map - 1].astype(np.uint8)
    if boundaries is not None:
        preview_rgb[boundaries == 255] = [0, 0, 0]
    return Image.fromarray(preview_rgb, "RGB")


# ╭───────────────────────────── CLI ─────────────────────────────────────╮

def parse_args():
    p = argparse.ArgumentParser(description="Génère fiche color-by-number fusionnée + aperçu.")
    p.add_argument("input_image", type=Path)
    p.add_argument("output_image", type=Path)
    p.add_argument("--segments", type=int, default=3500)
    p.add_argument("--palette", type=int, default=8)
    p.add_argument("--max-size", type=int, default=1024)
    p.add_argument("--line-thickness", type=int, default=1)
    p.add_argument("--legend", type=Path)
    p.add_argument("--preview", type=Path, help="Chemin PNG de l’aperçu colorié")
    return p.parse_args()


def main():
    args = parse_args()

    # 1▸ Chargement & redimensionnement.
    img = io.imread(args.input_image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = resize_if_needed(img, args.max_size)

    # 2▸ Segmentation SLIC.
    labels = slic_segmentation(img, args.segments)
    means = region_means(img, labels)

    # 3▸ Palettisation.
    palette_idx, palette_rgb = quantize_palette(means, args.palette)
    number_map = assign_palette(labels, palette_idx)

    # 4▸ Fusion contiguë même numéro.
    merged_labels, region_to_number = merge_same_color_regions(labels, number_map)

    # 5▸ Frontières + fiche.
    boundaries = generate_boundary_image(merged_labels, args.line_thickness)
    sheet = Image.fromarray(255 - boundaries).convert("L")
    draw_numbers(sheet, merged_labels, region_to_number)
    sheet.save(args.output_image)
    print("✅ Fiche enregistrée →", args.output_image)

    # 6▸ Légende.
    if args.legend:
        save_legend(args.legend, palette_rgb)
        print("📋 Légende enregistrée →", args.legend)
    else:
        print("\nLégende suggérée (index : RGB) :")
        for i, rgb in enumerate(palette_rgb, 1):
            print(f" {i:2d} : ({rgb[0]:3.0f}, {rgb[1]:3.0f}, {rgb[2]:3.0f})")

    # 7▸ Aperçu colorié.
    if args.preview:
        preview = generate_preview_image(number_map, palette_rgb, boundaries)
        preview.save(args.preview)
        print("🎨 Aperçu colorié enregistré →", args.preview)


if __name__ == "__main__":
    sys.exit(main())
