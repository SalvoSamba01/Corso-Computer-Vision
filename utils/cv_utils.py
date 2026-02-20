"""
cv_utils.py — Funzioni di supporto per il corso di Computer Vision
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from IPython.display import display, Image as IPImage
import io


# ─── Visualizzazione ──────────────────────────────────────────────────────────

def mostra_frame(frame: np.ndarray, titolo: str = "", figsize=(12, 7)):
    """Mostra un frame BGR (OpenCV) in un notebook Jupyter."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(rgb)
    plt.axis("off")
    if titolo:
        plt.title(titolo, fontsize=13)
    plt.tight_layout()
    plt.show()


def mostra_confronto(img_sinistra: np.ndarray, img_destra: np.ndarray,
                     titolo_sx: str = "Originale", titolo_dx: str = "Risultato",
                     figsize=(16, 7)):
    """Mostra due immagini affiancate per confronto."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, img, titolo in zip(axes, [img_sinistra, img_destra], [titolo_sx, titolo_dx]):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
        ax.imshow(rgb, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(titolo, fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def mostra_griglia(frames: list, titoli: list = None, colonne: int = 3, figsize=None):
    """Mostra una griglia di frame/immagini."""
    n = len(frames)
    righe = (n + colonne - 1) // colonne
    if figsize is None:
        figsize = (colonne * 5, righe * 4)
    fig, axes = plt.subplots(righe, colonne, figsize=figsize)
    axes = np.array(axes).flatten()
    for i, (ax, frame) in enumerate(zip(axes, frames)):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else frame
        ax.imshow(rgb, cmap="gray" if frame.ndim == 2 else None)
        if titoli and i < len(titoli):
            ax.set_title(titoli[i], fontsize=10)
        ax.axis("off")
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()


# ─── Video I/O ────────────────────────────────────────────────────────────────

def info_video(path: str) -> dict:
    """Restituisce informazioni su un file video."""
    cap = cv2.VideoCapture(path)
    info = {
        "path": path,
        "nome": Path(path).name,
        "larghezza": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "altezza": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "n_frame": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "durata_s": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
    }
    cap.release()
    return info


def stampa_info_video(path: str):
    """Stampa a schermo le informazioni del video."""
    info = info_video(path)
    print(f"  File    : {info['nome']}")
    print(f"  Risoluz.: {info['larghezza']}x{info['altezza']} px")
    print(f"  FPS     : {info['fps']:.1f}")
    print(f"  Frame   : {info['n_frame']}")
    print(f"  Durata  : {info['durata_s']:.1f} s")


def estrai_frame(path: str, n: int = 1, ogni_n: int = None) -> list:
    """
    Estrae frame da un video.
    - n=1, ogni_n=None → primo frame
    - ogni_n=30        → un frame ogni 30
    """
    cap = cv2.VideoCapture(path)
    frames = []
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if ogni_n is not None:
        indici = list(range(0, tot, ogni_n))[:n] if n else list(range(0, tot, ogni_n))
    else:
        indici = [0] if n == 1 else list(range(0, min(n, tot)))

    for idx in indici:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def video_in_mp4(input_path: str, output_path: str, fps: float = None) -> str:
    """Ricodifica un video in MP4 (H.264) compatibile con Colab."""
    cap = cv2.VideoCapture(input_path)
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()
    return output_path


# ─── Degradazione controllata ────────────────────────────────────────────────

def riduci_risoluzione(img: np.ndarray, scala: float) -> np.ndarray:
    """Riduce la risoluzione di img e la riporta alla dimensione originale."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scala)), max(1, int(h * scala))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def applica_blur(img: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Applica blur gaussiano. kernel_size deve essere dispari."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def aggiungi_rumore(img: np.ndarray, intensita: float = 25.0) -> np.ndarray:
    """Aggiunge rumore gaussiano all'immagine."""
    rumore = np.random.normal(0, intensita, img.shape).astype(np.float32)
    degradata = np.clip(img.astype(np.float32) + rumore, 0, 255).astype(np.uint8)
    return degradata


def degrada_immagine(img: np.ndarray, scala: float = 0.5,
                     blur: int = 0, rumore: float = 0.0) -> np.ndarray:
    """Pipeline di degradazione completa (risoluzione → blur → rumore)."""
    out = img.copy()
    if scala < 1.0:
        out = riduci_risoluzione(out, scala)
    if blur > 0:
        out = applica_blur(out, blur)
    if rumore > 0:
        out = aggiungi_rumore(out, rumore)
    return out


# ─── Heatmap & Traiettorie ────────────────────────────────────────────────────

def crea_heatmap(punti: list, shape: tuple, sigma: int = 20) -> np.ndarray:
    """
    Genera una heatmap dai punti (x, y) usando un accumulatore gaussiano.
    shape = (altezza, larghezza)
    """
    heatmap = np.zeros(shape[:2], dtype=np.float32)
    for x, y in punti:
        xi, yi = int(x), int(y)
        if 0 <= xi < shape[1] and 0 <= yi < shape[0]:
            heatmap[yi, xi] += 1.0
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap, sigma=sigma)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def sovrapponi_heatmap(frame: np.ndarray, heatmap: np.ndarray,
                       alpha: float = 0.5) -> np.ndarray:
    """Sovrappone una heatmap normalizzata [0,1] su un frame BGR."""
    hm_uint8 = (heatmap * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1 - alpha, hm_color, alpha, 0)


# ─── Detection & Tracking — annotazione ─────────────────────────────────────

_COLORI_CLASSE = {
    0: (0, 255, 0),    # person     → verde
    1: (255, 165, 0),  # bicycle    → arancione
    2: (0, 0, 255),    # car        → rosso
    3: (255, 0, 255),  # motorcycle → magenta
    5: (0, 165, 255),  # bus        → arancione chiaro
    7: (0, 80, 255),   # truck      → blu scuro
}


def disegna_detection(frame: np.ndarray, risultati,
                      soglia_conf: float = 0.3,
                      classi_filtro: list = None,
                      nomi_classi: dict = None) -> np.ndarray:
    """
    Disegna i bounding box delle detection YOLO su un frame.

    Args:
        frame: immagine BGR
        risultati: output di YOLO (lista di Results)
        soglia_conf: confidenza minima per mostrare una detection
        classi_filtro: lista di ID classe da visualizzare (None = tutte)
        nomi_classi: dict {id: nome}, tipicamente model.names (opzionale)
    """
    out = frame.copy()
    nomi = nomi_classi or {}
    for box in risultati[0].boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        if conf < soglia_conf:
            continue
        if classi_filtro is not None and cls_id not in classi_filtro:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        colore = _COLORI_CLASSE.get(cls_id, (200, 200, 200))
        nome = nomi.get(cls_id, str(cls_id))
        cv2.rectangle(out, (x1, y1), (x2, y2), colore, 2)
        cv2.putText(out, f'{nome} {conf:.2f}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colore, 2)
    return out


_PALETTE_TRACKS: np.ndarray = None


def _get_palette(n: int = 200) -> np.ndarray:
    global _PALETTE_TRACKS
    if _PALETTE_TRACKS is None:
        rng = np.random.default_rng(42)
        _PALETTE_TRACKS = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    return _PALETTE_TRACKS


def disegna_tracks(frame: np.ndarray, tracks: np.ndarray,
                   id_target: int = None) -> np.ndarray:
    """
    Disegna i bounding box con ID del tracker su un frame.

    tracks shape: (N, 7+) — [x1, y1, x2, y2, track_id, conf, cls, ...]

    Args:
        frame: immagine BGR
        tracks: array output del tracker (BoxMOT / ByteTrack)
        id_target: se specificato, evidenzia solo questo ID; gli altri in grigio
    """
    out = frame.copy()
    if tracks is None or len(tracks) == 0:
        return out
    palette = _get_palette()
    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        tid = int(track[4])
        if id_target is not None and tid != id_target:
            cv2.rectangle(out, (x1, y1), (x2, y2), (130, 130, 130), 1)
        else:
            c = palette[tid % len(palette)]
            colore = (int(c[0]), int(c[1]), int(c[2]))
            spessore = 3 if id_target is not None else 2
            cv2.rectangle(out, (x1, y1), (x2, y2), colore, spessore)
            label = f'★ ID:{tid}' if id_target is not None else f'ID:{tid}'
            cv2.putText(out, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colore, 2)
    return out


# ─── Metriche ─────────────────────────────────────────────────────────────────

def psnr(img_ref: np.ndarray, img_test: np.ndarray) -> float:
    """Calcola il PSNR (Peak Signal-to-Noise Ratio) tra due immagini."""
    mse = np.mean((img_ref.astype(np.float64) - img_test.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))
