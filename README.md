# Corso Pratico di Computer Vision

Modulo di laboratorio — 3 giorni, 4 ore al giorno.

## Struttura del corso

| Giorno | Topic | Notebook |
|--------|-------|----------|
| 1 | Object Detection & Multi-Object Tracking | `notebooks/day_1_detection_e_tracking.ipynb` |
| 2 | Vehicle/People Counting & Trajectory Analysis | `notebooks/day_2_counting_e_traiettorie.ipynb` |
| 3 | Image/Video Restoration & Caso Studio Targhe | `notebooks/day_3_restoration.ipynb` |

## Apri su Google Colab

Clicca sul link del giorno per aprire direttamente il notebook su Colab:

| Giorno | Link |
|--------|------|
| Giorno 1 — Detection & Tracking | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SalvoSamba01/Corso-Computer-Vision/blob/main/notebooks/day_1_detection_e_tracking.ipynb) |
| Giorno 2 — Counting & Traiettorie | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SalvoSamba01/Corso-Computer-Vision/blob/main/notebooks/day_2_counting_e_traiettorie.ipynb) |
| Giorno 3 — Restoration & OCR | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SalvoSamba01/Corso-Computer-Vision/blob/main/notebooks/day_3_restoration.ipynb) |

> Il notebook clona automaticamente il repo e installa le dipendenze alla prima esecuzione.

## Setup (Google Colab)

Ogni notebook clona automaticamente il repo e installa le dipendenze alla prima esecuzione. Nessun setup manuale richiesto.

## Struttura del repo

```
Corso-Computer-Vision/
├── notebooks/          # Notebook Jupyter per ogni giorno
├── data/
│   ├── day_1/          # Dataset MOT20, UA-DETRAC, PETS2009 (subset)
│   ├── day_2/          # Dataset CityFlow, ShanghaiTech (subset)
│   └── day_3/          # Dataset CCTV, CCPD (subset)
├── utils/
│   └── cv_utils.py     # Funzioni di supporto (visualizzazione, I/O video)
└── requirements.txt    # Dipendenze Python
```

## Dataset utilizzati

### Giorno 1
- [MOT20 Challenge](https://motchallenge.net/data/MOT20/) — pedestrian tracking
- [UA-DETRAC](https://www.kaggle.com/datasets/bratjay/ua-detrac-orig) — vehicle detection & tracking
- [PETS2009](https://www.kaggle.com/datasets/yeeandres/pets2009) — crowd surveillance

### Giorno 2
- [CityFlow](https://www.aicitychallenge.org/) — vehicle counting & re-identification
- [ShanghaiTech](https://www.kaggle.com/datasets/tthien/shanghaitech) — crowd counting

### Giorno 3
- [CCTV Action Recognition](https://www.kaggle.com/datasets/jonathannield/cctv-action-recognition-dataset) — CCTV a bassa risoluzione
- [CCPD](https://github.com/detectRecog/CCPD) — Chinese license plates

## Modelli utilizzati (solo inference, nessun retraining)

- **YOLOv8** (`ultralytics`) — object detection
- **BoxMOT** — multi-object tracking (ByteTrack, StrongSORT)
- **Supervision** (Roboflow) — annotazione, conteggio, zone
- **Norfair** — trajectory tracking & analysis
- **Real-ESRGAN** — super-resolution, denoising, deblur
- **YOLOv11** (HuggingFace) — license plate detection
- **EasyOCR** — lettura automatica targa (opzionale)
