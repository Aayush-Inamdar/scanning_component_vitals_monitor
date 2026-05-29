# Scanning Component — Vitals Monitor

Camera-based vitals monitoring using remote photoplethysmography (rPPG). Estimates heart rate, breathing rate, HRV (RMSSD), blood pressure, and cardiac load (rate pressure product) from a webcam feed.

## Requirements

- Python 3.10+
- Webcam
- Dependencies listed in `requirements.txt`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Run

From the project root:

```bash
python main.py
```

- Vitals are sampled every **5 seconds** when signal quality (SQI) is above **0.3**.
- Press **q** in the preview window or **Ctrl+C** in the terminal to quit.
- Measurements append to `data/vitals_log.csv`.

## Project layout

| Path | Role |
|------|------|
| `main.py` | Application entry point |
| `config/settings.py` | Paths, thresholds, and constants |
| `models/vitals_record.py` | `VitalsRecord` dataclass |
| `services/camera_service.py` | `VitalsMonitor` camera loop |
| `services/bp_estimator.py` | BP estimation from BVP |
| `services/cardiac.py` | Cardiac load (RPP) |
| `services/logger.py` | CSV init and append |
| `data/vitals_log.csv` | Output log (created on first run) |
| `tests/` | Unit tests |

## Disclaimer

Blood pressure values are **estimates** derived from the BVP waveform, not clinical measurements. Do not use for medical diagnosis.
