# KLA AI Hackathon: AI-Based Restoration of Degraded Images

A compact, fully-convolutional neural network that restores and **2x upscales** degraded grayscale
semiconductor inspection images. Built for the KLA AI Hackathon (Semicon Hackathon 2026) and designed to run as an
offline command-line evaluator.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-model-EE4C2C?logo=pytorch&logoColor=white)
![Offline](https://img.shields.io/badge/runtime-fully%20offline-6e7781)

## Quick start

The submission lives in
[`KLA_Restoration_Submission_FINAL/KLA_Restoration_Submission/`](KLA_Restoration_Submission_FINAL/KLA_Restoration_Submission).

```bash
cd KLA_Restoration_Submission_FINAL/KLA_Restoration_Submission
pip install -r requirements.txt
python run.py <input-dir> <output-dir>
```

`run.py` reads every `.npy` file in the input directory, restores each image at 2x resolution and writes a
same-named `.npy` to the output directory. For the supplied 400-file evaluation set, 128x128 inputs become
256x256 outputs.

## Approach

- **Model:** a compact residual CNN: feature extraction, residual blocks, feature fusion, **PixelShuffle** 2x
  upsampling and a final grayscale reconstruction layer. Being fully convolutional, it accepts any input size.
- **Training data:** 50 degraded 256x256 `.npy` images paired with 50 ground-truth 512x512 PNGs.
- **Training setup:** L1 + MSE loss, AdamW, random rotations and flips, and synthetic Gaussian and speckle noise
  augmentation to make a small dataset go further.
- **Input scaling:** inference detects whether an array is on the 0–255 training scale or already normalised, and
  scales accordingly, so out-of-range values are handled safely.
- **Output contract:** float32, shape `(2H, 2W)`, values clamped to `[0, 1]`, NaN/Inf sanitised, filenames preserved.

## Why it's easy to run

No internet access, API keys, external model downloads or interactive input. Weights are bundled in
`models/restoration_model.pth`, and the code uses CUDA when available with a CPU fallback.

## Repository layout

```text
KLA_Restoration_Submission_FINAL/
└── KLA_Restoration_Submission/
    ├── run.py             # CLI entry point (inference)
    ├── requirements.txt
    ├── README.md          # full submission notes: I/O contract, model, training
    ├── models/            # bundled weights
    └── training/          # training code
```

See the [full submission README](KLA_Restoration_Submission_FINAL/KLA_Restoration_Submission/README.md) for the
complete input/output contract and training details.

---

<sub>Built by <a href="https://github.com/dragunxwho13">Aniket Khuntia</a> for the KLA AI Hackathon.</sub>
