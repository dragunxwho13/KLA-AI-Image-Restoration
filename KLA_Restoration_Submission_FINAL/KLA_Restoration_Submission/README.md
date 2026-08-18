# KLA AI Hackathon - AI-Based Restoration of Degraded Images

This submission contains a compact fully-convolutional grayscale restoration model for the supplied KLA degraded-image dataset.

The model performs learned 2x restoration/upscaling and is designed for an offline command-line evaluator.

## Submission Layout

```text
team_name/
|-- run.py
|-- requirements.txt
|-- README.md
`-- models/
    `-- restoration_model.pth
```

## Run

```bash
python run.py <input-dir> <output-dir>
```

`run.py` reads every `.npy` file directly inside the input directory, creates the output directory when required, restores each image at 2x spatial resolution, and writes an output `.npy` using the same filename.

## Input / Output Contract

### Input

- NumPy `.npy` file
- Grayscale array of shape `(H,W)` or `(H,W,1)`
- Float32 values
- The supplied evaluation inputs may contain values below 0 or above 1.

### Output

- NumPy `.npy` file
- Grayscale float32 array
- Shape `(2H,2W)`
- Values constrained to `[0,1]`
- NaN and Inf values are sanitized
- Input filenames are preserved

For the supplied 400-file evaluation set, the implementation produces 256x256 outputs from 128x128 inputs.

## Model

The restoration network is a compact residual convolutional neural network.

Main components:

- Initial convolutional feature extraction
- Residual feature blocks
- Feature fusion
- PixelShuffle-based 2x upsampling
- Final grayscale reconstruction layer

The architecture is fully convolutional and supports different spatial input sizes.

## Training

The supplied paired training set contains:

- 50 degraded `.npy` images at 256x256
- 50 corresponding ground-truth grayscale PNG images at 512x512

Training uses:

- L1 reconstruction loss
- MSE reconstruction loss
- Random rotations and flips
- Synthetic additive Gaussian noise augmentation
- Synthetic multiplicative/speckle noise augmentation
- AdamW optimization

The bundled model weights are stored in:

`models/restoration_model.pth`

## Inference Scaling

The inference implementation detects the numerical scale of each input array.

Arrays consistent with the training 0-255 representation are normalized using a 255 scale factor. Arrays already in the released normalized range are processed without applying that 255 factor.

## Offline / Runtime Design

The submission does not require:

- Internet access
- API keys
- External model downloads
- Interactive user input

CUDA is automatically used when an NVIDIA GPU is available. CPU fallback is included.

## Reproducibility

All required model weights are bundled locally.

The evaluator only needs the supplied Python dependencies and the submission directory. No external network resource is required.
