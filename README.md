# Legacy simulation-based inference

This repository preserves the legacy likelihood-free inference implementation removed from NMMA.

The code was originally integrated into NMMA by commit `9e88cffb` and provided a PyTorch similarity embedding plus an `nflows` normalizing flow for `Ka2017` kilonova light curves. It was invoked through `lightcurve-analysis --sampler neuralnet`.

## Contents

- `nmma/mlmodel/`: legacy model, preprocessing, inference, and training modules
- `nmma/mlmodel/*.pth`: trained embedding and normalizing-flow weights
- `doc/lfi_analysis.md`: the original usage documentation

This is archival code. It is preserved for reproducibility and historical reference, and is not part of the maintained NMMA runtime.

## Original dependencies

- Python
- PyTorch
- nflows
- torchvision
- NMMA/Bilby for the original command-line integration
