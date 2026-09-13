# Legacy simulation-based inference

This repository preserves the legacy likelihood-free inference implementation removed from NMMA.

The code was originally integrated into NMMA by commit `9e88cffb` and provided a PyTorch similarity embedding plus an `nflows` normalizing flow for `Ka2017` kilonova light curves. It was invoked through `lightcurve-analysis --sampler neuralnet`.

## Contents

- `nmma/mlmodel/`: legacy model, preprocessing, inference, and training modules
- `nmma/mlmodel/*.pth`: trained embedding and normalizing-flow weights
- `doc/lfi_analysis.md`: the original usage documentation

This is archival code. It is preserved for reproducibility and historical reference, and is not part of the maintained NMMA runtime.

## Install as a standalone package

::: [!note]
In general, this package is not actively maintained and is not recommended for new development. It is provided for archival purposes and to support reproducibility of legacy workflows.
:::

From a checkout, install the model library with:

```bash
python -m pip install .
```

This installs the preprocessing, embedding, and normalizing-flow modules and includes the bundled trained weights. Bilby is only needed for the result-conversion helper, and can be installed with:

```bash
python -m pip install ".[bilby]"
```

The package does not provide the NMMA `lightcurve-analysis` command or light-curve generation. Those workflows still require a compatible NMMA installation; the standalone package preserves and exposes the model code and its assets.

## Original dependencies

- Python
- PyTorch
- nflows
- tensorboard (used by the training helpers)
- Bilby (optional, only for `cast_as_bilby_result`)
- NMMA (optional, only for the original command-line integration)
