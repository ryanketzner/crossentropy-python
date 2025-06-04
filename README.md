# crossentropy-python

A lightweight Python package implementing the cross-entropy method and importance sampling using Gaussian and Gaussian Mixture proposal families.

---

## Dependencies

The library relies on:

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib` (for the examples)

Install them via `conda` or `pip`.

---

## Installation

### Clone the repository

```bash
git clone https://github.com/ryanketzner/crossentropy-python.git
cd crossentropy-python
```

### Install with Conda using YAML file

An environment.yaml file is provided to quickly create a conda environment in which to run this software. Otherwise,
the software can be run in any environment which has the four dependencies above installed.


```bash
conda env create -f environment.yaml
conda activate gauss_env
```

### Install the package locally (editable mode)

```bash
pip install -e .
```

This will make the `crossentropy` package importable anywhere inside the environment.

---

## Package Structure

### `crossentropy/`

| Module                    | Description |
|---------------------------|-------------|
| `gaussian.py`             | Implements `Gaussian` class: sampling, pdf, logpdf, weighted MLE |
| `gaussian_mixture.py`     | Implements `GaussianMixture` class with EM and weighted EM fitting |
| `cross_entropy.py`        | Core Cross-Entropy algorithm |
| `importance_sampling.py`  | Computes importance sampling estimates, variance, relative error, and confidence intervals |

---

## Examples

All examples are located in the `examples/` directory and can be run independently.

### 1. `example_gaussian.py`

**Single Gaussian proposal** optimized via Cross-Entropy for a rare event defined by distance to a single target center.

Run:
```bash
python examples/example_gaussian.py
```

---

### 2. `example_gmm.py`

**Gaussian Mixture Model (GMM)** same as previous example, but using a Gaussian mixture proposal family. This demonstrates
that a GMM can still be used for unimodal event regions.

Run:
```bash
python examples/example_gmm.py
```

---

### 3. `example_multimodal_gmm.py`

Multi-target example using a GMM proposal family for a rare event defined by distance to multiple target centers.

Run:
```bash
python examples/example_multimodal_gmm.py
```

## Contact

ketzner@ucf.edu