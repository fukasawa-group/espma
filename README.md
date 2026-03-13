# Pr-* (working title)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

Official implementation of a machine learning model for **biological vs. non-biological interface classification**.

For quick access to the available Colab notebooks, see [Quick Start](#quick-start) below.

> **Status:** This repository accompanies a manuscript currently under submission.
> The code and documentation may be updated as the paper and experiments are finalized.

---

## Overview

This repository contains the current public codebase for the **Pr-*** project (working title), a model for **classifying biological and non-biological interfaces**.

It is designed to:

- **integrate sequence and structure-derived signals for protein–protein interface assessment**
- **support accurate identification of native-like protein–protein interfaces**
- **provide both lightweight and structure-aware variants for flexible use across different application settings**

This repository provides:

- inference code
- training scripts
- pretrained checkpoints
- notebooks for testing and analysis

---

## News

- **2026-03-12**: Initial repository release
- **2026-03-14**: Documentation cleanup

---

## Quick Start

### Open in Colab

You can try the current **Pr-*** implementation directly in Google Colab without local installation.
Two notebook variants are provided:

#### Integrated variant

Sequence and structure-aware version.

[![Open Integrated Variant In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fukasawa-group/prsurflm/blob/main/notebooks/inference.ipynb)

This notebook includes:

- environment setup
- example input preparation
- pretrained model inference
- output visualization is planned for a future update

#### Lightweight variant

Lightweight version for simpler evaluation and faster testing.

[![Open Lightweight Variant In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fukasawa-group/prsurflm/blob/main/notebooks/inference_lr.ipynb)

This notebook includes:

- environment setup
- example input preparation
- pretrained model inference
- output visualization is planned for a future update

---

## Installation

Clone the repository:

```bash
git clone https://github.com/fukasawa-group/prsurflm.git
cd prsurflm