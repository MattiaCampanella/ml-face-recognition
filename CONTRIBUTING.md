# Repository Structure and Best Practices

This document serves as a guide for keeping your project organized. A well-structured repository not only facilitates teamwork but also acts as the calling card for your project in academic and professional contexts.

---

## 📁 1. Directory Organization

In the fork you have cloned, you will find a pre-configured structure. Here is how to use it correctly:

* **`README.md`**: File for the technical documentation of the repository. Replace the placeholders in the `README.md` file with the technical information about the repository.
* **`data/`**: Store your datasets here. Remember that **these files should NEVER be uploaded to GitHub**. The `.gitignore` file is already configured to exclude files contained in this folder, but always pay attention.
* **`src/`**: The heart of your code. In addition to the folders already included, you can add any others you deem necessary for your project.
  * `datasets/`: Scripts for downloading, parsing, and PyTorch dataloaders.
  * `models/`: Neural network architectures (e.g., definitions of `nn.Module` classes).
  * `training/`: Training loops, custom loss functions, optimizers.
  * `evaluation/`: Code for testing, inference, and metric calculation.
  * `utils/`: Auxiliary functions, loggers, visualizations.
* **`notebooks/`**: Jupyter Notebooks. To be used *only* for initial data exploration, quick visualizations, and proof of concepts. The actual training code must reside in the Python files inside `src/`.
* **`experiments/`**:
  * `configs/`: `.yaml` or `.json` files to manage experiment parameters (hyperparameters, paths).
  * `logs/`: TensorBoard outputs, Weights & Biases, or simple log files. *(Do not commit)*.
* **`figures/`**: Graphs, test plots, and images used in the README or presentation.
* **`docs/`**: Files for detailed documentation, final presentation slides, and the final report (`docs/REPORT.md`).

---

## 💻 2. Code Standards (Clean Code)

Having readable code is essential, especially in Machine/Deep Learning projects where messy implementations can easily mask logical bugs.

* **Meaningful Names**: Use clear and descriptive names (e.g., `compute_cross_entropy` instead of `calc_ce`).
* **Modularity**: Avoid "monster" files with 2000 lines. Each file should have a single responsibility. For example, the training loop should not also contain the neural network architecture or tensor transformations.
* **Docstrings and Comments**: 
  * Document the main classes and functions with short but comprehensive Docstrings.
  * Limit "in-line" comments mostly to counter-intuitive / complex blocks (e.g., complicated operations between dimensional tensors that modify the shape).
* **Type Hinting**: The inclusion of classic types like `int`, `str`, `List`, etc., is highly recommended, as it makes the code's intent immediately clear to reviewers.

---

## 📦 3. Dependency Management

Reproducibility is a fundamental cornerstone of Deep Learning system design. Another person must be able to recreate your environment in an instant.
* Keep the `environment.yml` file **always updated**.
* If you download a new crucial package via pip or conda, make sure to note it in the file (with version numbers when appropriate).
* Avoid including OS-specific libraries or those not strictly required for the project in the environment file.

---

## 🌳 4. Workflow and Git Usage

For groups consisting of more than one person, we strongly advise against committing your progress solely to the `main` branch.
* **Branch Usage**: Work by features. Want to test a ResNet50? Create a `feature/resnet50` branch. When the implementation is solid, open a "Pull Request" on GitHub and merge it into `main`.
* **Commit Messages**: Avoid commits like "update" or "fix bug". Try to provide information: `Add custom triplet loss in training loop` or `Fix dataloader out of bounds exception`.

---

By following these guidelines, you will develop habits aligned with the standards commonly used in the software industry and research!
