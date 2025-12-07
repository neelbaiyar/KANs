# KANs: Kolmogorov-Arnold Networks vs Standard Activations

This project benchmarks **Kolmogorov-Arnold Networks (KANs)** against
standard neural network activation functions (ReLU, GELU, etc.) on image
classification tasks. We evaluate performance across **MNIST** and
**CIFAR-10**, comparing:

-   Test accuracy
-   Training time
-   Validation loss
-   Sparsity

The goal is to test whether **learned functional activations (KANs)**
can achieve **competitive accuracy with improved sparsity and
convergence behavior**.

------------------------------------------------------------------------

## Project Structure

    KANs/
    ├── src/
    │   ├── activations.py        # Standard + custom activation functions
    │   ├── kan_model.py          # Kolmogorov–Arnold Network implementation
    │   ├── models.py             # Standard MLP models
    │   ├── data_loader.py        # MNIST + CIFAR-10 loaders
    │   └── train_loop.py         # Shared training + early stopping logic
    │
    ├── results/
    │   ├── full_results_*.json   # Full experiment logs
    │   ├── accuracy_table.csv
    │   ├── training_time_table.csv
    │   ├── sparsity_table.csv
    │   └── val_loss_table.csv
    │
    ├── KANtaloupe.ipynb          # Main experiment notebook (run-all pipeline)
    └── README.md

------------------------------------------------------------------------

## How to Run the Project

### 1. Clone the Repository

``` bash
git clone https://github.com/neelbaiyar/KANs.git
cd KANs
```

------------------------------------------------------------------------

### 2. Install Dependencies

``` bash
pip install torch torchvision matplotlib pandas tqdm
```

------------------------------------------------------------------------

### 3. Run the Main Notebook

Open and run:

    KANtaloupe.ipynb

This notebook automatically:

-   Loads MNIST and CIFAR-10
-   Benchmarks ReLU, GELU, and KAN
-   Applies early stopping
-   Saves:
    -   Accuracy table
    -   Training time table
    -   Sparsity table
    -   Validation loss table
-   Generates publication-ready plots

------------------------------------------------------------------------

## Experiment Configuration

Inside `KANtaloupe.ipynb`:

``` python
BASE_ACTIVATIONS = ["relu", "gelu"], etc.
USE_KAN = True

EPOCHS_BY_DATASET = {
    "mnist": 20,
    "cifar10": 50
}
```

You can:

-   Add new activation functions in `activations.py`
-   Toggle KAN on/off
-   Adjust epochs per dataset
-   Scale experiments without touching core training logic

------------------------------------------------------------------------

## Early Stopping

All models use **validation-loss based early stopping**:

-   Training halts if validation loss does not improve for **5
    consecutive epochs**
-   Prevents overfitting
-   Saves compute
-   Ensures fair convergence comparisons

------------------------------------------------------------------------

## Output Artifacts

After a full run, the following files are generated:

-   `full_results_*.json`: Full training + evaluation logs
-   `accuracy_table.csv`: Final test accuracy comparison
-   `training_time_table.csv`: Total training time per model
-   `sparsity_table.csv`: First-layer sparsity metric
-   `val_loss_table.csv`: Best & final validation loss

Plots include:

-   Validation Loss vs Epoch (MNIST & CIFAR-10)
-   Accuracy comparisons
-   Speed vs Accuracy tradeoffs

------------------------------------------------------------------------

## Key Findings

-   KAN achieves accuracy comparable to ReLU and GELU
-   KAN converges in fewer effective epochs on CIFAR-10
-   KAN and ReLU produce significantly higher sparsity than GELU
-   GELU slightly dominates in absolute accuracy

These results support the hypothesis that:

> Learned functional activations can match standard activations while
> offering interpretability and convergence advantages.

------------------------------------------------------------------------

## Datasets Used

-   **MNIST**: 28×28 grayscale handwritten digit images
-   **CIFAR-10**: 32×32 RGB natural images

Both datasets are downloaded automatically via `torchvision`.

------------------------------------------------------------------------

## Authors

-   **Neel Aiyar**: Experiment pipeline, training system,
    benchmarking, visualization
-   **Vadim Pelyushenko**: Custom activation functions, theoretical analysis,
    write-up

------------------------------------------------------------------------

## License

This project is for academic and research use ONLY.
