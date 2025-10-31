# Exploring the Limits of Linearity on MNIST

> From pure linear classifiers to ReLU MLPs — quantify what each component (Softmax, ReLU, Dropout, BatchNorm) buys you.

## Project Goals
1. Start with **pure linear** model `y = XW + b` on MNIST.
2. Switch the loss to **CrossEntropy** (Softmax) and compare.
3. Add **ReLU** (single hidden layer) → observe nonlinearity benefits.
4. Add **Dropout** and **BatchNorm** → observe generalization & stability.
5. Produce **tables/plots** and a short **report**.

## Quick Start
```bash
# (Optional) create venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Stage 1: Pure Linear (MSE loss)
python -m src.train --model linear --loss mse --epochs 5

# Stage 2: Linear + CrossEntropy (Softmax)
python -m src.train --model linear --loss crossentropy --epochs 5

# Stage 3: MLP + ReLU
python -m src.train --model mlp --hidden 256 --epochs 5

# Stage 4: MLP + ReLU + Dropout
python -m src.train --model mlp --hidden 256 --dropout 0.5 --epochs 5

# Stage 4b: + BatchNorm
python -m src.train --model mlp --hidden 256 --batchnorm --epochs 5
```

## Expected Ballpark (test accuracy, for reference only)
| Model | Loss | Test Acc (±) |
|------|------|---------------|
| Linear | MSE | ~80–85% |
| Linear | CrossEntropy | ~88–92% |
| MLP(784→256→10) + ReLU | CrossEntropy | ~96–97% |
| + Dropout / + BatchNorm | CrossEntropy | ~96.5–97.7% |

> Numbers depend on epochs, learning rate, and randomness. Treat them as reference.

## Repo Structure
```
.
├── README.md
├── requirements.txt
├── src
│   ├── data.py
│   ├── models.py
│   ├── utils.py
│   └── train.py
├── notebooks
│   └── 01_exploring_linearity.ipynb
└── scripts
    ├── run_linear_mse.sh
    ├── run_linear_ce.sh
    └── run_mlp_relu.sh
```

## Reporting
Use the notebook to:
- Plot training/validation **loss & accuracy** curves.
- Visualize **weight templates** of the linear classifier.
- Show **misclassified** examples and confusion matrices.
- Generate a short **HTML/PDF** report.

## License
MIT
