# AutoFair Explainability Frameworks

This repository contains two explainability methods developed for the **AutoFair** project:

- **FACTS**: *Fairness-Aware Counterfactuals for Subgroups* — a model-agnostic, highly parameterizable framework for auditing subgroup fairness through counterfactual explanations.
- **GLANCE**: *Global Actions in a Nutshell for Counterfactual Explainability* — a versatile and adaptive framework for generating global counterfactual explanations.
- **LiCE**: *Likely Counterfactual Explanations* — a method of finding high-quality plausible counterfactual explanations.
- **FCX**: *Feasible Counterfactual Explanations* - a novel framework that generates realistic and low-cost counterfactuals by enforcing both hard feasibility constraints provided by domain experts and soft causal constraints inferred from data.
  
---

## Documentation

Full API documentation is available at: **[https://humancompatible-explain.readthedocs.io/en/latest/index.html](https://humancompatible-explain.readthedocs.io/en/latest/index.html)**

---

## Project Structure

The `humancompatible/explain/` folder contains the corresponding code for the implemented methods.


---

## Setup Instructions

We recommend using [Anaconda](https://www.anaconda.com/) or Python virtual environments to avoid package conflicts.

### 1. Clone the repository

```bash
git clone https://github.com/humancompatible/explain.git
cd explain
```

### 2. Create and activate a virtual enviroment
**Using Conda**:<br><br>
```bash
conda create --name explain python=3.10.4
conda activate explain
```
<br>

**Using by using Python venv**:<br>
```bash
python3 -m venv env
source env/bin/activate
```
### 3. Install required dependencies

```bash
pip install -e .
```

### 4. (Optional) Jupyter setup for notebooks
```bash
python -m ipykernel install --user --name=autofair --display-name "AutoFair Env"
jupyter notebook
```

## Example notebooks
Explore the functionality through example notebooks in the examples/ directory:

- [demo_FACTS.ipynb](examples/facts/demo_FACTS.ipynb) – Demonstrates FACTS usage and subgroup fairness evaluation with the UCI Adult dataset.
- [demo_GLANCE.ipynb](examples/glance/demo_GLANCE.ipynb) – Demonstrates GLANCE with the UCI Adult dataset.
- [demo_LiCE.ipynb](examples/lice/demo_LiCE.ipynb) – Demonstrates LiCE with the [Give me some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data) dataset.

These notebooks offer adjustable parameters and serve as entry points for integrating your own models or datasets.


