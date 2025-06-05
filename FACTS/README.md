# Fairness Aware Counterfactuals for Subgroups

This repository is the implementation of the paper Fairness Aware Counterfactuals for Subgroups (FACTS). FACTS is a framework for auditing subgroup fairness through counterfactual explanations. We aim to (a) formulate different aspects of the difficulty of individuals in certain subgroups to achieve recourse, i.e. receive the desired outcome, either at the micro level, considering members of the subgroup individually, or at the macro level, considering the subgroup as a whole, and (b) introduce notions of subgroup fairness that are robust, if not totally oblivious, to the cost of achieving recourse. Below, appears a subgroup audited by one of our fairness metrics.  

In our work, we call the above representation "Comparative Subgroup Counterfactuals". These if-then rules, along with the effectiveness that each action manages to achieve (micro or micro, see Section 2.2 of our paper) give, in our opinion, a very clear and intuitive insight into certain type of bias that a machine learning model may exhibit.

## Requirements

All experiments were run on the [Anaconda](https://www.anaconda.com/) platform using python version 3.9.16. To avoid bugs due to incompatible package versions, we have exported the [requirements.txt](requirements.txt) for the conda environment on which we worked.

To create a conda environment with the same configuration, run:

    ```bash
    conda create --name facts python==3.9.16
    ```

and then activate it with

```setup
conda activate facts
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

## Example notebooks

s a gateway for the user, there exist an example notebook in the `examples/` directory:

- `demo_FACTS.ipynb`


Upon successful completion of the setup, the notebooks should execute as intended. They demonstrate basic usage of our framework, allowing users to adjust parameter values as needed.





