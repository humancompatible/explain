# GLANCE: Global Actions in a Nutshell for Counterfactual Explainability

This is the home directory of our Global Actions in a Nutshell for Counterfactual Explainability (GLANCE) framework. GLANCE is a versatile and adaptive framework for generating global counterfactual explanations.
Before trying to run our project, please consult the setup instructions below.

## Table of Contents
- [Setup Instructions](#setup-instructions)
- [Example Notebooks](#example-notebooks)
- [Note](#note)
  
## Setup Instructions

### Installation Steps

1. **Create a Virtual Environment:**
   
    Preferable, create a virtual environment with python==3.10.4 and activate it. 

    Using Conda:
    ```bash
    conda create --name glance python==3.10.4
    conda activate glance
    ```

    Using python venv:
    ### Prerequisites
    
    Make sure you have the following prerequisites installed:
    - **Python** (version 3.10.4)
    - **pip**
    ```bash
    python -m venv glance
    glance/bin/activate  
    ```


3. **Install Dependencies:**
   
    Use the file `requirements.txt` to install all dependencies of our framework:

    ```bash
    pip install -r requirements.txt
    ```

4. **Set Up Jupyter Notebook (Optional for Example Notebooks):**
   
    Additionally, if you wish to run the example notebooks (described below) you should run the following commands to appropriately setup and start the `jupyter notebook` application:

    ```bash
    python -m ipykernel install --user --name=glance --display-name "glance env"
    jupyter notebook

    ```

## Example Notebooks

As a gateway for the user, there exist an example notebook in the `examples/` directory:

- `Adult.ipynb`


Upon successful completion of the setup, the notebooks should execute as intended. They demonstrate basic usage of our framework, allowing users to adjust parameter values as needed.

