FCX: Finding Feasible Counterfactual Explanation
========================
Feasible Counterfactual Explanations (FCX) is a novel framework that generates realistic and low-cost counterfactuals by enforcing both hard feasibility constraints provided by domain experts and soft causal constraints inferred from data. Built on a modified Variational Autoencoder and optimized with a multi-factor loss function, FCX produces sparse, diverse, and actionable counterfactuals while preserving causal relationships, offering both individual-level explanations and global model feasibility assessments across multiple datasets
_[#icdew2024]_.

.. rubric:: References

.. [#icdew2024] K. Markou, D. Tomaras, V. Kalogeraki and D. Gunopulos,  
   "A Framework for Feasible Counterfactual Exploration incorporating Causality, Sparsity and Density,"  
   2024 IEEE 40th International Conference on Data Engineering Workshops (ICDEW), Utrecht, Netherlands, 2024,  
   pp. 254–261, doi: 10.1109/ICDEW61823.2024.00038.  
   
.. toctree::
   :maxdepth: 1
   :caption: Main modules

   fcx_unary_generation_adult
   fcx_binary_generation_adult
   blackbox_model_train
   fcx_evaluate_unary_adult
   fcx_evaluate_binary_adult
   
