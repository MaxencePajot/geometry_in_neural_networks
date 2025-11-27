This is a repository for the data and code of:

Can Neural networks model the human perception of geometric shapes?

# Environment 

The environment is a python 3.10 environment. The list of dependencies is listed in the pyproject.toml file. To reproduce the environment used, use the command line:

`uv sync`

or 

`pip install requirements.txt`

You also need R for the plots. The R version used is 4.5.0

# Use 

Neural networks are defined in the get_model.py script and used in the different experiments.
The three folders 1,2 and 3 correspond to the different experiments described in the paper, studying a different aspect of geometric shape representations in neural networks. In each case, to generate a Representational Dissimilarity Matrix (RDM), run make_RDMs.py 'model_name'. 
