# kernel_adaptation

### installation
1. install the conda environment using `environment.yml`
2. It is highly recommended to use jax-gpu, this would speed up the computation quite a bit.

### usage
see the example files under `/tests/`
1. all the data needed to be converted to jax.numpy.array()
2. `example.py` shows the adaptation method of a regression task under simple latent shifts. The prediction error is the averge $\ell_2$ of the regression outcomes and the true $y$. The data generation code is in `/src/gen_data.py`

