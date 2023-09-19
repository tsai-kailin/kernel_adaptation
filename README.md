# kernel_adaptation

### installation
1. install the conda environment using `environment.yml`
2. It is highly recommended to install jax-gpu, this would speed up the computation quite a bit.
3. Build the package
```
python setup.py install
```
### usage
see the example files under `/tests/`

#### Kernel method
1. all the data needed to be converted to jax.numpy.array()
2. `example.py` shows the adaptation method of a regression task under simple latent shifts. The prediction error is the averge $\ell_2$ of the regression outcomes and the true $y$. The data generation code is in `/src/gen_data.py`
3. see `train.py` for demo of the toy example on estimation error v.s. sample size. There are two tasks. 
4. example code is test on NVIDIA A40. 

#### Deep Kernel method
1. see `deep_example.py` for the demon of deep features full adaptation. The dataset is Demand dataset
2. specify the network structure in `/models/deep_kernel/nn_structure` and add the network structure name in the function `build_extractor()` in `/models/deep_kernel/nn_structure/__init__.py`
3. specify the training parameters and models using `.json` file. See `/tests/configs/demand_config.json`
4. see `deep_example_partial.py` for the demon of deep features partial adaptation. The dataset is Demand dataset
5. example code is test on NVIDIA GeForce GTX TITAN X
### Acknowledgement

