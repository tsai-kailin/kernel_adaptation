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
4. example code is test on single NVIDIA A40. 

#### Deep Kernel method
1. see `deep_example.py` for the demo of deep features full adaptation. The dataset is Demand dataset [[1]](#1).
2. specify the network structure in `/models/deep_kernel/nn_structure` and add the network structure name in the function `build_extractor()` in `/models/deep_kernel/nn_structure/__init__.py`
3. specify the training parameters and models using `.json` file. See `/tests/configs/demand_config.json` for example. 
4. see `deep_example_partial.py` for the demo of deep features partial adaptation. The dataset is Demand dataset [[1]](#1)
5. example code is test on single NVIDIA GeForce GTX TITAN X
### License
See the License file.

### Acknowledgement
Part of the code is copied or adapted from [[1]](#1) and [[2]](#2). We thank the authors for generously open-sourcing the code under MIT License.


### References
<a id="1">[1]</a> 
Xu, L., Kanagawa, H., & Gretton, A. (2021). Deep proxy causal learning and its application to confounded bandit policy evaluation. Advances in Neural Information Processing Systems, 34, 26264-26275. https://github.com/liyuan9988/DeepFeatureProxyVariable/tree/master


<a id="1">[2]</a> 

Mastouri, A., Zhu, Y., Gultchin, L., Korba, A., Silva, R., Kusner, M., ... & Muandet, K. (2021, July). Proximal causal learning with kernels: Two-stage estimation and moment restriction. In International conference on machine learning (pp. 7512-7523). PMLR. https://github.com/yuchen-zhu/kernel_proxies
