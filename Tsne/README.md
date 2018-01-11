# T-sne
T-distributed stochastic neighbor embedding rewrite by ourselves, it's a good dimensionality reduction method.

# Supported python versions
- python Python 3.6

# Python package dependencies
- [numpy](https://github.com/numpy/numpy)

# Documentation
Setup model
```python
from Tsne import Tsne
tsne = Tsne(is_reduce_dim=0, reduce_dim=None, out_dim=2, perplexity=30.0, max_iters=1000)
```

Transform data
```python
tsne.transform(data)
```