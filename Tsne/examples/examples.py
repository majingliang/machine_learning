from Tsne import Tsne
import numpy as np

data = np.random.random((1000,100))
tsne = Tsne()
new_data = tsne.transform(data)

