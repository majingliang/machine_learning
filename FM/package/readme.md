example:
```
import sys
import pandas as pd
import numpy as np
sys.path.append('/Users/slade/Documents/GitHub/machine_learning/FM')
from package import fm
path = '/Users/slade/Documents/GitHub/machine_learning/data/data_all.txt'
X = pd.read_table(path)
X = np.array(X.iloc[:,2:])
#train
fm.fit(path,iter = 1)
#predict
fm.predict(X)
```