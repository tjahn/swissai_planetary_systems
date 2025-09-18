# %% 


import numpy as np
import pandas as pd
import pathlib

import matplotlib.pyplot as plt 




data = pd.read_csv(pathlib.Path(__file__).parent.parent / "data" / "Easier Dataset.csv")
# %%


print(data.columns)
print("num systems", data["system_number"].nunique())
# %%

data.hist("total_mass", bins=np.linspace(0,1,100))

# %%

