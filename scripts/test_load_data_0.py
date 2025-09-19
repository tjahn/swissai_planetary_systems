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

plt.scatter(
np.log10(data["a"]),
np.log10(data["total_mass"]),
1
)

# %%
# 
