import os
import numpy as np
# To load the array back
filepath = os.path.dirname(__file__)
arr = np.load(filepath + "/preprocessed_data.npy")
print(arr[1])