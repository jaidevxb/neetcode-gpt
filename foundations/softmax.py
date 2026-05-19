import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z = z - np.max(z)
        exp_vals = np.exp(z)
        sum_val = np.sum(exp_vals)
        prob_vals = exp_vals / sum_val
        return np.round(prob_vals, 4)

 