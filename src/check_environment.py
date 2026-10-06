import sys

import numpy as np
import pandas as pd
import sklearn
import statsmodels


def main():
    print("Adaptive Diabetes MLOps")
    print("-----------------------")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Pandas: {pd.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"Statsmodels: {statsmodels.__version__}")
    print("Environment status: OK")


if __name__ == "__main__":
    main()
