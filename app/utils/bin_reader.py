import pandas as pd
import numpy as np

def read_bin(path: str) -> pd.DataFrame:
    values = np.fromfile(path, dtype=np.float64)
    df = pd.DataFrame({
        "index": np.arange(len(values)),
        "value": values,
    })
    return df

if __name__ == '__main__':
    bin = "app/app/results/diag.bin"
    correct_bin = "app/app/results/correct_diag.bin"
    df = read_bin(bin).sort_values("value")
    correct_df = read_bin(correct_bin).sort_values("value")
    print("DF:")
    print(df.head(10))
    print("CORRECT")
    print(correct_df.head(10))

