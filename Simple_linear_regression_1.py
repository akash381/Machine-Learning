import numpy as np
import pandas as pd
 
def estimate_coef(x, y):
    n = np.size(x)
    m_x, m_y = np.mean(x), np.mean(y)
    SS_xy = np.sum(y*x - n*m_y*m_x)
    SS_xx = np.sum(x*x - n*m_x*m_x)
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return(b_0, b_1)
 
def main():
    dataset = pd.read_csv('data1.csv')
    x = dataset.iloc[:, 7:8].values
    y = dataset.iloc[:, 13].values
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {}  \
          \nb_1 = {}".format(b[0], b[1]))
 
if __name__ == "__main__":
    main()
