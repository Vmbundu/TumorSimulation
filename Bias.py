import numpy as np
import variables
class Bias:

    def __init__(self, N, slen) -> None:
        self.N = N
        self.slen = slen

    def index_bias(self):
        index_bias = []
        kn=1
        index_bias0=[]
        seq = range(-kn,kn+1)
        for i in range(-kn,kn+1):
            index_bias0 = [*index_bias0, *(np.array(seq) + self.N*i )]
        for j in range(-kn,kn+1):
            for ele in index_bias0:
                index_bias.append(ele+self.slen*j)
        return np.array(index_bias)

    def pres_bias(self):
        knn = 20     
        pres_bias0 = []
        pres_bias = []
        seq = range(-knn,knn+1)
        for i in range(-knn,knn+1):
            pres_bias0 = [*pres_bias0, *(np.array(seq) + self.N*i)]
        for j in range(-knn,knn+1):
            for ele in pres_bias0:
                pres_bias.append(ele+self.slen*j)
        return np.array(pres_bias)

