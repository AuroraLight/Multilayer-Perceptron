import numpy as np

class Neuron():
    
    def __init__(self, col):
        self.w = np.random.random_sample(col,) #隨機一組鍵結值
        #print(self.w)

    def sigmoidal(self,v):
        y = 1/(1+np.exp(-1*v))
        return y
    
    def accumulator(self, x):
        #print("self.w:",self.w, "  x:",x)
        v = (x * self.w).sum() #累加取得v
        #print("v:",v)
        y = self.sigmoidal(v)
        #print(y)
        return y