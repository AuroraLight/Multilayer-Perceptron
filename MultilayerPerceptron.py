import numpy as np
import Neuron as neu
import math

class MultilayerPerceptron() :

    def split_data(self, data, col):
        x, d = data, data
        x = np.delete(x, col-1, 1)
        d = np.delete(d, np.s_[0:col-1], 1)
        return x, d

    def normalization_desired_output(self, d):
        max_d = d.max()
        min_d = d.min()
        normal_d = d
        for i,temp_d in enumerate(d):
            normal_d[i] = (temp_d-min_d) / (max_d-min_d)
        return normal_d

    def data_process(self, file_name):
        data = np.loadtxt(file_name)
        row, self.col = data.shape
        max_array = data.max(axis=1)
        self.max_value = max_array.max()
        min_array = data.min(axis=1)
        self.min_value = min_array.min()
 
        if row > 20:
            number_train = int(row * 2 / 3) 
            data_random = data
            np.random.shuffle(data_random) 
            self.train_data, self.test_data = data_random[:number_train], data_random[number_train:]
        else:
            self.train_data = data
            self.test_data = data
        self.train_x, self.train_d = self.split_data(self.train_data, self.col)
        self.train_x = np.insert(self.train_x, 0, -1.0, 1)
        self.train_d = self.normalization_desired_output(self.train_d)
        self.test_x, self.test_d = self.split_data(self.test_data, self.col)
        self.test_x = np.insert(self.test_x, 0, -1.0, 1)  
        self.test_d = self.normalization_desired_output(self.test_d)
        self.clusters = np.unique(self.train_d)
        #self.num_cluster = len(self.clusters)

    def calcu_rate(self, x, d):
        correct = 0 #正確次數
        number = len(x)
        for i in range(number):
            y1 = self.firstNue.accumulator(x[i])
            y2 = self.secondNue.accumulator(x[i])
            y = [-1.0, y1, y2]
            o = self.outputNue.accumulator(y)
            if (o<0.5 and d[i]==self.clusters[0]) or (o>0.5 and d[i]==self.clusters[1]):
                correct += 1
        new_reg_rate = correct / number
        return new_reg_rate

    def calcu_rmse(self):
        x = np.append(self.train_x, self.test_x, 0)
        d = np.append(self.train_d, self.test_d, 0)
        sum = 0.0
        for i in range(len(x)):
            y1 = self.firstNue.accumulator(x[i])
            y2 = self.secondNue.accumulator(x[i])
            y = [-1.0, y1, y2]
            o = self.outputNue.accumulator(y)
            if(o>0.5):
                o+=0.5
            else:
                o-=0.5     
            sum += math.pow(o-d[i],2)
        return math.sqrt(sum/len(x))

    def training_perceptron(self, learning_rate, n):
        self.traning_reg_rate = 0.0 #訓練辨識率
        #初始化神經元
        self.firstNue = neu.Neuron(self.col)
        self.secondNue = neu.Neuron(self.col)
        self.outputNue = neu.Neuron(self.col)
        '''self.firstNue.w = np.array([-1.2, 1, 1])
        self.secondNue.w = np.array([0.3, 1, 1])
        self.outputNue.w = np.array([0.5, 0.4, 0.8])'''
        self.optimal_w = [self.firstNue.w, self.secondNue.w, self.outputNue.w]
        for j in range(n): #執行迭代次數
            self.train_x = np.append(self.train_x, self.train_d, 1) #合併X[]與D[]
            np.random.shuffle(self.train_x)  #註:要隨機訓練資料必須連同期望輸出array
            self.train_x, self.train_d = self.split_data(self.train_x, self.col+1) #因為此時閥值已經加入X[0]所以col+1
            for row_index in range(len(self.train_x)): #圖樣模式(pattern learning)
                #Pre-delivery()
                #得到所有神經元的輸出
                y1 = self.firstNue.accumulator(self.train_x[row_index])
                y2 = self.secondNue.accumulator(self.train_x[row_index])
                y = [-1.0, y1, y2]
                #y = np.vstack([firstNue.accumulator(self.train_x), secondNue.accumulator(self.train_x)])
                o = self.outputNue.accumulator(y)
                #Back-propagation()
                #調整鍵結值向量
                if (o>0.5 and self.train_d[row_index]==self.clusters[0]) or (o<0.5 and self.train_d[row_index]==self.clusters[1]):
                    output_regional_gradient = (self.train_d[row_index]-o)*o*(1-o)
                    self.outputNue.w += learning_rate * output_regional_gradient * y
                    self.firstNue.w += learning_rate * y1*(1-y1)*output_regional_gradient*self.outputNue.w[1]
                    self.secondNue.w += learning_rate * y2*(1-y2)*output_regional_gradient*self.outputNue.w[2]
            new_traning_reg_rate = self.calcu_rate(self.train_x, self.train_d) #取得辨識率
            #紀錄訓練辨識率高的鍵結值
            if new_traning_reg_rate > self.traning_reg_rate:
                self.optimal_w = [self.firstNue.w, self.secondNue.w, self.outputNue.w]
                self.traning_reg_rate = new_traning_reg_rate
            #完全分群跳出迴圈
            if self.traning_reg_rate == 1.0:
                break
        self.firstNue.w = self.optimal_w[0]
        self.secondNue.w = self.optimal_w[1]
        self.outputNue.w = self.optimal_w[2]

    def run(self, learning_rate, n):
        self.training_perceptron(learning_rate, n) #傳入學習率以及迭代次數以訓練感知機，取得最佳鍵結值、訓練辨識率
        self.RMSE = self.calcu_rmse()
        self.testing_reg_rate = self.calcu_rate(self.test_x, self.test_d) #測試辨識率

    def __init__(self):
        pass
