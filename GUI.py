import sys
import tkinter as tk
from tkinter import filedialog
import MultilayerPerceptron as mper
import numpy as np
import matplotlib
#在tkinter中畫出matplot所需class及function
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
matplotlib.use('TkAgg')

class GUI():
    """Graphical User Interface of Multilayer Perceptron"""
    def show_data_on_gui(self):
        #檢查訓練資料text是否有字，若有則清空
        if self.train_data_text.get('1.0', 'end') != '':
            self.train_data_text.delete('1.0', 'end')
        self.train_data_text.insert('end', self.perceptron_obj.train_data)
        #檢查測試資料text是否有字，若有則清空
        if self.test_data_text.get('1.0', 'end') != '':
            self.test_data_text.delete('1.0', 'end')
        self.test_data_text.insert('end', self.perceptron_obj.test_data)

    def nonlinear_transform(self, train_x):
        temp_x = train_x.copy()
        for i in range(len(train_x)):
            temp_x[i][1] = self.perceptron_obj.firstNue.accumulator(train_x[i])
            temp_x[i][2] = self.perceptron_obj.secondNue.accumulator(train_x[i])
        return temp_x
        
    def draw_point(self, subplot, x, d, clusters):
        for i in range(len(x)):
            #第一群用紅色叉
            if d[i] == clusters[0]:
                subplot.plot(x[i,1], x[i,2], 'rx')
            #第二群用藍色叉
            if d[i] == clusters[1]:
                subplot.plot(x[i,1], x[i,2], 'bx')

    def show_result(self):
        w = self.perceptron_obj.optimal_w #訓練後的最佳鍵結值
        self.perceptron_obj.firstNue.w = w[0]
        self.perceptron_obj.secondNue.w = w[1]
        self.perceptron_obj.outputNue.w = w[2]
        train_x = self.perceptron_obj.train_x #訓練資料input
        train_d = self.perceptron_obj.train_d #訓練資料期望輸出
        test_x = self.perceptron_obj.test_x #測試資料input
        test_d = self.perceptron_obj.test_d #測試資料期望輸出
        clusters = self.perceptron_obj.clusters #群編號array
        min_x = self.perceptron_obj.min_value #x[1]最小值
        max_x = self.perceptron_obj.max_value #x[1]最大值
        #串聯結果的字串
        str_traning_reg_rate = '訓練辨識率: ' + str(self.perceptron_obj.traning_reg_rate)
        str_testing_reg_rate = '測試辨識率: ' + str(self.perceptron_obj.testing_reg_rate)
        str_RMSE = 'RMSE: ' + str(self.perceptron_obj.RMSE)
        str_optimal_w = '鍵結值: ' + str(w[0].round(5)) + '    ' + str(w[1].round(5)) + '    ' + str(w[2].round(5)) #鍵結值取到小數點第五位
        #結果顯示到GUI
        self.train_reg_rate_label.config(text=str_traning_reg_rate)
        self.test_reg_rate_label.config(text=str_testing_reg_rate)
        self.RMSE_label.config(text=str_RMSE)
        self.optimal_w_label.config(text=str_optimal_w)
        #清空訓練結果圖以及測試結果圖
        self.train_plot.clear()
        self.test_plot.clear()
        #訓練結果方程式
        axis_x1 = np.arange(0, 2)
        axis_x2 = (w[2][0] - w[2][1]*axis_x1) / w[2][2] #w[1]*x1 + w[2]*x2 - w[0] = 0
        #畫上訓練結果方程式
        self.train_plot.plot(axis_x1,axis_x2,'k-')
        self.test_plot.plot(axis_x1,axis_x2,'k-')
        #畫上各自的資料點
        temp_train_x = self.nonlinear_transform(train_x)
        self.draw_point(self.train_plot, temp_train_x, train_d, clusters)
        temp_test_x = self.nonlinear_transform(test_x)
        self.draw_point(self.test_plot, temp_test_x, test_d, clusters)
        #圖顯示在GUI
        self.train_canvas.draw()
        self.test_canvas.draw()

    def open_file(self):
        filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("text","*.txt"),("all files","*.*")))
        self.perceptron_obj = mper.MultilayerPerceptron()
        self.perceptron_obj.data_process(filename)
        self.show_data_on_gui()

    def run(self):
        learning_rate = float(self.learning_rate_entry.get())
        n = int(self.n_entry.get()) #收斂條件
        self.perceptron_obj.run(learning_rate, n)
        self.show_result()

    def exit_application(self):
        sys.exit()

    def init_label(self):
        self.train_result_label = tk.Label(self.master, text = '訓練結果:')
        self.test_result_label = tk.Label(self.master, text = '測試結果:')
        self.learning_rate_label = tk.Label(self.master, text = '學習率:')
        self.n_label = tk.Label(self.master, text = '收斂條件:')
        self.train_data_label = tk.Label(self.master, text='訓練資料:')
        self.test_data_label = tk.Label(self.master, text='測試資料:')
        self.train_reg_rate_label = tk.Label(self.master, text='訓練辨識率:')
        self.test_reg_rate_label = tk.Label(self.master, text='測試辨識率:')
        self.RMSE_label = tk.Label(self.master, text='RMSE:')
        self.optimal_w_label = tk.Label(self.master, text='鍵結值:')

    def init_entry(self):
        var_learning_rate = tk.StringVar()
        var_n = tk.StringVar()
        var_learning_rate.set('0.8')
        var_n.set('1000')
        self.learning_rate_entry = tk.Entry(self.master, textvariable=var_learning_rate)
        self.n_entry = tk.Entry(self.master, textvariable=var_n)

    def init_button(self):
        self.open_file_button = tk.Button(self.master, text='開啟檔案', command=self.open_file, width = 10)
        self.run_button = tk.Button(self.master, text='Run', command=self.run, width = 10)
        self.exit_button = tk.Button(self.master, text='Exit', command=self.exit_application, width = 10)

    def init_text(self):
        text_area_height = 15
        text_area_width = 42
        self.train_data_text = tk.Text(self.master, height=text_area_height, width=text_area_width)
        self.test_data_text = tk.Text(self.master, height=text_area_height, width=text_area_width)

    def on_key_train_event(self, event):
        key_press_handler(event, self.train_canvas, self.train_plot_toolbar)

    def on_key_test_event(self, event):    
        key_press_handler(event, self.test_canvas, self.test_plot_toolbar)
    
    def init_matplot_in_tkinter(self):
        #init fig
        #訓練結果的Figure並加入plot
        train_plot_fig = Figure(figsize=(5,6))
        self.train_plot = train_plot_fig.add_subplot(111)
        #測試結果的Figure並加入plot
        test_plot_fig = Figure(figsize=(5,6))
        self.test_plot = test_plot_fig.add_subplot(111)

        #init canvas
        self.train_canvas = FigureCanvasTkAgg(train_plot_fig, master=self.master)
        self.test_canvas = FigureCanvasTkAgg(test_plot_fig, master=self.master)        
        
        #init frame 把toolbar放入frame中，目的:調整toolbar位置
        self.train_toolbar_frame = tk.Frame(self.master)
        self.test_toolbar_frame = tk.Frame(self.master)
        
        #init toolbar 初始化matplotlib提供的toolbar，並指定key_press_event處理函數
        #訓練結果的toolbar
        self.train_plot_toolbar = NavigationToolbar2TkAgg(self.train_canvas, self.train_toolbar_frame)        
        self.train_plot_toolbar.update()
        self.train_canvas.mpl_connect('key_press_event', self.on_key_train_event)
        #測試結果的toolbar
        self.test_plot_toolbar = NavigationToolbar2TkAgg(self.test_canvas, self.test_toolbar_frame)
        self.test_plot_toolbar.update()
        self.test_canvas.mpl_connect('key_press_event', self.on_key_test_event)
        
    def place_component(self):
        #layout 13*3(row*column)
        spacing = 20 #間距
        first_row = 20
        first_column = 20
        second_column = 535
        third_column = 1050
        thirteen_row = 700
        # first row
        self.train_result_label.place(x=first_column, y=first_row)
        self.test_result_label.place(x=second_column, y=first_row)
        self.open_file_button.place(x=third_column, y=first_row)
        # second row
        self.train_canvas.get_tk_widget().place(x=first_column, y=first_row + spacing*1.5)
        self.test_canvas.get_tk_widget().place(x=second_column, y=first_row + spacing*1.5)
        self.learning_rate_label.place(x=third_column, y=first_row + spacing*1.5)
        # third to 9th row
        self.learning_rate_entry.place(x=third_column, y=first_row + spacing*2.5)
        self.n_label.place(x=third_column, y=first_row + spacing*3.5)
        self.n_entry.place(x=third_column, y=first_row + spacing*4.5)
        self.train_data_label.place(x=third_column, y=first_row + spacing*6)
        self.train_data_text.place(x=third_column, y=first_row + spacing*7)
        self.test_data_label.place(x=third_column, y=first_row + spacing*18)
        self.test_data_text.place(x=third_column, y=first_row + spacing*19)
        # 10th row     
        self.train_reg_rate_label.place(x=third_column, y=first_row + spacing*30)
        # 11th row
        self.test_reg_rate_label.place(x=third_column, y=first_row + spacing*31)
        # 12th row
        self.train_toolbar_frame.place(x=first_column, y=first_row + spacing*32)
        self.test_toolbar_frame.place(x=second_column, y=first_row + spacing*32)
        self.RMSE_label.place(x=third_column, y=first_row + spacing*32)
        # 13th row
        self.optimal_w_label.place(x=second_column, y=thirteen_row)
        self.run_button.place(x=third_column + 50, y=thirteen_row)
        self.exit_button.place(x=third_column + 180, y=thirteen_row)

    def __init__(self, master):
        self.master = master
        master.title('Multilayer Perceptron Application')
        master.geometry('1366x768')
        #initial layout component
        self.init_label()
        self.init_entry()
        self.init_button()
        self.init_text()
        self.init_matplot_in_tkinter()
        self.place_component()