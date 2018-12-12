# Multilayer-Perceptron

### 一、程式執行說明
>* 執行程式:
>點選main.exe檔開啟程式
>* Step 1: 
>點選開啟檔案，選擇要開啟的資料集。開啟檔案後，若該檔案資料大於20筆，則訓練資料會顯示出2/3筆資料、測試資料會顯示出1/3筆資料，反之，訓練資料跟測試資料會相同。
>* Step 2: 
>設定學習率、收斂條件
>* Step 3:
>按下Run，訓練結果、測試結果分別顯示出資料點及鍵結值方程式，訓練辨識率、測試辨識率、鍵結值顯示出結果。
>* Step 4:
>按下Exit或右上角X可結束程式。
>備註：
>1. 讀入檔案後，可重複案Run，資料不會重新隨機分配，但會重新random鍵結值，並秀出新的鍵結值得執行結果。
>2. 開啟檔案會隨機分配2/3筆當訓練資料、1/3筆當測試資料
>3. Matplotlib的toolbar可以移動圖、放大、存圖檔。

### 二、簡介
>用python 3.5版開發，GUI使用python內建的Tkinter，結果圖使用需額外下載的matplotlib，程式分為四個檔案
>* main.py是程式進入點
>* MultilayerPerceptron.py為所有多層感知機的處理函式，包含讀入檔案、資料處理、計算辨識率、計算RMSE、前饋階段以及倒傳遞調整鍵結值
>* Neuron.py為神經元class，作為模擬神經元裡頭有儲存隨機鍵結值、累加函式、sigmoidal函式
>* GUI.py定義使用者介面，並顯示結果

###  三、重點程式碼說明
>在GUI按下run後會執行training_perceptron函式，首先會創建出三個神經元object，前兩個為隱藏層的神經元，最後一個神經元當作輸出層(全連接層)神經元，在create出神經元object時會random各自的鍵結值，在使用者設定的迭代次數內訓練鍵結值，每次迭代都會打亂輸入的訓練資料，採用pattern learning模式調整鍵結值，當所有訓練資料都調整或不調整權重後，計算該次的鍵結值所得的辨識率，辨識率高就存下來(即口袋演算法)，若是辨識率達到100%則會直接跳出for loop。
 
### 四、實驗結果(所有資料集都須有實驗結果和截圖)
>1. 基本題—perceptron1  
  ![](https://i.imgur.com/xc7O9fM.png)
 
>2. 基本題—perceptron2  
  ![](https://i.imgur.com/rC3JS8U.png)
 
>3. 基本題—2Ccircle1  
  ![](https://i.imgur.com/VjBG5kU.png)
 
>4. 基本題—2Circle1  
  ![](https://i.imgur.com/1ngZA70.png)
 
>5. 基本題—2 Circle2  
  ![](https://i.imgur.com/V2f7yQ0.png)
 
>6. 基本題—2CloseS  
  ![](https://i.imgur.com/Cy6SiN3.png)
 
>7. 基本題—2CloseS2  
 ![](https://i.imgur.com/9rhZdS0.png)
 
>8. 基本題—2CloseS3  
  ![](https://i.imgur.com/yfsbbsH.png)
 
>9. 基本題—2cring  
  ![](https://i.imgur.com/QceXHVT.png)
 
>10. 基本題—2CS  
  ![](https://i.imgur.com/h1rxOuq.png)
 
>11. 基本題—2Hcircle1  
  ![](https://i.imgur.com/QQmyAYd.png)
 
>12. 基本題—2ring  
  ![](https://i.imgur.com/h0loNZd.png)
 
### 五、實驗結果分析及討論
>學習率越小鍵結值每次的調整量越少，但只要訓練次數夠就能得到較佳的鍵結值。訓練次數少的話，初始隨機的鍵結值就會高度影響正確率。
>理論上隱藏層數增加能應對更複雜的case，則在訓練次數夠多的情況下，資料就算非線性可分割，訓練辨識率及測試辨識率還是能達到100%。
>RMSE在辨識率高的情況下會相當的小，反之則較大(接近1)。
