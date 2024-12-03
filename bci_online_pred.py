from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import *

from bci_gui import Ui_MainWindow

import sys
import multiprocessing
from multiprocessing import Queue
import time
from datetime import datetime
import os
import shutil
import serial
from serial.tools import list_ports
# import serial.tools.list_ports
import numpy as np
from scipy import signal
import torch
from torch.autograd import Variable

from eeg_decoder import Decoder, Filter
from EEGConformer_spectrum import Conformer

SUBJECT_NAME = 'all_subjects_30_flip'
# ROOT_DIR = f"../EEG_dataset/4classes_all_for_train/offline"
# ROOT_DIR = f"../EEG_dataset/4classes_all_for_train/online_1"
ROOT_DIR = f"../EEG_dataset/2classes_all_for_train_spectrum"

model_state_dict_path = f"{ROOT_DIR}/{SUBJECT_NAME}/best_model_{SUBJECT_NAME}.pth"
mean_std_path         = f"../EEG_dataset/2classes_all_for_train_spectrum/{SUBJECT_NAME}/mean_std.txt"

def styled_text(text=None, color="#999999"):
    if text is None:
        text = datetime.now().strftime("%H:%M:%S")                
    return f"<span style=\" font-size:8pt; color:{color};\" >" + text + "</span>"    

class MyMainWindow(QtWidgets. QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)

        self.setupUi(self)
        self.setWindowTitle('Brain GUI')    # Window title

        # 按鍵功能
        self.btnCon.clicked.connect(self.StartConnection)   # 連線
        self.btnDisCon.clicked.connect(self.Disconnection)  # 斷線
        self.btnSave.clicked.connect(self.Savedata)  # 存檔

        # 多線程
        self.queue_data_save_flag = Queue()
        self.queue_plot_data = Queue()
        self.queue_model_data = Queue()
        self.queue_comport = Queue()
        self.queue_gui_message = Queue()

        # 建立資料接收class
        self.dt = DataReceiveThreads()  

        # 建立模型預測class
        self.mpt = ModelPredictThreads(model_state_dict_path, mean_std_path)

        # 多線程 : 開始接收封包
        self.multipDataRecv = multiprocessing.Process(target=self.dt.data_recv, 
                                                      args=(self.queue_comport, self.queue_plot_data, self.queue_data_save_flag, 
                                                            self.queue_model_data, self.queue_gui_message))
        
        # 多線程 2 : 模型預測
        self.multipOnlineBCI = multiprocessing.Process(target=self.mpt.online_bci, 
                                                       args=(self.queue_model_data, ))
        
        
        self.decoder = Decoder()
        self.raw_total = ""
        
        # ------------------------------------ #
        # Show all COM Port in combobox 
        # ------------------------------------ #
        default_idx = -1
        ports = serial.tools.list_ports.comports()
        for i, port in enumerate(ports):
            # port.device = 'COMX'
            
            if "USB 序列裝置" in port.description:
                default_idx = i
                self.queue_comport.put(port.device)
                print(f"Selected default COM : {port.description}")

                self.message.append(styled_text())                
                self.message.append(f'>> Default COM : {port.device}')

            self.comboBox.addItem(port.device + ' - ' + port.description)
        
        self.comboBox.setCurrentIndex(default_idx)
        self.comboBox.currentIndexChanged.connect(self.on_combobox_changed)

        self.timer_activate = False

    def on_combobox_changed(self, index):
        if index < 0:
            return
        # 取得選擇的 COM Port
        COM_PORT = self.comboBox.itemText(index).split(' ')[0]
        print(f'Selected Port: {COM_PORT}')
        self.queue_comport.put(COM_PORT)

        self.message.append(styled_text()) 
        self.message.append(f'>> Selected Port: {COM_PORT}')

    def StartConnection(self):
        # 連線        
        self.multipDataRecv.start()
        self.queue_data_save_flag.put(False)
        self.multipOnlineBCI.start()

        while True:
            if not self.queue_gui_message.empty():
                # Get last selected COM port name from queue
                message = self.queue_gui_message.get()

                self.message.append(styled_text()) 
                self.message.append(f'>> {message}')                
                break
        
    def Disconnection(self):
        if not self.multipDataRecv.is_alive():            
            print ("Process has not started")
        else:
            self.message.append(styled_text()) 
            self.message.append(f'>> All Processes had been\nterminated. You can close\nthis window.')

            print("All processes had been killed\nYou can close this window")
            self.multipDataRecv.terminate()
            self.multipOnlineBCI.terminate()
            self.queue_data_save_flag.put(False)

        if self.timer_activate:
            self.timer.stop()
            self.timer_2.stop()


    def Savedata(self):
        self.message.append(styled_text())      
        self.message.append(f'>> Start data streaming')
        self.queue_data_save_flag.put(True)
        
        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)   
        self.timer_activate = True

        self.start_time = time.time()
        self.timer_2 = QtCore.QTimer()
        self.timer_2.timeout.connect(self.update_time)
        self.timer_2.start(10)   

class DataReceiveThreads(Ui_MainWindow):
    def __init__(self):
        self.if_save = "0"
        self.data = ""
        self.count = 0
        self.total_data = ""

        self.count_model = 0
        self.total_data_model = ""
        self.small_data = ""

        # 創立當前時間的txt檔案
        ts = time.time()
        data_time = datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H%M')
        self.fileDir = './exp/{}'.format(data_time)

        if not os.path.isdir(self.fileDir):
            os.mkdir(self.fileDir)
            os.mkdir(self.fileDir + '\\1\\')
        else:
            shutil.rmtree(self.fileDir)
            os.mkdir(self.fileDir)
            os.mkdir(self.fileDir + '\\1\\')
        
        self.fileName = 'EEG.txt'

    def data_recv(self, queue_comport, queue_plot_data, queue_data_save_flag, queue_model_data, queue_gui_message):
        while True:            
            if not queue_comport.empty():
                # Get last selected COM port name from queue
                COM_PORT = queue_comport.get()
                break

        print(f"Open {COM_PORT}...")
        ser = serial.Serial(COM_PORT, 460800)
        print(f"Successfull Receive")
        queue_gui_message.put("Successfull Receive!\nReady to data streaming")

        # while True:
        #     ser.reset_output_buffer() 
        #     ser.reset_input_buffer()            
        #     if not queue_data_save_flag.empty():
        #         self.save_flag = queue_data_save_flag.get()
        #     if self.save_flag:
        #         break
    
        # f = open(f"{self.fileDir}/1/{self.fileName}", "a")

        # while True:     
        #     if not queue_data_save_flag.empty():
        #         self.save_flag = queue_data_save_flag.get()
        #     if not self.save_flag:
        #         # 結束後寫入最後收到的資料到EEG.txt
        #         with open(f"{self.fileDir}/1/{self.fileName}", "a") as f:
        #             f.write(self.total_data)
        #             f.close()
                
        #     # 每次讀取 32 bytes(一組EEG data的大小)並轉成16進位。收一次等於 1ms 的時間
        #     self.data = ser.read(32).hex() 
        #     self.total_data = self.total_data + self.data
        #     self.count = self.count + 1
            
        #     # -------------------------------------------------------- #
        #     # 送去畫圖的資料 (每 100 ms 寫入資料到txt的最尾端)
        #     # -------------------------------------------------------- #   
                                 
        #     if self.count >= 100:
        #         queue_plot_data.put(self.total_data)

        #         f.write(self.total_data)
        #         self.count = 0
        #         self.total_data = ""                

        #     # -------------------------------------------------------- #
        #     # 送進模型的資料
        #     # -------------------------------------------------------- #
        #     self.total_data_model = self.total_data_model + self.data
        #     self.count_model = self.count_model + 1

        #     if self.count_model >= 3000:
        #         # 將3s的資料丟進queue
        #         queue_model_data.put(self.total_data_model)
                
        #         # 經過 100 ms，raw長度 = 64*100 = 6400
        #         self.count_model     -= 100
        #         self.total_data_model = self.total_data_model[6400:]

class ModelPredictThreads(Ui_MainWindow):
    def __init__(self, model_state_dict_path, mean_std_path):
        self.decoder = Decoder()
        self.filter  = Filter()

        self.model_state_dict_path = model_state_dict_path
        self.mean_std_path = mean_std_path

    def z_score_normalization(self, data, mean, std):
        return (data - mean)/std

    def data_preprocessing(self, eeg_raw, mean_std):    
        """
        `eeg_raw` : decoded raw eeg data with shape = (3000, 8)

        return 
        shape of preprocessed_data = Tensor(1, 1, 8, 500)
        """

        # --------------------------------------------------------------------------------------------- #
        # 4-40 bandpass filtering
        # 沒有切掉超過閾值的訊號
        eeg_filtered = self.filter.filter(eeg_raw)
        data = eeg_filtered[900: -100].T
        data = data[0:8]
        # --------------------------------------------------------------------------------------------- #
        
        # --------------------------------------------------------------------------------------------- #
        # 1-100 bandpass filtering then cutoff and finally 4-40 bandpass filtering
        # 理論上要經過這些預處理
        # eeg_filtered = self.filter.filter(eeg_raw, hp_freq = 1, lp_freq = 100) 
        # for i in range(8):
        #     # threshold cutoff
        #     eeg_filtered[:, i][np.where(eeg_filtered[:, i] >=  5e-05)] = 5e-05
        #     eeg_filtered[:, i][np.where(eeg_filtered[:, i] <= -5e-05)] = -5e-05

        #     # filtering
        #     eeg_filtered[:, i] = self.filter.butter_bandpass_filter(eeg_filtered[:, i], 4, 40, 1000) 

        # data = eeg_filtered[900:-100].T
        # data = data[0:8]
        # --------------------------------------------------------------------------------------------- #


        # 降採樣
        data = signal.resample(data, 500, axis = 1) # shape = (8, 500)

        # z-socre normalization
        z_normalized_data = (data - mean_std[0])/mean_std[1]

        # 擴增維度，為符合模型輸入形狀。
        z_normalized_data = np.expand_dims(z_normalized_data, axis = 0) # shape = (1, 8, 500)
        preprocessed_data = np.expand_dims(z_normalized_data, axis = 1) # shape = (1, 1, 8, 500)
        preprocessed_data = torch.from_numpy(preprocessed_data)
        preprocessed_data = Variable(preprocessed_data.cuda().type(torch.cuda.FloatTensor))

        return preprocessed_data


    def model_prediction(self, data, model):
        """
        shape of data_list = (8, 2000)
        model_list : a python list of model entity
        """
        
        tok, outputs  = model(data)
        outputs = outputs.cpu().detach().numpy()
        predictions  = np.argmax(outputs)
                
        return predictions       


    def online_bci(self, queue_model_data):
        # ------------------------------------------------------------------- #
        # Load model
        # 如果在__init__()中載入模型會有問題，可能是記憶體不能共享
        # ------------------------------------------------------------------- #
        print(f"Load model with state_dict '{self.model_state_dict_path}'")

        self.model = Conformer(in_channels=1, eeg_channels=8, in_size=1640, n_classes=4, depth=3)
        self.model.load_state_dict(torch.load(self.model_state_dict_path))        
        self.model = self.model.cuda()           
        self.model.eval()

        # Load train mean & std
        with open(self.mean_std_path, 'r') as f:
            mean_std = f.read().split(',')
            self.mean_std = [float(mean_std[0]), float(mean_std[1])]

        # 預先載入模型，避免延遲
        print("Model pre-test result: ", end='')
        data   = torch.from_numpy(np.zeros((1, 1, 8, 500)))
        data   = Variable(data.cuda().type(torch.cuda.FloatTensor))        
        y_pred = self.model_prediction(data, self.model)
        print(y_pred)

        # ------------------------------------------------------------------- #
        # Model prediction
        # ------------------------------------------------------------------- #

        # 初始化滑動窗口以平滑輸出值
        num_classes = 2
        window_size = 10
        threshold   = 0.8
        window      = np.zeros((num_classes, window_size))
        
        while True:            
            try:                           
                queue_size = queue_model_data.qsize()                
                raw        = queue_model_data.get()
                time_start = time.time()

                # ------------------------------- #
                # Decode & filtering
                # ------------------------------- # 
                eeg_raw = self.decoder.get_BCI(raw, show_progress = False)

                # shape = (1, 1, 8, 500)
                data    = self.data_preprocessing(eeg_raw, self.mean_std) 

                # ------------------------------- #
                # model prediction
                # ------------------------------- #                
                y_pred = self.model_prediction(data, self.model)

                # ------------------------------- #
                # Smooth the outputs        
                # ------------------------------- #
                idx = y_pred
                window[:, 1:] = window[:, :-1]
                window[:, 0] = 0
                window[idx][0] = 1

                mean_window = np.mean(window, axis=1)
                max_prob = np.max(mean_window)
                if max_prob < threshold:
                    smooth_output = 0
                else:
                    smooth_output = np.argmax(mean_window)

                # ------------------------------- #
                # Print prediction and save
                # ------------------------------- #
                # print(f"{time_str} [{mean_window[0]:>3}, {mean_window[1]:>3}, {mean_window[2]:>3}, {mean_window[3]:>3}] {smooth_output} {y_pred}")
    
                time_str = datetime.now().strftime('%H:%M:%S:%f')[:-5]
                duration = (time.time() - time_start)*1000    
                print(f"{time_str} - {duration:>4.1f}ms    "\
                    f"\033[90mqueue_size\033[0m  {queue_size}    "\
                    f"\033[90mprediction\033[0m  {y_pred}")

                try:
                    with open("pred.txt", mode="w") as f:
                        f.write(str(y_pred))
                except:
                    print("File access conflict !")
            except:
                print("Error")
                pass

if __name__ == '__main__':
    # Create an instance of QApplication
    app = QtWidgets.QApplication(sys.argv)
    # Create an instance of our window
    mainWindow = MyMainWindow()
    # Show the window
    mainWindow.show()
    # Run the application's event loop
    sys.exit(app.exec_())