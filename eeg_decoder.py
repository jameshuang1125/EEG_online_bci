# 解碼收到的腦波資料 EEG.txt，並濾波
# text color : https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
#
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from matplotlib.ticker import StrMethodFormatter
from scipy import signal


def get_latest_date(exp_path):
    # exp_path : path of exp directory. e.g. './exp'
    # return : latest date in exp_path. e.g. '2022_01_01_0000'

    # find all dirs in exp_path which format is 'XXXX_XX_XX_XXXX' (YEAR_MM_DD_XXXX) 
    exp_date = findall_datadate_dirs(exp_path)

    exp_date_int = [int("".join(item.split("_"))) for item in exp_date]
    data_date = exp_date[np.argmax(exp_date_int)]

    return data_date


def findall_datadate_dirs(EXP_DIR):
    """
    列出EXP_DIR中所有檔名格式為'XXXX_XX_XX_XXXX' (YEAR_MM_DD_XXXX)的腦波資料夾

    假設EXP_DIR中長這樣
    
    - EXP_DIR
        - 2023_01_XX_XXXX
        - 2023_02_XX_XXXX
        - 2023_03_XX_XXXX
        - new_dir
        - text.txt

    會回傳['2023_01_XX_XXXX', '2023_02_XX_XXXX', '2023_03_XX_XXXX']
    """
    import re
    regex = re.compile(r"\d{4}_\d{2}_\d{2}_\d{4}")
    data_dates = []
    for name in os.listdir(EXP_DIR):
        is_matched = regex.match(name)
        if (is_matched):
            data_dates.append(regex.match(name).group())
    return data_dates    

    
def progress_bar(title, temp, total):
    if temp >= total:
        temp = total
    # print('\r' + '['+title+']:[%s%s] %s(%.2f%%)' % ('█' * int(temp/total*20), ' ' * (20-int(temp/total*20)), str(temp)+'/'+str(total), float(temp/total*100)), end='')
    print('\r{}: |{}{}| {}/{} [{:.2f}%]'.format(title, 
                                                '█' * int(temp/total*25), ' ' * (25-int(temp/total*25)), 
                                                str(temp), str(total), 
                                                float(temp/total*100)),   
                                                end='')

class Decoder():
    def __init__(self):
        self.comp = 8388607  # complement (7FFFFF)??why
        self.F = Filter()

    def decode(self, raw, show_progress=False, hp_freq=4, lp_freq=40):
        """
        decode EEG.txt and filtering

        Parameters
        ----------------------------------------
        `raw` : raw HEX eeg data, e.g. content of './exp/2022_10_20_XXXX/1/EEG.txt'
        `show_progress` : print progress bar if true, not print if false


        Return
        ----------------------------------------
        `eeg_filtered` : filtered eeg data with shape = (n, 10)   


        Examples
        ----------------------------------------
        >>> decoder = Decoder()
        >>> eeg_filtered = decoder.decode(raw, show_progress = True)
        """
        
        eeg_raw = self.get_BCI(raw, show_progress)        
        eeg_filtered = self.F.filter(eeg_raw, hp_freq=hp_freq, lp_freq=lp_freq)
        return eeg_filtered

    def decode_to_txt(self, eeg_txt_path, return_data = False):
        """
        # 解碼16進制EEG.txt資料至10進制1.txt資料
        decode, filtering EEG.txt and then write into 
        1.txt/1.npy (4 - 40Hz) and 1_100.txt/1_100.npy (1 - 100Hz)

        Parameters
        ----------------------------------------
        `eeg_txt_path` : raw HEX eeg data, e.g. content of './exp/2022_10_20_XXXX/1/EEG.txt'
        `return_data` : True : return filtered eeg data with shape = (n, 10)
                        False : just write txt and plot eeg

        Return
        ----------------------------------------
        `eeg_4_40` : filtered eeg data with shape = (n, 10)   

        Examples
        ----------------------------------------
        >>> decoder = Decoder()
        >>> eeg_filtered = decoder.decode_to_txt(eeg_txt_path = eeg_txt_path, return_data=True) 
        """
        start_t = time.time()
        print("Process file >>", end='')
        print('\033[92m {} \033[0m\n'.format(eeg_txt_path))

        with open(eeg_txt_path, 'r') as f:
            raw = f.read()

        eeg_raw   = self.get_BCI(raw, show_progress = True) 
        eeg_4_40  = self.F.filter(eeg_raw.copy(), hp_freq = 4, lp_freq = 40)
        eeg_1_100 = self.F.filter(eeg_raw.copy(), hp_freq = 1, lp_freq = 100)        

        # ---------------------------------------------------------------------------- #
        print("")        
        print("\nSaving 1.txt file...")
        np.savetxt('/'.join(eeg_txt_path.split('/')[0:-1]) + '/1.txt', eeg_4_40, delimiter='\t')
        np.savetxt('/'.join(eeg_txt_path.split('/')[0:-1]) + '/1_100.txt', eeg_1_100, delimiter='\t')


        print("Saving 1.npy file...")
        np.save('/'.join(eeg_txt_path.split('/')[0:-1]) + '/1.npy', eeg_4_40) 
        np.save('/'.join(eeg_txt_path.split('/')[0:-1]) + '/1_100.npy', eeg_1_100) 


        print('\nCost: {0:.1f} s'.format(int((time.time() - start_t) * 10) * 0.1))
        print(f"Decoded EEG data containing {np.size(eeg_4_40, 0):,} timepoints (~ {np.size(eeg_4_40, 0)//1000} s) with {np.size(eeg_4_40, 1)} channels")
        print("Ploting...")
        print('-'*40)
        
        self.plot_eeg(eeg_4_40, png_path='/'.join(eeg_txt_path.split('/')[0:-2]) + '/EEG_4_40.png')

        if return_data:
            return eeg_4_40   

    def read_decoded(self, file_path):
        """
        read decoded eeg data file. e.g. 1.txt

        Parameters
        ----------------------------------------
        `file_path` : path of eeg data file. e.g. ./exp/2022_10_20_XXXX/1/1.txt

        Return
        ----------------------------------------
        `data` : eeg data with shape = (n, 10)


        Examples
        ----------------------------------------
        >>> decoder = Decoder()
        >>> file_path = ./exp/2022_10_20_XXXX/1/1.txt
        >>> eeg_data = decoder.read_decoded(file_path) # 讀取解碼後的EEG資料，1.txt
        """        

        if not (os.path.exists(file_path)):
            print(f"\n路徑 : \033[92m{file_path}\033[0m 1.txt(1.npy)檔案不存在(解碼EEG.txt後產生之檔案)，請解碼後再執行\n")        
            raise Exception('目標資料夾沒有 1.txt 或是 1.npy 檔案，請確認後再執行!')

        print("Read file >>", end='')
        print('\033[92m {} \033[0m'.format(file_path))
        print("")

        if file_path[-3:] == 'npy':
            # 讀取解碼後的EEG資料，1.npy
            print("Loading...")     
            data = np.load(file_path) 
              
        else:
            # 讀取解碼後的EEG資料，1.txt            
            data = []
            f = open(file_path)
            num_lines = sum([1 for line in open(file_path)])
            for i, line in enumerate(f.readlines()):
                content = line.split()  # 指定空格作為分隔符對line進行切片
                content = list(map(float, content)) # string轉成float
                data.append(content)
                if i % 1000 == 0 or i + 1000 >= num_lines-1:
                    progress_bar("Reading", i, num_lines-1)
        return np.array(data)

    def find_trigger(self, eeg_data):
        """
        預期會收到連續五個一樣的trigger，找出trigger的位置和值
        如果trigger值不為not_trigger時，暫存接下來的5個trigger，
        以這5個trigger中出現最多次的值當作trigger，
        再接下來的10個trigger都不去判斷，避免誤判

        Parameters
        ----------------------------------------
        `eeg_data` : eeg data with shape = (n, 10) which
                     8 is the number of channel,
                     n is the number of sample points
                     
        Return
        ----------------------------------------
        `triggers` : a 3d list with shape = (num_trigger, index, value) 
                     contains trigger index and value e.g. [[index1, trigger1], [index2, trigger2],...]


        Examples
        ----------------------------------------
        >>> decoder = Decoder()
        >>> triggers = decoder.find_trigger(eeg_data, not_triggered)
        """   
        # 找出not_trigger的值
        trigger_count = np.unique(eeg_data.T[9][:], return_counts=True)
        not_triggered = trigger_count[0][np.argmax(trigger_count[1])]        

        flag = 0
        count = 0
        triggers = []
        temp_trigger = []
        for i in range(len(eeg_data)):
            trigger = eeg_data[i][9]
            
            if count > 0:
                count -= 1
                # 抓前五個trigger值
                if count > 10: 
                    temp_trigger.append(trigger)

                if count == 0:
                    flag = 1
            else:            
                # trigger出現時
                if trigger != not_triggered and trigger != 0:
                    temp_trigger.append(trigger)   
                    # 設定不反應的長度 (只抓trigger變化的瞬間)         
                    count = 15 

            if flag == 1:
                trigger_count = np.unique(temp_trigger, return_counts=True) # retrun [[unique_value], [counts]]
                # find max counts of trigger
                trigger = trigger_count[0][np.argmax(trigger_count[1])] 
                
                triggers.append([i-5, trigger])
                temp_trigger = []
                flag = 0   

        return triggers

    def plot_eeg(self, eeg_data, png_path):
        """
        remove first 1 second and and last 1 second

        Parameters
        ----------------------------------------
        `eeg_data` : eeg data with shape = (n, 10) which
                     8 is the number of channel,
                     n is the number of sample points
        
        `png_path` : path for save figure
        

        Examples
        ----------------------------------------
        >>> decoder = Decoder()
        >>> decoder.plot_eeg(eeg_filtered, png_path='/'.join(eeg_txt_path.split('/')[0:-2]) + '/EEG.png', not_triggered = 0)
        """
        
        eeg_data = eeg_data[1000:-1000] # remove first 1 second and last 1 second
        n = len(eeg_data) # number of samples 
        num_channel = 8

        # check the continuity of frameID
        frame_id = np.zeros([n - 1])
        for i in range(n - 1):
            frame_id[i] = eeg_data[i + 1][8] - eeg_data[i][8]
            if frame_id[i] == -255:
                frame_id[i] = 1   

        # trigger
        # 找出not_trigger的值
        trigger_count = np.unique(eeg_data.T[9][:], return_counts=True)
        not_triggered = trigger_count[0][np.argmax(trigger_count[1])]          
        triggers = self.find_trigger(eeg_data)
        
        # set x_tick and xticklabels, 不同秒數設定不同的時間跨度
        sec = n // 1000
        if sec < 10:                        point_gap = 1000
        elif (10 <= sec)  and (sec < 50):   point_gap = 5000 
        elif (50 <= sec)  and (sec < 100):  point_gap = 10000
        elif (100 <= sec) and (sec < 200):  point_gap = 50000 
        elif (200 <= sec) and (sec < 500):  point_gap = 100000     
        elif (500 <= sec) and (sec < 1000): point_gap = 200000
        else:                               point_gap = 200000

        x_tick = np.arange(1, n, point_gap)
        x_tick = np.append(x_tick, n) # 最後一筆資料秒數的位置
        xticklabels = [str(int(time/1000)+1) for time in range(0, n, point_gap)] # name of x_ticks (time)                                
        xticklabels[0] = 1 # 從第一秒開始
        xticklabels.append(str(round(n/1000, 1)+1)) # 顯示最後一筆資料的秒數
        
        #
        # plot
        fig = plt.subplots(4, 3, figsize=(14, 6))
        color = '#17395C'
        linewidth = 0.5
        fontsize = 8                       
        plt.style.use('ggplot')
        label_list = ['F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'FrameID', 'Trigger\n(invert logic,\nsubtracted by 255)']

        #
        eeg_data = np.array(eeg_data).T # shape (n, 10) -> (10, n)

        x_axis = np.arange(1, n + 1)
        for i in range(num_channel + 2):
            row = i // 3
            column = i % 3
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.5f}')) # 4 decimal places
            if i < 8: # ch0 - ch7
                y = eeg_data[i]                
                plt.subplot2grid((4, 3), (row, column))
                plt.plot(x_axis, y, label=label_list[i], c=color, lw=linewidth)
                
            elif i == 8: # frameID
                y = frame_id
                x_axis = np.arange(1, n)
                plt.subplot2grid((4, 3), (row, column))
                plt.plot(x_axis, y, label=label_list[i], c=color, lw=linewidth)                
                plt.ylim([-0.2, 2.2])
                plt.yticks([])

            elif i == 9: # trigger
                y = eeg_data[i]

                trigger = [[i, 1] if x != not_triggered else [0, 0] for i, x in enumerate(y)] # 找出所有trigger，高度顯示設定1
                x_axis = np.arange(1, n + 1)
                plt.subplot2grid((4, 3), (3, 0), colspan = 3)
                plt.plot(x_axis, [x[1] for x in trigger], label=label_list[i], c=color, lw=1)
                
                # 顯示trigger的值
                for trigger in triggers: 
                    if not_triggered > 0:
                        # 反邏輯 (not_triggered = 255, trigger = 254, 253, 252,...)
                        plt.text(trigger[0], 1.5, int(not_triggered - trigger[1]), c=color, ha='center', va='center', fontsize=fontsize)
                    else:
                        # 正邏輯 (not_triggered = 0, trigger = 1, 2, 3,...)
                        plt.text(trigger[0], 1.5, int(trigger[1]), c=color, ha='center', va='center', fontsize=fontsize)
                                        
                plt.yticks([])
                plt.ylim([-0.2, 2])
                plt.margins(x = 0.01)  

            plt.xticks(x_tick, xticklabels, fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.legend(labelcolor='darkorange', fontsize=fontsize, loc='upper right')
        plt.suptitle("EEG (sec)")                        
        plt.tight_layout()
        plt.savefig(png_path, dpi=500)  
        

    def HEX_to_DEC(self, hexdata):  
        # 16轉10進制 
        return int(hexdata, base=16)
    

    # 改寫後的程式，但執行速度沒有原本快
    # def get_BCI(self, raw_data, show_progress = False):
    #     """
    #     解碼raw eeg data(16進制)成10進制

    #     Parameters
    #     ----------------------------------------
    #     `raw_data` : raw eeg data, e.g. EEG.txt 
  

    #     Return
    #     ----------------------------------------
    #     `output` : shape = (n, 10), 8ch + framid + trigger


    #     Examples
    #     ----------------------------------------        
    #     >>> with open('EEG.txt', 'r') as f:
    #     >>>     raw = f.read()
    #     >>> decoder = Decoder()
    #     >>> eeg_raw = decoder.get_BCI(raw, show_progress)
    #     >>> eeg_filtered = F.filter(eeg_raw)        
    #     """
    #     import re

    #     pattern = r"5170\w{56}a1"
    #     matched_raw = np.array(re.findall(pattern, raw_data))

    #     # 封包數量
    #     # n = len(matched_raw)
    #     n = (len(raw_data) // 64)  # 確認幾組資料，每個封包的大小是64，所以n為封包數目
    #     output = np.zeros([n, 11])

    #     for i, raw in enumerate(matched_raw):
    #         # raw     = matched_raw[i]

    #         # 擷取每筆資料的 ch1~ch8
    #         ch1     = self.HEX_to_DEC(raw[10: 16])  # 6
    #         ch2     = self.HEX_to_DEC(raw[16: 22])  # 6
    #         ch3     = self.HEX_to_DEC(raw[22: 28])  # 6
    #         ch4     = self.HEX_to_DEC(raw[28: 34])  # 6
    #         ch5     = self.HEX_to_DEC(raw[34: 40])  # 6
    #         ch6     = self.HEX_to_DEC(raw[40: 46])  # 6
    #         ch7     = self.HEX_to_DEC(raw[46: 52])  # 6
    #         ch8     = self.HEX_to_DEC(raw[52: 58])  # 6

    #         # 
    #         frameID = self.HEX_to_DEC(raw[4: 6])  # 2
    #         leadOff = self.HEX_to_DEC(raw[7: 9])  # 2
    #         leadOff = bin(int(leadOff))[2:] # 轉成2進制
    #         trigger = self.HEX_to_DEC(raw[58: 60])

    #         # 
    #         output[i, :-1] = [ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, frameID, trigger]

    #         # 讓八通道腦波資料的十進位值有正負
    #         mask = output[i, :] > self.comp
    #         output[i, :] = np.where(mask, output[i, :] - 2*self.comp, output[i, :])
    #         output[i, -1] = leadOff
    #         # 將收到的data換成實際電壓 8388607 = 2**23 - 1
    #         output[i, :8] = output[i, :8]*2.5/8388607

    #         if show_progress:
    #             if i % 1000 == 0 or i + 1000 >= n-1:
    #                 progress_bar("Decoding", i, n-1)                    

    #     # 八通道腦波資料(每個通道以十進制存), shape = (n, 11)
    #     return output          

    # 原本程式
    def get_BCI(self, raw_data, show_progress = False):
        datalength = len(raw_data)
        n = (datalength // 64)  # 確認幾組資料，每個封包的大小是64，所以n為封包數目
        # output = np.zeros([n, 10])  # 返回用0填充的數組 10*n的陣列，9+trigger
        output = np.zeros([n, 11])  # 返回用0填充的數組 10*n的陣列，9+trigger+leadOff
        ch1 = np.zeros([n])
        ch2 = np.zeros([n])
        ch3 = np.zeros([n])
        ch4 = np.zeros([n])
        ch5 = np.zeros([n])
        ch6 = np.zeros([n])
        ch7 = np.zeros([n])
        ch8 = np.zeros([n])
        frameID = np.zeros([n]) 
        trigger = np.zeros([n])
        leadOff = np.zeros([n])

        i = 0
        j = 0
        while i < len(raw_data) - 62:            
            # 確認頭尾封包(頭2bytes:5170 ; 尾1byte: a1)
            if raw_data[i:i + 4] == "5170" and raw_data[i + 60:i + 62] == "a1":
                # 擷取每筆資料的 frame_id

                frameID[j] = self.HEX_to_DEC(raw_data[i + 4:i + 6])  # 2
                output[j, 8] = frameID[j]

                leadOff[j] = self.HEX_to_DEC(raw_data[i + 7:i + 9])  # 2
                leadOff[j] = bin(int(leadOff[j]))[2:] # 轉成2進制
                output[j, 10] = leadOff[j]              

                # 擷取每筆資料的 ch1~ch8
                ch1[j] = self.HEX_to_DEC(raw_data[i + 10:i + 16])  # 6
                if ch1[j] > self.comp:  # 讓他有正有負
                    ch1[j] = ch1[j]-2*self.comp
                output[j, 0] = ch1[j]

                ch2[j] = self.HEX_to_DEC(raw_data[i + 16:i + 22])  # 6
                if ch2[j] > self.comp:
                    ch2[j] = ch2[j]-2*self.comp
                output[j, 1] = ch2[j]

                ch3[j] = self.HEX_to_DEC(raw_data[i + 22:i + 28])  # 6
                if ch3[j] > self.comp:
                    ch3[j] = ch3[j]-2*self.comp
                output[j, 2] = ch3[j]

                ch4[j] = self.HEX_to_DEC(raw_data[i + 28:i + 34])  # 6
                if ch4[j] > self.comp:
                    ch4[j] = ch4[j]-2*self.comp
                output[j, 3] = ch4[j]

                ch5[j] = self.HEX_to_DEC(raw_data[i + 34:i + 40])  # 6
                if ch5[j] > self.comp:
                    ch5[j] = ch5[j]-2*self.comp
                output[j, 4] = ch5[j]

                ch6[j] = self.HEX_to_DEC(raw_data[i + 40:i + 46])  # 6
                if ch6[j] > self.comp:
                    ch6[j] = ch6[j]-2*self.comp
                output[j, 5] = ch6[j]

                ch7[j] = self.HEX_to_DEC(raw_data[i + 46:i + 52])  # 6
                if ch7[j] > self.comp:
                    ch7[j] = ch7[j]-2*self.comp
                output[j, 6] = ch7[j]

                ch8[j] = self.HEX_to_DEC(raw_data[i + 52:i + 58])  # 6
                if ch8[j] > self.comp:
                    ch8[j] = ch8[j]-2*self.comp
                output[j, 7] = ch8[j]

                trigger[j] = self.HEX_to_DEC(raw_data[i + 58:i + 60])  # trigger
                if trigger[j] > self.comp:
                    trigger[j] = trigger[j]-2*self.comp
                output[j, 9] = trigger[j]

                i += 64  # 一組資料32bytes(每讀完一組平移32bytes)
                j += 1  # 每組整理好後的資料
                if show_progress:
                    if j % 1000 == 0 or j + 1000 >= n-1:
                        progress_bar("Decoding", j, n-1)
            else:
                i += 2  # 若沒有讀到頭尾封包，往後找1byte            
            
        output[:, :8] = output[:, :8] * 2.5 / (2**23 - 1)  # 將收到的data換成實際電壓
        return output


class Filter():
    def __init__(self):
        self.fs = 1000

    def filter(self, eeg_raw, hp_freq=4, lp_freq=40):
        """
        濾波 : 60Hz市電、120諧波，4 - 40 Hz 帶通濾波取出腦波頻率範圍

        Parameters
        ----------------------------------------
        `eeg_raw` : shape = (n, 10)，8通道+frameID+Trigger
  

        Return
        ----------------------------------------
        `eeg_raw` : shape = (n, 10) 


        Examples
        ----------------------------------------
        >>> F = Filter()
        >>> eeg_filtered = F.filter(eeg_raw)        
        """

        for i in range(8): # ch1 ~ ch8
            # 60 Hz notch
            eeg_raw[:, i] = self.butter_bandstop_filter(eeg_raw[:, i], 55, 65, self.fs)            
            # 120 Hz notch, 60Hz 諧波
            eeg_raw[:, i] = self.butter_bandstop_filter(eeg_raw[:, i], 115, 125, self.fs)            
            # 4 - 40 Hz bandpass
            eeg_raw[:, i] = self.butter_bandpass_filter(eeg_raw[:, i], hp_freq, lp_freq, self.fs) 
        return eeg_raw
            
    def butter_bandpass(self, lowcut, highcut, fs, order=3):  # fs & order??#EEG:3，EMG:6
        nyq = 0.5 * fs
        # nyquist frequency(fs is the sampling rate, and 0.5 fs is the corresponding Nyquist frequency.)
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandstop(self, lowcut, highcut, fs, order=3):  # 55~65Hz
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='stop')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=3):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def butter_bandstop_filter(self, data, lowcut, highcut, fs, order=3):
        b, a = self.butter_bandstop(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

