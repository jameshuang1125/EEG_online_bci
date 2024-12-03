from multiprocessing import Process, Queue

def func(q, name):                        #以參數的方式將對列物件以參數型式導入子進程          
    q.put('My Process_name is %s and put the data to the id %d queue' %(name, id(q)))
       
if __name__ == "__main__":
    q = Queue()              #於主進程創建隊列物件
    process_list=[]
    print("main queue id: %d" %id(q))

    for i in range(3):
        proc = Process(target=func, args=(q, 'Process-%d' %i))
        process_list.append(proc)
        proc.start()

    print(q.get())    #往管道中取數據
    print(q.get())
    print(q.get())

    for each_process in process_list:
        each_process.join()