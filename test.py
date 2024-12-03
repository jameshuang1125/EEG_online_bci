import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton

#定義按鍵動作用的函式
def buttonClick():
    print("Button Clicked")

app = QApplication(sys.argv)
qwidget = QWidget()
qwidget.setWindowTitle("My first GUI button")

#建構按鍵
button = QPushButton("Fist Button", qwidget) #("按鍵名稱", 放置的widget)
#setToolTip是鼠標移動到按鍵上顯示的提示(不用按)
button.setToolTip("This will display message when I take mouse on button")
#移動按鍵(可以先不加執行看看位置)
button.move(100,100)  #視窗左上=(0, 0)  向右為+x, 向下為+y
#將按鍵與函式作結合
button.clicked.connect(buttonClick) #connect(函數名稱不用括號)

qwidget.show()

sys.exit(app.exec_())