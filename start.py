import tkinter
from face_recognition import collect, train, recognition


def faces_collect_click():
    result = tkinter.simpledialog.askstring(title='获取信息', prompt='请输入姓名：')
    collect.start(result)
    train.start()
    tkinter.messagebox.showinfo(title='人脸录入', message=result+'录入成功！')
def faces_train_click():
    result = train.start()
    tkinter.messagebox.showinfo(title='训练模型', message='训练完成！正确率：'+result+'!')
def face_recognition_click():
    result = recognition.start()
    if result == 0:
        tkinter.messagebox.showwarning(title='识别结果', message='识别失败！')
    else:
        tkinter.messagebox.showinfo(title='识别结果', message=result)


root = tkinter.Tk()
root.geometry('220x100+500+200')
root.title('人脸识别')

faces_collect_button = tkinter.Button(root, text="录入", command=lambda: faces_collect_click())
faces_recognition_button = tkinter.Button(root, text="识别", command=lambda: face_recognition_click())

faces_collect_button.pack(side='letrain', padx=30, ipadx=10)
faces_recognition_button.pack(side='letrain', padx=20, ipadx=10)
root.mainloop()