from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from train import begin

if __name__ == "__main__":
    root = Tk()
    root.title("风格迁移小工具")
    # root.geometry("1600x900")
    root.geometry("1000x700")
    # setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E + W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N + S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N + S + E + W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH, expand=1)

    path1 = 0
    path2 = 0


    # function to be called when mouse is clicked
    def printcoords_connet():
        global path1, path2
        File = filedialog.askopenfilename(parent=root,
                                          initialdir="D:\\learning\\deepLearning\\coputerVisualByPytorch\\picturetransform",
                                          title='Choose an image.')
        path1 = File
        img = Image.open(File)
        h, w = img.size
        if h > w:
            filename = ImageTk.PhotoImage(img.resize((700, 400)))
            canvas.image = filename  # <--- keep reference of your image
            canvas.create_image(150, 150, anchor='nw', image=filename)
        else:
            filename = ImageTk.PhotoImage(img.resize((400, 500)))
            canvas.image = filename  # <--- keep reference of your image
            canvas.create_image(300, 120, anchor='nw', image=filename)


    def printcoords_style():
        global path1, path2
        File = filedialog.askopenfilename(parent=root,
                                          initialdir="D:\\learning\\deepLearning\\coputerVisualByPytorch\\picturetransform",
                                          title='Choose an image.')
        path2 = File
        img = Image.open(File)
        h, w = img.size
        if h > w:
            filename = ImageTk.PhotoImage(img.resize((700, 400)))
            canvas.image = filename  # <--- keep reference of your image
            canvas.create_image(150, 150, anchor='nw', image=filename)
        else:
            filename = ImageTk.PhotoImage(img.resize((400, 500)))
            canvas.image = filename  # <--- keep reference of your image
            canvas.create_image(300, 120, anchor='nw', image=filename)


    def printcoods_result():
        global path1, path2
        if path1 != 0 and path2 != 0:
            path = begin(path1, path2)
            img = Image.open(path)
            h, w = img.size
            if h > w:
                filename = ImageTk.PhotoImage(img.resize((700, 400)))
                canvas.image = filename  # <--- keep reference of your image
                canvas.create_image(150, 150, anchor='nw', image=filename)
            else:
                filename = ImageTk.PhotoImage(img.resize((400, 500)))
                canvas.image = filename  # <--- keep reference of your image
                canvas.create_image(300, 120, anchor='nw', image=filename)


    connet = Button(root, text='选择内容图', height=2, width=14, font=36, command=printcoords_connet)
    connet.place(x=125, y=15)
    style = Button(root, text="选择风格图", height=2, width=14, font=36, command=printcoords_style)
    style.place(x=725, y=15)
    result = Button(root, text="合成图像", height=2, width=14, font=36, command=printcoods_result)
    result.place(x=425, y=15)
    root.mainloop()
