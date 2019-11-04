from tkinter import *  # Comes Defalut With Python3
from tkinter import filedialog as fd
from tkinter import messagebox as ms
import PIL  # Install Using PIP
from PIL import ImageTk, Image


class Image_Filters:
    'Common base class for all employees'

    def __init__(self):
        root = Tk()
        root.configure(bg='white')
        root.title('Image Filters')
        root.geometry("1000x500")
        root.mainloop()

    def displayCount(self):
        print


    def displayEmployee(self):
        print
