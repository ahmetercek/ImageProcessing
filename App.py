import tkinter
from tkinter.ttk import Combobox

import cv2
import PIL.Image, PIL.ImageTk
from PIL import ImageTk
from matplotlib import pyplot as plt
import numpy as np
from tkinter import filedialog as fd
from tkinter import messagebox as ms
from PIL import ImageTk, Image

class App:

    def __init__(self, window, window_title, image_path="coin.jpg"):
        self.image_path = image_path
        self.window = window
        self.window.title(window_title)
        # Load an image using OpenCV
        self.cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
        self.height, self.width, no_channels = self.cv_img.shape

        # Create a canvas that can fit the above image
        self.canvas = tkinter.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))



        # Add a PhotoImage to the Canvas

        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        self.canvas.grid(row=1, column=2)

        self.photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))
        self.canvas2 = tkinter.Canvas(window, width=self.width, height=self.height)
        self.canvas2.create_image(0, 0, image=self.photo2, anchor=tkinter.NW)
        self.canvas2.grid(row=1, column=0)


        self.btn_loadImage = tkinter.Button(window, text="Select Image", width=30, command=self.make_image, fg='white', bg='blue')
        self.btn_loadImage.grid(row=0, column=0)

        self.btn_loadOriginalImage = tkinter.Button(window, text="Reset Changes", width=30, command=self.display_original, fg='white', bg='red')
        self.btn_loadOriginalImage.grid(row=0, column=1)

        self.btn_SaveImage = tkinter.Button(window, text="Save Image", width=30, command=self.save_image, fg='white', bg='green')
        self.btn_SaveImage.grid(row=0, column=2)

        # Image_Enhancement
        w = tkinter.Label(window, text="Image Filters", width=50)
        w.grid(row=2, column=0)
        secenek = ["Averaging Filtering", "Blur Filtering", "Gaussian Filtering", "Median Filtering", "Bilateral Filtering",
                   "SobelX Filtering", "SobelY Filtering", "SobelXY Filtering", "Laplacian Filtering", "Canny Filtering"]
        self.combo = Combobox(window, values=secenek, width=50)
        self.combo.grid(row=2, column=1)
        self.btn_blur = tkinter.Button(window, text="Apply", width=50, command=self.image_filter)
        self.btn_blur.grid(row=2, column=2)

        # Histogram
        w2 = tkinter.Label(window, text="Histogram Operations", width=50)
        w2.grid(row=3, column=0)
        secenek2 = ["Histogram", "Histogram Equalization"]
        self.combo2 = Combobox(window, values=secenek2, width=50)
        self.combo2.grid(row=3, column=1)
        self.btn_blur2 = tkinter.Button(window, text="Apply", width=50, command=self.image_histogram)

        self.btn_blur2.grid(row=3, column=2)

        # Transform
        w3 = tkinter.Label(window, text="Transform Operations", width=50)
        w3.grid(row=4, column=0)
        secenek3 = ["Resize  1/2", "Cropping", "Rotation 90 Clockwise", "Perspective Transformation", "Affine Transformation"]
        self.combo3 = Combobox(window, values=secenek3, width=50)
        self.combo3.grid(row=4, column=1)
        self.btn_blur3 = tkinter.Button(window, text="Apply", width=50, command=self.image_transforms)

        self.btn_blur3.grid(row=4, column=2)

        # Intensity
        w4 = tkinter.Label(window, text="Intensity Operations", width=50)
        w4.grid(row=5, column=0)
        secenek4 = ["++Brightness", "--Brightness"]
        self.combo4 = Combobox(window, values=secenek4, width=50)
        self.combo4.grid(row=5, column=1)
        self.btn_blur4 = tkinter.Button(window, text="Apply", width=50, command=self.give_brightness)

        self.btn_blur4.grid(row=5, column=2)

        # Morphological
        w5 = tkinter.Label(window, text="Morphological Operations", width=50)
        w5.grid(row=6, column=0)
        secenek5 = ["Binary Erosion", "Binary Dilation", "Opening", "Closing", "Morphological Gradient",
                    "Top Hat", "Black Hat", "Morphological Rect", "Morphological Ellipse", "Morphological Cross"]
        self.combo5 = Combobox(window, values=secenek5, width=50)
        self.combo5.grid(row=6, column=1)
        self.btn_blur5 = tkinter.Button(window, text="Apply", width=50, command=self.morphological_operations)

        self.btn_blur5.grid(row=6, column=2)


        self.btn_OpenVideoProcesser = tkinter.Button(window, text="Video Edit ('q' for quit)", width=100, command=self.OpenVideoProcesser, fg='white', bg='red')
        self.btn_OpenVideoProcesser.grid(row=7, column =1)

        self.window.mainloop()

    def blur_image(self):
        self.blur_imageFile = cv2.blur(self.cv_img, (3, 3))
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.blur_imageFile))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
    def make_image(self):
        try:
            File = fd.askopenfilename()
            self.image_path = File
            self.cv_img = cv2.cvtColor(cv2.imread(File), cv2.COLOR_BGR2RGB)
            # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
            self.height, self.width, no_channels = self.cv_img.shape
            # Create a canvas that can fit the above image
            # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
            image = PIL.Image.fromarray(self.cv_img)
            image = image.resize((500,500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            # Add a PhotoImage to the Canvas
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

            self.photo2 = PIL.ImageTk.PhotoImage(image=image)
            # Add a PhotoImage to the Canvas
            self.canvas2.create_image(0, 0, image=self.photo2, anchor=tkinter.NW)
        except:
            ms.showerror('Error!', 'File type is unsupported.')
    def save_image(self):
        try:
            File = fd.asksaveasfile(defaultextension=".jpg", filetypes=(("JPEG file", "*.jpg"),("All Files", "*.*")))
            myimage = PIL.ImageTk.getimage(self.photo)
            rgb_im = myimage.convert('RGB')
            if File:
                rgb_im.save(File)
            File.close()
        except:
            ms.showerror('Error!', 'File type is unsupported.')
    def display_original(self):

        try:
            self.cv_img = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
            # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
            self.height, self.width, no_channels = self.cv_img.shape
            # Create a canvas that can fit the above image
            # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
            image = PIL.Image.fromarray(self.cv_img)
            image = image.resize((500,500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            # Add a PhotoImage to the Canvas
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        except:
            ms.showerror('Error!', 'File type is unsupported.')

    def image_filter(self):
        text = self.combo.get()
        if(text == "Blur Filtering"):
            self.blur_imageFile = cv2.blur(self.cv_img, (10, 10))
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500,500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif(text == "Averaging Filtering"):
            self.blur_imageFile = cv2.blur(self.cv_img, (3, 3))
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500,500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif (text == "Gaussian Filtering"):
            self.blur_imageFile = cv2.GaussianBlur(self.cv_img, (5, 5),0)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500,500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif(text == "Median Filtering"):
            self.blur_imageFile = cv2.medianBlur(self.cv_img, 5)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500,500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif(text == "Bilateral Filtering"):
            self.blur_imageFile = cv2.bilateralFilter(self.cv_img, 9,75,75)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500,500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif(text == "SobelX Filtering"):
            self.cv_img2 = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            self.blur_imageFile = cv2.Sobel(self.cv_img2, cv2.CV_64F, 1 , 0)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500,500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif(text == "SobelY Filtering"):
            self.cv_img2 = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            self.blur_imageFile = cv2.Sobel(self.cv_img2, cv2.CV_64F, 0 , 1)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500,500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif(text == "SobelXY Filtering"):
            self.cv_img2 = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            self.blur_imageFile = cv2.Sobel(self.cv_img2, cv2.CV_64F, 1 , 1)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500,500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif(text == "Laplacian Filtering"):
            self.cv_img2 = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            self.blur_imageFile = cv2.Laplacian(self.cv_img2, cv2.CV_64F)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500,500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif(text == "Canny Filtering"):
            self.cv_img2 = cv2.imread(self.image_path)
            self.blur_imageFile = cv2.Canny(self.cv_img2, 100, 150)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500,500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        else:
            print("Please Select")
    def image_histogram(self):
        text = self.combo2.get()
        if(text == "Histogram"):
            plt.hist(self.cv_img.ravel(), 256, [0, 256])
            plt.show()
        elif(text == "Histogram Equalization"):
            self.cv_img2 = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            self.blur_imageFile = cv2.equalizeHist(self.cv_img2)
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.blur_imageFile))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        else:
            print("Please Select")


    def image_transforms(self):
        text = self.combo3.get()
        if(text == "Resize  1/2"):
            self.resized = cv2.imread(self.image_path)
            self.blur_imageFile = cv2.resize(self.resized, None, fx=0.50, fy=0.50, interpolation=cv2.INTER_CUBIC)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((250, 250), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif(text == "Cropping"):
            self.cropped = cv2.imread(self.image_path)
            self.blur_imageFile = self.cropped[100:300, 100:300]
            image = PIL.Image.fromarray(self.blur_imageFile)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif (text == "Rotation 90 Clockwise"):
            self.cropped = cv2.imread(self.image_path)
            self.blur_imageFile = cv2.rotate(self.cropped, cv2.ROTATE_90_CLOCKWISE)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500, 500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif (text == "Perspective Transformation"):
            self.cropped = cv2.imread(self.image_path)
            pts1 = np.float32([[170, 106], [279, 198], [32, 322], [980, 125]])
            pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            self.blur_imageFile = cv2.warpPerspective(self.cropped, matrix, (500, 600))
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500, 500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif (text == "Affine Transformation"):
            self.cropped = cv2.imread(self.image_path)
            rows, cols, ch = self.cropped.shape
            pts1 = np.float32([[83, 90], [447, 90], [83, 472]])
            pts2 = np.float32([[0, 0], [447, 90], [150, 472]])
            matrix = cv2.getAffineTransform(pts1, pts2)
            self.blur_imageFile = cv2.warpAffine(self.cropped, matrix, (cols, rows))
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500, 500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        else:
            print("Please Select")
    def give_brightness(self):
        text = self.combo4.get()
        if(text == "++Brightness"):
            self.resized = cv2.imread(self.image_path)
            self.blur_imageFile = cv2.add(self.resized, np.array([50.0]))
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500, 500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif(text == "--Brightness"):
            self.resized = cv2.imread(self.image_path)
            self.blur_imageFile = cv2.add(self.resized, np.array([-50.0]))
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500, 500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        else:
            print("Please Select")

    secenek5 = ["Binary Erosion", "Binary Dilation", "Opening", "Closing", "Diamond",
                "Disk", "Cube", "Ball", "White TopHat", "Black TopHat"]
    def morphological_operations(self):
        text = self.combo5.get()
        if(text == "Binary Erosion"):
            self.resized = cv2.imread(self.image_path)
            kernel = np.ones((5, 5), np.uint8)
            self.blur_imageFile = cv2.erode(self.resized, kernel, iterations=1)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500, 500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif(text == "Binary Dilation"):
            self.resized = cv2.imread(self.image_path)
            kernel = np.ones((5, 5), np.uint8)
            self.blur_imageFile = cv2.dilate(self.resized, kernel, iterations=1)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500, 500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif (text == "Opening"):
            self.resized = cv2.imread(self.image_path)
            kernel = np.ones((5, 5), np.uint8)
            self.blur_imageFile = cv2.morphologyEx(self.resized, cv2.MORPH_OPEN, kernel)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500, 500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif (text == "Closing"):
            self.resized = cv2.imread(self.image_path)
            kernel = np.ones((5, 5), np.uint8)
            self.blur_imageFile = cv2.morphologyEx(self.resized, cv2.MORPH_CLOSE, kernel)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500, 500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif (text == "Morphological Gradient"):
            self.resized = cv2.imread(self.image_path)
            kernel = np.ones((5, 5), np.uint8)
            self.blur_imageFile = cv2.morphologyEx(self.resized, cv2.MORPH_GRADIENT, kernel)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500, 500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif (text == "Top Hat"):
            self.resized = cv2.imread(self.image_path)
            kernel = np.ones((5, 5), np.uint8)
            self.blur_imageFile = cv2.morphologyEx(self.resized, cv2.MORPH_TOPHAT, kernel)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500, 500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif (text == "Black Hat"):
            self.resized = cv2.imread(self.image_path)
            kernel = np.ones((5, 5), np.uint8)
            self.blur_imageFile = cv2.morphologyEx(self.resized, cv2.MORPH_BLACKHAT, kernel)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500, 500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif (text == "Morphological Rect"):
            self.resized = cv2.imread(self.image_path)
            kernel = np.ones((5, 5), np.uint8)
            self.blur_imageFile = cv2.morphologyEx(self.resized, cv2.MORPH_RECT, kernel)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500, 500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif (text == "Morphological Ellipse"):
            self.resized = cv2.imread(self.image_path)
            kernel = np.ones((5, 5), np.uint8)
            self.blur_imageFile = cv2.morphologyEx(self.resized, cv2.MORPH_ELLIPSE, kernel)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500, 500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        elif (text == "Morphological Cross"):
            self.resized = cv2.imread(self.image_path)
            kernel = np.ones((5, 5), np.uint8)
            self.blur_imageFile = cv2.morphologyEx(self.resized, cv2.MORPH_CROSS, kernel)
            image = PIL.Image.fromarray(self.blur_imageFile)
            image = image.resize((500, 500), Image.ANTIALIAS)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        else:
            print("Please Select")
    def OpenVideoProcesser(self):
        master2 = tkinter.Tk()
        master2.geometry("600x600")
        master2.title("Video Processor")
        label = tkinter.Label(master2, text="Please press \'q\' for quit and close  windows")
        label.pack()
        try:
            cap = cv2.VideoCapture(0)

            while (True):
                # Capture frame-by-frame
                ret, frame = cap.read()

                # Our operations on the frame come here
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # Display the resulting frame
                cv2.imshow('frame', gray)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
        except:
            ms.showerror('Error!', 'Please sure computer camera works')

        # When everything done, release the capture

        master2.mainloop()





App(tkinter.Tk(), "Tkinter and OpenCV")