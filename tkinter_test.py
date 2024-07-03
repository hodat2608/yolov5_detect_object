# import shutil
# import tkinter as tk
# from tkinter import ttk
# import threading

# def copyfile2folder():
#     filename1 = "C:/FH/New folder (3)/2024-05-21_18-45-10-523816_luu_xuat.jpg"
#     filepath1 = "C:/Users/CCSX009/Documents/yolov5/test_image/camera1"

#     filename2 = "C:/FH/New folder (3)/2024-05-21_15-50-51-289134_luu_xuat.jpg"
#     filepath2 = "C:/Users/CCSX009/Documents/yolov5/test_image/camera2"

#     shutil.copy(filename1, filepath1)
#     shutil.copy(filename2, filepath2)

# def on_button_click():
#     threading.Thread(target=copyfile2folder).start()

# def main():
#     window = tk.Tk()
#     window.title("File Copy GUI")
#     window.geometry('600x400')

#     frame = ttk.Frame(window, padding="20")
#     frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

#     label = ttk.Label(frame, text="Press the button to copy files to the folders", font=('Segoe UI', 14))
#     label.grid(row=0, column=0, padx=10, pady=20)

#     copy_button = ttk.Button(frame, text="Copy Files", command=on_button_click, width=20)
#     copy_button.grid(row=1, column=0, padx=10, pady=20)

#     window.mainloop()

# if __name__ == "__main__":
#     main()

model_names = {0: 'keo_truc', 1: 'keo_comi', 2: 'keo_it', 3: 'keo_lo', 4: '1', 5: '2', 6: '3', 7: '4'} 
# for i1, model_name in model_names.items():
#     print( model_name)  
model_name = model_names.get(5)
print(model_name)