# import tkinter
# from tkinter import ttk
# import torch
# import numpy as np
# import cv2,os
# import matplotlib.pyplot as plt
# from IPython.display import display, Image
# import PySimpleGUI as sg
# from PIL import Image
# import glob
# import time

# def print_widget_values():
#     for i1 in range(len(model1.names)):
#         print("Values for", model1.names[i1])
#         print("Model Name Label Text:", model_name_labels[i1].cget("text"))
#         print("Model Checkbox State:", model_checkboxes[i1].instate(["selected"]))
#         print("OK Radio Value:", ok_radios[i1].cget("value"))
#         print("Num Input Value:", num_inputs[i1].get())
#         print("NG Radio Value:", ng_radios[i1].cget("value"))
#         print("WN Input Value:", wn_inputs[i1].get())
#         print("WX Input Value:", wx_inputs[i1].get())
#         print("HN Input Value:", hn_inputs[i1].get())
#         print("HX Input Value:", hx_inputs[i1].get())
#         print("PLC Input Value:", plc_inputs[i1].get())
#         print("Conf Scale Value:", conf_scales[i1].get())
#         print()

# def open_image():
#     print_widget_values()

# root = tkinter.Tk()
# root.title('HuynhLeVu')

# main_frame = ttk.Frame(root)
# main_frame.pack(fill='both', expand=True)

# user_info_frame = tkinter.LabelFrame(main_frame, text="User Information")
# user_info_frame.pack(side="left", padx=20, pady=10) 


# image_label = ttk.Label(user_info_frame, image='', background='black')
# image_label.pack(pady=10)


# time_label = ttk.Label(user_info_frame, text='Time Processing : 0 ms', foreground='black')
# time_label.pack(pady=10)


# confidence_label = ttk.Label(user_info_frame, text='Confidence',  foreground='black')
# confidence_label.pack(pady=5)
# confidence_scale = ttk.Scale(user_info_frame, from_=1, to=100, length=200)
# confidence_scale.set(30)
# confidence_scale.pack(pady=5)

# model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5','custom', path="C:/Users/CCSX009/Documents/yolov5/file_train/X75_M100_DEN_CAM1_2024-03-11.pt", source='local',force_reload =False)
# wx_inputs = []
# model_name_labels = []
# model_checkboxes = []
# ok_radios = []
# num_inputs = []
# ng_radios = []
# wn_inputs = []
# wx_inputs = []
# hn_inputs = []
# hx_inputs = []
# plc_inputs = []
# conf_scales = []

# for i1 in range(len(model1.names)):
#     model_frame = ttk.Frame(user_info_frame)
#     model_frame.pack(pady=5)
    
#     model_name_label = ttk.Label(model_frame, text=f'{model1.names[i1]}_1', font=('Helvetica', 15), foreground='black')
#     model_name_label.grid(row=0, column=0)
#     model_name_labels.append(model_name_label)
    
#     model_checkbox = ttk.Checkbutton(model_frame)
#     model_checkbox.grid(row=0, column=1)
#     model_checkboxes.append(model_checkbox)
    
#     ok_radio = ttk.Radiobutton(model_frame, value='OK')
#     ok_radio.grid(row=0, column=2)
#     ok_radios.append(ok_radio)
    
#     num_input = ttk.Entry(model_frame, width=5)
#     num_input.insert(0, '1')
#     num_input.grid(row=0, column=3)
#     num_inputs.append(num_input)
    
#     ng_radio = ttk.Radiobutton(model_frame, value='NG')
#     ng_radio.grid(row=0, column=4)
#     ng_radios.append(ng_radio)
    
#     wn_input = ttk.Entry(model_frame, width=7)
#     wn_input.insert(0, '0')
#     wn_input.grid(row=0, column=5)
#     wn_inputs.append(wn_input)
    
#     wx_input = ttk.Entry(model_frame, width=7)
#     wx_input.insert(0, '1600')
#     wx_input.grid(row=0, column=6)
#     wx_inputs.append(wx_input)
    
#     hn_input = ttk.Entry(model_frame, width=7)
#     hn_input.insert(0, '0')
#     hn_input.grid(row=0, column=7)
#     hn_inputs.append(hn_input)
    
#     hx_input = ttk.Entry(model_frame, width=7)
#     hx_input.insert(0, '1200')
#     hx_input.grid(row=0, column=8)
#     hx_inputs.append(hx_input)
    
#     plc_input = ttk.Entry(model_frame, width=7)
#     plc_input.insert(0, '0')
#     plc_input.grid(row=0, column=9)
#     plc_inputs.append(plc_input)
    
#     conf_scale = ttk.Scale(model_frame, from_=1, to=100, length=200)
#     conf_scale.grid(row=0, column=10)
#     conf_scales.append(conf_scale)


# open_image_button = ttk.Button(main_frame, text='Open Image', command=open_image)
# open_image_button.pack(pady=10)

# exit_button = ttk.Button(main_frame, text='Exit', command=root.quit)
# exit_button.pack(pady=10)

# while True:
#     root.mainloop()

import tkinter as tk

# Danh sách tên và thông số tương ứng
name_info = {
    "johns": {"age": 25, "address": "123 Main St", "phone": "123-456-7890", "gender": "Male", "school": "ABC School"},
    "mora": {"age": 30, "address": "456 Elm St", "phone": "234-567-8901", "gender": "Female", "school": "XYZ School"},
    "liqid": {"age": 22, "address": "789 Oak St", "phone": "345-678-9012", "gender": "Male", "school": "DEF School"},
    "scoter": {"age": 28, "address": "321 Pine St", "phone": "456-789-0123", "gender": "Male", "school": "GHI School"},
    "madison": {"age": 26, "address": "543 Cedar St", "phone": "567-890-1234", "gender": "Female", "school": "JKL School"},
    "messi": {"age": 35, "address": "876 Birch St", "phone": "678-901-2345", "gender": "Male", "school": "MNO School"}
}

# Hàm hiển thị thông tin khi chọn một tên từ danh sách
def show_info():
    selected_name = name_var.get()
    info = name_info.get(selected_name, {})

    # Xóa thông tin cũ trước khi hiển thị thông tin mới
    for widget in info_frame.winfo_children():
        widget.destroy()

    # Hiển thị thông tin mới
    for i, (key, value) in enumerate(info.items()):
        label = tk.Label(info_frame, text=f"{key.capitalize()}: {value}")
        label.grid(row=i, column=0, sticky="w")

# Tạo giao diện đồ họa
root = tk.Tk()
root.title("Thông tin cá nhân")

# Tạo danh sách tên
names = list(name_info.keys())

# Tạo label và dropdown menu để chọn tên
name_label = tk.Label(root, text="Chọn tên:")
name_label.pack()

name_var = tk.StringVar(root)
name_var.set(names[0])  # Mặc định chọn tên đầu tiên
name_menu = tk.OptionMenu(root, name_var, *names, command=show_info)
name_menu.pack()

# Tạo frame để hiển thị thông tin
info_frame = tk.Frame(root)
info_frame.pack()

# Hiển thị thông tin ban đầu cho tên đầu tiên
show_info()

root.mainloop()
