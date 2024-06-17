import PySimpleGUI as sg
import os
import glob
import cv2
import time
import numpy as np
import torch

# def detect_ChatGPT():
#     img_folder = "C:/Users/CCSX009/Documents/yolov5/camera1/"
#     for image_path in glob.glob(img_folder + "*.jpg"):
#         t1 = time.time()
#         results = model1(image_path, 608, 0.15)
#         area_remove = []
#         table1 = results.pandas().xyxy[0]        
#         for item, (name_label, confidence) in enumerate(zip(table1['name'], table1['confidence'] * 100)):
#             for str_key, data_value in model1.names.items():
#                 if values[f'{data_value}_1'] and data_value == name_label and confidence < int(values[f'{data_value}_Conf_1']):
#                     table1.drop(item, axis=0, inplace=True)
#                     area_remove.append(item)
#                     break 
#         name = list(table1['name'])
#         print(name)   
#         show1 = np.squeeze(results.render(area_remove))
#         show1 = cv2.resize(show1, (1000,800), interpolation=cv2.INTER_AREA)
#         show1 = cv2.cvtColor(show1, cv2.COLOR_BGR2RGB)
#         imgbytes1 = cv2.imencode('.png', show1)[1].tobytes()
#         window['image1'].update(data=imgbytes1)
#         os.remove(image_path)
#         t2 = time.time() - t1
#         time_processing = str(int(t2*1000)) + 'ms'
#         window['time_cam1'].update(value=time_processing, text_color='black')

# def change_model():
#     global model1
#     model_file = sg.popup_get_file('Select Model File', file_types=(("Model files", "*.pt"),))
#     if model_file:
#         model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5','custom', path=model_file, source='local', force_reload=False)
#         sg.popup("Model changed successfully!")

# def make_window():
#     layout = [
#         [sg.Image(filename='', size=(1000,800), key='image1', background_color='black')],
#         [sg.Text(' ')],
#         [sg.Text('Time Processing : ', justification='center', font=('Helvetica',30), text_color='red', expand_y=True),sg.Text('0 ms', font=('Helvetica',40), key='time_cam1', expand_x=True)],
#         [sg.Text('Confidence', size=(12,1), font=('Helvetica',15), text_color='red'), sg.Slider(range=(1,100), default_value=30, orientation='h', size=(60,20), font=('Helvetica',11), key='conf_thres1'),],
#         [sg.Frame('',[
#                 [
#                     sg.Text(f'{model1.names[i1]}_1', size=(12,1), font=('Helvetica',15), text_color='yellow'), 
#                     sg.Checkbox('', size=(3,1), default=True, font=('Helvetica',15), key=f'{model1.names[i1]}_1'), 
#                     sg.Radio('', group_id=f'Cam1 {i1}', size=(3,1), default=False, font=('Helvetica',15), key=f'{model1.names[i1]}_OK_1'), 
#                     sg.Input('1', size=(2,1), font=('Helvetica',15), key=f'{model1.names[i1]}_Num_1', text_color='navy'), 
#                     sg.Text('', size=(2,1), font=('Helvetica',15), text_color='red'), 
#                     sg.Radio('', group_id=f'Cam1 {i1}', size=(2,1), default=False, font=('Helvetica',15), key=f'{model1.names[i1]}_NG_1'), 
#                     sg.Input('0', size=(7,1), font=('Helvetica',15), key=f'{model1.names[i1]}_Wn_1', text_color='navy'), 
#                     sg.Text('', size=(1,1), font=('Helvetica',15), text_color='red'), 
#                     sg.Input('1600', size=(7,1), font=('Helvetica',15), key=f'{model1.names[i1]}_Wx_1', text_color='navy'), 
#                     sg.Text('', size=(1,1), font=('Helvetica',15), text_color='red'), 
#                     sg.Input('0', size=(7,1), font=('Helvetica',15), key=f'{model1.names[i1]}_Hn_1', text_color='navy'), 
#                     sg.Text('', size=(1,1), font=('Helvetica',15), text_color='red'), 
#                     sg.Input('1200', size=(7,1), font=('Helvetica',15), key=f'{model1.names[i1]}_Hx_1', text_color='navy'), 
#                     sg.Text('', size=(2,1), font=('Helvetica',15), text_color='red'), 
#                     sg.Input('0', size=(7,1), font=('Helvetica',15), key=f'{model1.names[i1]}_PLC_1', text_color='navy'),
#                     sg.Slider(range=(1,100), default_value=30, orientation='h', size=(28,9), font=('Helvetica',10), key=f'{model1.names[i1]}_Conf_1'),
#                 ] for i1 in range(len(model1.names))
#             ], relief=sg.RELIEF_FLAT)],
#         [sg.Button('Open Image'), sg.Button('Change Model'), sg.Button('Exit')]
#     ]
#     layout_option1 = [[sg.Column(layout, scrollable=True)]]
#     layout = [[sg.TabGroup([[sg.Tab('Main', layout_option1)]])]]
#     window = sg.Window('afbb', layout, location=(0,0),resizable=True).Finalize()
#     window.Maximize()
#     return window

# model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5','custom', path="V:/test/best.pt", source='local', force_reload=False)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# window = make_window()
# while True:
#     event, values = window.read(timeout=20)
#     if event == sg.WINDOW_CLOSED or event == 'Exit':
#         break
#     elif event == 'Open Image':
#         # Code to handle opening image
#         pass
#     elif event == 'Change Model':
#         change_model()
#         window = make_window()
#     else:
#         detect_ChatGPT()

# window.close()


import PySimpleGUI as sg
import torch
import cv2
import os
import glob
import time

def detect_ChatGPT(image_path):
    t1 = time.time()
    # print(model1)
    results = model1(image_path, 608, 0.15)
    area_remove = []
    table1 = results.pandas().xyxy[0]  
    for item in range(len(table1.index)):
        conf1 = table1['confidence'][item] * 100
        label_name = table1['name'][item]
        for i1 in range(len(model1.names)):
            if values[f'{model1.names[i1]}_1'] == True:
                    if label_name == model1.names[i1]:
                        if conf1 < int(values[f'{model1.names[i1]}_Conf_1']):
                            table1.drop(item, axis=0, inplace=True)
                            area_remove.append(item)
            else:         
                if label_name == model1.names[i1]:
                    table1.drop(item, axis=0, inplace=True)
                    area_remove.append(item)

    name = list(table1['name'])
    show1 = np.squeeze(results.render(area_remove))
    show1 = cv2.resize(show1, (1000,800), interpolation=cv2.INTER_AREA)
    show1 = cv2.cvtColor(show1, cv2.COLOR_BGR2RGB)
    imgbytes1 = cv2.imencode('.png', show1)[1].tobytes()
    window['image1'].update(data=imgbytes1)
    t2 = time.time() - t1
    time_processing = str(int(t2*1000)) + 'ms'
    window['time_cam1'].update(value=time_processing, text_color='black')

def change_model():
    global model1
    model_file = sg.popup_get_file('Select Model File', file_types=(("Model files", "*.pt"),))
    if model_file:
        model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5','custom', path=model_file, source='local', force_reload=False)
        sg.popup("Model changed successfully!")
    return model_file

def open_image_folder():
    global images
    folder = sg.popup_get_folder('Select Image Folder')
    if folder:
        images = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        return images
    return None

def detect_images_in_folder(images, current_index):
    if images:
        current_index %= len(images)
        print(current_index)
        print(images[current_index])
        detect_ChatGPT(images[current_index])
        return current_index
    return None

def make_window():
    layout = [
        [sg.Image(filename='', size=(1416,846), key='image1', background_color='black')],
        [sg.Text(' ')],
        [sg.Button('Open Image'), sg.Button('Change Model'), sg.Button('Prev'), sg.Button('Next'), sg.Button('Exit')],
        [sg.Text('', size=(50,1), font=('Helvetica',15), text_color='yellow',key='model'), ],
        [sg.Text('Time Processing : ', justification='center', font=('Helvetica',30), text_color='red', expand_y=True),sg.Text('0 ms', font=('Helvetica',40), key='time_cam1', expand_x=True)],
        [sg.Text('Confidence', size=(12,1), font=('Helvetica',15), text_color='red'), sg.Slider(range=(1,100), default_value=30, orientation='h', size=(60,20), font=('Helvetica',11), key='conf_thres1'),],
        [sg.Frame('',[
                [
                    sg.Text(f'{model1.names[i1]}_1', size=(12,1), font=('Helvetica',15), text_color='yellow'), 
                    sg.Checkbox('', size=(3,1), default=True, font=('Helvetica',15), key=f'{model1.names[i1]}_1'), 
                    sg.Radio('', group_id=f'Cam1 {i1}', size=(3,1), default=False, font=('Helvetica',15), key=f'{model1.names[i1]}_OK_1'), 
                    sg.Input('1', size=(2,1), font=('Helvetica',15), key=f'{model1.names[i1]}_Num_1', text_color='navy'), 
                    sg.Text('', size=(2,1), font=('Helvetica',15), text_color='red'), 
                    sg.Radio('', group_id=f'Cam1 {i1}', size=(2,1), default=False, font=('Helvetica',15), key=f'{model1.names[i1]}_NG_1'), 
                    sg.Input('0', size=(7,1), font=('Helvetica',15), key=f'{model1.names[i1]}_Wn_1', text_color='navy'), 
                    sg.Text('', size=(1,1), font=('Helvetica',15), text_color='red'), 
                    sg.Input('1600', size=(7,1), font=('Helvetica',15), key=f'{model1.names[i1]}_Wx_1', text_color='navy'), 
                    sg.Text('', size=(1,1), font=('Helvetica',15), text_color='red'), 
                    sg.Input('0', size=(7,1), font=('Helvetica',15), key=f'{model1.names[i1]}_Hn_1', text_color='navy'), 
                    sg.Text('', size=(1,1), font=('Helvetica',15), text_color='red'), 
                    sg.Input('1200', size=(7,1), font=('Helvetica',15), key=f'{model1.names[i1]}_Hx_1', text_color='navy'), 
                    sg.Text('', size=(2,1), font=('Helvetica',15), text_color='red'), 
                    sg.Input('0', size=(7,1), font=('Helvetica',15), key=f'{model1.names[i1]}_PLC_1', text_color='navy'),
                    sg.Slider(range=(1,100), default_value=30, orientation='h', size=(28,9), font=('Helvetica',10), key=f'{model1.names[i1]}_Conf_1'),
                ] for i1 in range(len(model1.names))
            ], relief=sg.RELIEF_FLAT)],
       
    ]
    layout_option1 = [[sg.Column(layout, size=(2000,1000), scrollable = True, vertical_scroll_only=True)]]
    layout = [[sg.TabGroup([[sg.Tab('Main', layout_option1)]])]]
    window = sg.Window('afbb', layout, location=(0,0),resizable=True).Finalize()
    window.Maximize()
    return window

model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5','custom', path="C:/X75_M100_DEN_CAM1_2024-03-11.pt", source='local', force_reload=False)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
window = make_window()
current_index = 0
while True:
    event, values = window.read(timeout=20)
    if event == sg.WINDOW_CLOSED or event == 'Exit':
        break
    elif event == 'Open Image':
        images = open_image_folder()
        if images:
            current_index = detect_images_in_folder(images, current_index)
    elif event == 'Change Model':
        file_model = change_model()
        window = make_window()
        window['model'].update(file_model)
    elif event == 'Prev':
        current_index -= 1
        current_index = detect_images_in_folder(images, current_index)
    elif event == 'Next':
        current_index += 1
        current_index = detect_images_in_folder(images, current_index)
    # else:
    #     detect_ChatGPT(image_path)

window.close()
