import PySimpleGUI as sg
from collections import Counter
import glob
import torch

import os
import random
import shutil

import datetime 
import cv2

import subprocess
import traceback
from os.path import getsize

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def time_to_name():
    current_time = datetime.datetime.now() 
    name_folder = str(current_time)
    name_folder = list(name_folder)
    for i in range(len(name_folder)):
        if name_folder[i] == ':':
            name_folder[i] = '-'
        if name_folder[i] == ' ':
            name_folder[i] ='_'
        if name_folder[i] == '.':
            name_folder[i] ='-'
    name_folder = ''.join(name_folder)
    return name_folder

def test():
    import threading
    layout = [[sg.Text('Copying'), sg.Text('',key='percent')],
              [sg.ProgressBar(max_value=100, orientation='h', size=(20, 20), key='progress_1')]]

    popup_window = sg.Window('Pprogress', layout, finalize=True)
    current_value = 1
    popup_window['progress_1'].update(current_value)
    stop_threads = False
    threading.Thread(target=another_function, args=(popup_window, ), daemon=True).start()
    
    while True:
        window, event, values = sg.read_all_windows()
        if event == 'Exit':
            break
        if event.startswith('update_'):
            #print(f'event: {event}, value: {values[event]}')
            key_to_update = event[len('update_'):]
            window[key_to_update].update(values[event])
            window.refresh()
            continue
        # process any other events ...
    window.close()

def another_function(cuaso):
    tatca = glob.glob(values['input_image4a'] + '/**/*.*', recursive=True)
    i=ht=0
    if len(tatca) > 0:
        #window['progress1'].update(visible=True)
        for f in tatca:
            tenf = os.path.basename(f)
            print(tenf)
            shutil.copyfile(f,values['directory_save4a'] + '/' + tenf)
            i += 1
            if ht != int(100 * (i/len(tatca))):
                ht = int(100 * (i/len(tatca)))
                cuaso.write_event_value('update_progress_1', ht)
                cuaso.write_event_value('update_percent', str(ht) + '%')

    cuaso.write_event_value('Exit', '')

theme_dict = {'BACKGROUND': '#2B475D',
                'TEXT': '#FFFFFF',
                'INPUT': '#F2EFE8',
                'TEXT_INPUT': '#000000',
                'SCROLL': '#F2EFE8',
                'BUTTON': ('#000000', '#C2D4D8'),
                'PROGRESS': ('#FFFFFF', '#C7D5E0'),
                'BORDER': 1,'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}

# sg.theme_add_new('Dashboard', theme_dict)     # if using 4.20.0.1+
sg.LOOK_AND_FEEL_TABLE['Dashboard'] = theme_dict
sg.theme('Dashboard')

BORDER_COLOR = '#C7D5E0'
#BORDER_COLOR = '#fcfffd'
DARK_HEADER_COLOR = '#1B2838'
MYBPAD = ((20,10), (10, 10))


# mypath1 = 'best.pt'
# model1 = torch.hub.load('./levu','custom', path= mypath1, source='local',force_reload =False)

cls_names = []
f=open('classes.txt','r')
while True:
    line = f.readline()
    if not line:
        break
    tmp = line.split()
    cls_names.append(tmp[0]) 

file_weights = [('Weights (*.pt)', ('*.pt'))]

def make_window():

    Step_1 =[
                [sg.Text('Step 1: Get image in folder', font='Any 20', text_color='yellow')],
                [sg.T('1.Choose folder get image             ', font='Any 15', text_color = 'orange')],
                [sg.Input(size=(int(0.7*30),1), font=('Helvetica',12), key='input_image0',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(int(0.7*12),1), font=('Helvetica',10),key= 'directory_image0',enable_events=True) ],
                [sg.T('2.Choose folder save image             ', font='Any 15', text_color = 'orange')],
                [sg.Input(size=(int(0.7*30),1), font=('Helvetica',12), key='input_save0',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(int(0.7*12),1), font=('Helvetica',10),key= 'directory_save0',enable_events=True) ],
                [sg.T('3.Enter name image you want get             ', font='Any 15', text_color = 'orange')],
                [sg.Multiline('',size=(int(0.7*40),10),text_color='navy' ,key='input_names0')], 
                [sg.Button('OK', size=(int(0.7*12),1), font=('Helvetica',10),key= 'button_names0')],
                [sg.T('4.Start get image           ', font='Any 15', text_color = 'orange')],
                [sg.Button('Start', size=(int(0.7*12),1), font=('Helvetica',10),key= 'start0')],
    ]


    Step_2 = [
                #[sg.Frame('',[
                [sg.Frame('',
                [   
                    [sg.Text('Step 2: Auto filter image', font='Any 20', text_color='yellow')],
                    [sg.T('1.Choose file weights         ', font='Any 15', text_color = 'orange'),
                    sg.Input(size=(int(0.7*70),1), font=('Helvetica',12), key='input_weight1',readonly= True, text_color='navy',enable_events= True),
                    sg.FileBrowse(file_types= file_weights, size=(int(0.7*12),1), font=('Helvetica',10),key= 'file_browse1',enable_events=True)],
                    [sg.T('2.Choose confident             ', font='Any 15', text_color = 'orange'),
                    sg.Slider(range=(1,100),default_value=25,orientation='h',size=(int(0.7*70),20),font=('Helvetica',11),key= 'input_conf1'),]
                ], relief=sg.RELIEF_FLAT),
                ],
                [sg.T(' 3.Choose Parameter', font='Any 15', text_color = 'orange')],
                [sg.Frame('',[
                    [sg.Text('Name',size=(int(0.7*15),1),font=('Helvetica',15), text_color='orange'), 
                    sg.Text('Join',size=(int(0.7*7),1),font=('Helvetica',15), text_color='orange'), 
                    sg.Text('OK',size=(int(0.7*7),1),font=('Helvetica',15), text_color='orange'), 
                    sg.Text('Num',size=(int(0.7*7),1),font=('Helvetica',15), text_color='orange'), 
                    sg.Text('NG',size=(int(0.7*8),1),font=('Helvetica',15), text_color='orange'),  
                    sg.Text('Wid Min',size=(int(0.7*11),1),font=('Helvetica',15), text_color='orange'), 
                    sg.Text('Wid Max',size=(int(0.7*12),1),font=('Helvetica',15), text_color='orange'), 
                    sg.Text('Hei Min',size=(int(0.7*11),1),font=('Helvetica',15), text_color='orange'),
                    sg.Text('Hei Max',size=(int(0.7*12),1),font=('Helvetica',15), text_color='orange'), 
                    sg.Text('Conf',size=(int(0.7*9),1),font=('Helvetica',15), text_color='orange')],
                ], relief=sg.RELIEF_FLAT)],
                [sg.Frame('',[
                    [
                        sg.Text(f'{model1.names[i1]}_1',size=(int(0.7*15),1),font=('Helvetica',15), text_color='pink'), 
                        sg.Checkbox('',size=(int(0.7*3),5),default=True,font=('Helvetica',15),  key=f'{model1.names[i1]}_1',enable_events=True), 
                        sg.Checkbox('',size=(int(0.7*3),5),font=('Helvetica',15),  key=f'{model1.names[i1]}_OK_1',enable_events=True), 
                        sg.Input('1',size=(int(0.7*2),1),font=('Helvetica',15),key= f'{model1.names[i1]}_Num_1',text_color='navy',enable_events=True), 
                        sg.Text('',size=(int(0.7*3),1),font=('Helvetica',15), text_color='red'), 
                        sg.Checkbox('',size=(int(0.7*5),5),font=('Helvetica',15),  key=f'{model1.names[i1]}_NG_1',enable_events=True), 
                        sg.Input('0',size=(int(0.7*8),1),font=('Helvetica',15),key= f'{model1.names[i1]}_Wn_1',text_color='navy',enable_events=True), 
                        sg.Text('',size=(int(0.7*2),1),font=('Helvetica',15), text_color='red'), 
                        sg.Input('1600',size=(int(0.7*8),1),font=('Helvetica',15),key= f'{model1.names[i1]}_Wx_1',text_color='navy',enable_events=True), 
                        sg.Text('',size=(int(0.7*2),1),font=('Helvetica',15), text_color='red'), 
                        sg.Input('0',size=(int(0.7*8),1),font=('Helvetica',15),key= f'{model1.names[i1]}_Hn_1',text_color='navy',enable_events=True), 
                        sg.Text('',size=(int(0.7*2),1),font=('Helvetica',15), text_color='red'), 
                        sg.Input('1200',size=(int(0.7*8),1),font=('Helvetica',15),key= f'{model1.names[i1]}_Hx_1',text_color='navy',enable_events=True),
                        sg.Slider(range=(1,100),default_value=70,orientation='h',size=(int(0.7*20),10),font=('Helvetica',11),key= f'{model1.names[i1]}_conf_1'), 
                    ] for i1 in range(len(model1.names))
                ], relief=sg.RELIEF_FLAT)],
                [sg.T(' 4.Choose folder image         ', font='Any 15', text_color = 'orange'),
                sg.Input(size=(int(0.7*70),1), font=('Helvetica',12), key='input_image1',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(int(0.7*12),1), font=('Helvetica',10),key= 'directory_image1',enable_events=True) ],
                [sg.T(' 5.Choose folder save           ', font='Any 15', text_color = 'orange'),
                sg.Input(size=(int(0.7*70),1), font=('Helvetica',12), key='input_save1',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(int(0.7*12),1), font=('Helvetica',10),key= 'directory_save1',enable_events=True) ],
                [sg.T(' 6.Start auto filter', font='Any 15', text_color = 'orange'),
                sg.Radio('Move',group_id='CHK1',size=(5,5),default=True,font=('Helvetica',15),  key='CHK_MOVE',enable_events=True), 
                sg.Radio('Copy',group_id='CHK1',size=(5,5),font=('Helvetica',15),  key='CHK_COPY',enable_events=True),
                 sg.Combo((416,608),default_value=416,key='sz2b'), 
                sg.Button('Start', size=(int(0.7*12),1), font=('Helvetica',10),key= 'start1')],
                ]

    Step_1n =[
                [sg.Text('Step 1 new: Get image', font='Any 20', text_color='yellow')],
               
    ]
   
    Step_2n = [
                [sg.Frame('',
                [   
                    [sg.Text('Step 2 new: Auto Label', font='Any 20', text_color='yellow')],
                    [sg.T('1.Choose file weights         ', font='Any 15', text_color = 'orange'),
                    sg.Input(size=(int(0.7*70),1), font=('Helvetica',12), key='input_weight2n',readonly= True, text_color='navy',enable_events= True),
                    sg.FileBrowse(file_types= file_weights, size=(int(0.7*12),1), font=('Helvetica',10),key= 'file_browse2n',enable_events=True)],
                    [sg.T('2.Choose confident             ', font='Any 15', text_color = 'orange'),
                    sg.Slider(range=(1,100),default_value=25,orientation='h',size=(int(0.7*70),20),font=('Helvetica',11),key= 'input_conf2n'),]
                ], relief=sg.RELIEF_FLAT),
                ],
                [sg.T(' 3.Choose Parameter', font='Any 15', text_color = 'orange')],
                [sg.Frame('',[
                    [sg.Text('Name',size=(int(0.7*20),1),font=('Helvetica',15), text_color='orange'), 
                    sg.Text('Join',size=(int(0.7*7),1),font=('Helvetica',15), text_color='orange'), 
                    sg.Text('Wid Min',size=(int(0.7*11),1),font=('Helvetica',15), text_color='orange'), 
                    sg.Text('Wid Max',size=(int(0.7*12),1),font=('Helvetica',15), text_color='orange'), 
                    sg.Text('Hei Min',size=(int(0.7*11),1),font=('Helvetica',15), text_color='orange'),
                    sg.Text('Hei Max',size=(int(0.7*12),1),font=('Helvetica',15), text_color='orange'), 
                    sg.Text('Conf',size=(int(0.7*9),1),font=('Helvetica',15), text_color='orange')],
                ], relief=sg.RELIEF_FLAT)],
                [sg.Frame('',[
                    [
                        sg.Text(f'{model2.names[i3]}_3',size=(int(0.7*20),1),font=('Helvetica',15), text_color='pink'), 
                        sg.Checkbox('',size=(int(0.7*3),5),default=True,font=('Helvetica',15),  key=f'{model2.names[i3]}_3',enable_events=True), 
                        sg.Input('0',size=(int(0.7*8),1),font=('Helvetica',15),key= f'{model2.names[i3]}_Wn_3',text_color='navy',enable_events=True), 
                        sg.Text('',size=(int(0.7*2),1),font=('Helvetica',15), text_color='red'), 
                        sg.Input('1600',size=(int(0.7*8),1),font=('Helvetica',15),key= f'{model2.names[i3]}_Wx_3',text_color='navy',enable_events=True), 
                        sg.Text('',size=(int(0.7*2),1),font=('Helvetica',15), text_color='red'), 
                        sg.Input('0',size=(int(0.7*8),1),font=('Helvetica',15),key= f'{model2.names[i3]}_Hn_3',text_color='navy',enable_events=True), 
                        sg.Text('',size=(int(0.7*2),1),font=('Helvetica',15), text_color='red'), 
                        sg.Input('1200',size=(int(0.7*8),1),font=('Helvetica',15),key= f'{model2.names[i3]}_Hx_3',text_color='navy',enable_events=True),
                        sg.Slider(range=(1,100),default_value=75,orientation='h',size=(int(0.7*40),10),font=('Helvetica',11),key= f'{model2.names[i3]}_conf_3'), 
                    ] for i3 in range(len(model2.names))
                ], relief=sg.RELIEF_FLAT)],
                [sg.T(' 4.Choose folder images          ', font='Any 15', text_color = 'orange'),
                sg.Input(size=(int(0.7*70),1), font=('Helvetica',12), key='input_image2n',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(int(0.7*12),1), font=('Helvetica',10),key= 'directory_image2n',enable_events=True) ],
                [sg.T(' 5.Choose folder save labels    ', font='Any 15', text_color = 'orange'),
                sg.Input(size=(int(0.7*70),1), font=('Helvetica',12), key='input_save2n',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(int(0.7*12),1), font=('Helvetica',10),key= 'directory_save2n',enable_events=True) ],
                [sg.T(' 6.Start create auto label with size', font='Any 15', text_color = 'orange'),
                sg.Combo((416,608),default_value=416,key='sz2n'),
                sg.Button('Start', size=(int(0.7*12),1), font=('Helvetica',10),key= 'start2n')],
                ]

    Step_3 = [
                [sg.Text('Step 3: Auto create labels', font='Any 20', text_color='yellow')],
                [sg.T('1.Choose file weights', font='Any 15', text_color = 'orange')],
                [sg.Input(size=(int(0.7*35),1), font=('Helvetica',12), key='input_weight2',readonly= True, text_color='navy',enable_events= True),
                sg.FileBrowse(file_types= file_weights,size=(int(0.7*12),1), font=('Helvetica',10),key= 'directory_weight2',enable_events=True) ],
                [sg.T('2.Choose confident', font='Any 15', text_color = 'orange')],
                [sg.Slider(range=(1,100),orientation='h',size=(int(0.7*48),20),font=('Helvetica',11),default_value=25 ,key= 'input_conf2')],
                [sg.T('3.Choose folder image', font='Any 15', text_color = 'orange')],
                [sg.Input(size=(int(0.7*35),1), font=('Helvetica',12), key='input_image2',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(int(0.7*12),1), font=('Helvetica',10),key= 'directory_image2',enable_events=True) ],
                [sg.T('4.Choose folder save', font='Any 15', text_color = 'orange')],
                [sg.Input(size=(int(0.7*35),1), font=('Helvetica',12), key='input_save2',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(int(0.7*12),1), font=('Helvetica',10),key= 'directory_model2',enable_events=True) ],
                [sg.T('5.Enter all name labels', font='Any 15', text_color = 'orange')],
                [sg.Multiline('',size=(int(0.7*40),int(0.7*10)),text_color='navy' ,key='input_classes2')],
                [sg.Button('OK', size=(int(0.7*12),1), font=('Helvetica',10),key= 'button_classes2')],
                [sg.T('6.Start create auto label', font='Any 15', text_color = 'orange')],
                [sg.Combo((416,608),default_value=416,key='sz2'),sg.Button('Start', size=(int(0.7*12),1), font=('Helvetica',10),key= 'start2')],
                
            ]

    Step_3a = [
                [sg.Text('Step 3a: Append labels', font='Any 20', text_color='yellow')],
                [sg.T('1.Choose file weights', font='Any 15', text_color = 'orange')],
                [sg.Input(size=(int(0.7*35),1), font=('Helvetica',12), key='input_weight2a',readonly= True, text_color='navy',enable_events= True),
                sg.FileBrowse(file_types= file_weights,size=(int(0.7*12),1), font=('Helvetica',10),key= 'directory_weight2a',enable_events=True) ],
                [sg.T('2.Choose confident', font='Any 15', text_color = 'orange')],
                [sg.Slider(range=(1,100),orientation='h',size=(int(0.7*48),20),font=('Helvetica',11),default_value=25 ,key= 'input_conf2a')],
                [sg.T('3.Choose folder image', font='Any 15', text_color = 'orange')],
                [sg.Input(size=(int(0.7*35),1), font=('Helvetica',12), key='input_image2a',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(int(0.7*12),1), font=('Helvetica',10),key= 'directory_image2a',enable_events=True) ],
                # [sg.T('4.Choose folder model', font='Any 15', text_color = 'orange')],
                # [sg.Input(size=(int(0.7*35),1), font=('Helvetica',12), key='input_model1',readonly= True, text_color='navy',enable_events= True),
                # sg.FolderBrowse(size=(int(0.7*12),1), font=('Helvetica',10),key= 'directory_model2',enable_events=True) ],
                [sg.T('4.Choose folder save', font='Any 15', text_color = 'orange')],
                [sg.Input(size=(int(0.7*35),1), font=('Helvetica',12), key='input_save2a',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(int(0.7*12),1), font=('Helvetica',10),key= 'directory_model2a',enable_events=True) ],
                [sg.T('5.Enter all name labels', font='Any 15', text_color = 'orange')],
                [sg.Multiline('',size=(int(0.7*40),int(0.7*10)),text_color='navy' ,key='input_classes2a')],
                [sg.Button('OK', size=(int(0.7*12),1), font=('Helvetica',10),key= 'button_classes2a')],
                #[sg.Listbox(values=CLASSES1,size=(int(0.7*2)3,4), text_color= 'navy',select_mode= sg.LISTBOX_SELECT_MODE_MULTIPLE, key='classes1')],
                [sg.T('6.Append label start class number', font='Any 15', text_color = 'orange')],
                [sg.Input('5',size=(5,2),key='sz2a'),sg.Button('Start', size=(int(0.7*12),1), font=('Helvetica',10),key= 'start2a')],
                
            ]
    
    Step_4 = [
                [sg.Text('Step 4: Labeling with LabelImg', font='Any 20', text_color='yellow')],
                [sg.T('1.Enter all name labels', font='Any 15', text_color = 'orange')],
                [sg.Multiline('',size=(int(0.7*40),int(0.7*10)),text_color='navy' ,key='input_classes3')],
                [sg.Button('OK', size=(int(0.7*12),1), font=('Helvetica',10),key= 'button_classes3')],     
                [sg.T('2.Open LabelImg', font='Any 15', text_color = 'orange')],
                [sg.Button('Open', size=(int(0.7*12),1), font=('Helvetica',10),key= 'program3')],       
            ]

                
    Step_5 = [
                [sg.Text('Step 5: Auto split image and label', font='Any 20', text_color='yellow')],
                [sg.T('1.Choose folder image', font='Any 15', text_color = 'orange')],
                [sg.Input(size=(int(0.7*35),1), font=('Helvetica',12), key='input_image4',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(int(0.7*12),1), font=('Helvetica',10),key= 'directory_image4',enable_events=True) ],
                [sg.T('2.Choose folder save', font='Any 15', text_color = 'orange')],
                [sg.Input(size=(int(0.7*35),1), font=('Helvetica',12), key='input_save4',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(int(0.7*12),1), font=('Helvetica',10),key= 'directory_save4',enable_events=True) ],
                #[sg.Listbox(values=CLASSES1,size=(int(0.7*2)3,4), text_color= 'navy',select_mode= sg.LISTBOX_SELECT_MODE_MULTIPLE, key='classes1')],
                [sg.T('3.Start split image and label', font='Any 15', text_color = 'orange')],
                [sg.Button('Start', size=(int(0.7*12),1), font=('Helvetica',10),key= 'start4')],
            ]
    
    Step_5a = [
                [sg.Text('Step 5a: Unsplit image and label', font='Any 20', text_color='yellow')],
                [sg.T('1.Choose folder contain train and valid', font='Any 15', text_color = 'orange')],
                [sg.Input(size=(int(0.7*50),1), font=('Helvetica',12), key='input_image4a',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(int(0.7*10),1), font=('Helvetica',10),key= 'directory_image4a',enable_events=True) ],
                [sg.T('2.Choose folder save Data', font='Any 15', text_color = 'orange')],
                [sg.Input(size=(int(0.7*50),1), font=('Helvetica',12), key='input_save4a',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(int(0.7*10),1), font=('Helvetica',10),key= 'directory_save4a',enable_events=True) ],
                #[sg.Listbox(values=CLASSES1,size=(int(0.7*2)3,4), text_color= 'navy',select_mode= sg.LISTBOX_SELECT_MODE_MULTIPLE, key='classes1')],
                [sg.T('3.Start Unsplit image and label', font='Any 15', text_color = 'orange')],
                [sg.Button('Start', size=(int(0.7*12),1), font=('Helvetica',10),key= 'start4a')]
                #[sg.ProgressBar(100, visible = False, style='winnative',orientation='h', size=(38,20), key='progress1')]
                #[sg.Slider(range=(1,100),default_value=0,orientation='h',size=(45,20),font=('Helvetica',11),enable_events= True,disabled=True,key= 'speed5a')]
            ]

    Step_5b = [
                [sg.Text('Step 5b: Filter label', font='Any 20', text_color='yellow')],
                [sg.T('1.Choose folder contain image and label', font='Any 15', text_color = 'orange')],
                [sg.Input(size=(int(0.7*50),1), font=('Helvetica',12), key='input_image4b',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(int(0.7*10),1), font=('Helvetica',10),enable_events=True) ],
                [sg.T('2.Image and Label copy to Folder', font='Any 15', text_color = 'orange')],
                [sg.Input(size=(int(0.7*50),1), font=('Helvetica',12), key='input_save4b',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(int(0.7*10),1), font=('Helvetica',10),key= 'directory_save4b',enable_events=True) ],
                [sg.T('3.Select class number (label)', font='Any 15', text_color = 'orange'),sg.Combo((0,1,2,3,4,5,6,7,8,9),default_value=0,key='cls5'),],
                [sg.Radio('Move',group_id='CHK2',size=(5,5),default=True,font=('Helvetica',15),  key='CHK_MOVE1',enable_events=True), 
                sg.Radio('Copy',group_id='CHK2',size=(5,5),font=('Helvetica',15),  key='CHK_COPY1',enable_events=True), 
                sg.Button('Start', size=(int(0.7*12),1), font=('Helvetica',10),key= 'start4b')]
            ]

    Step_6 = [
                [sg.Text('Step 6: Auto Training', font='Any 20', text_color='yellow')],
                [sg.T('1.Choose folder contain train and valid', font='Any 15', text_color = 'orange')],
                #[sg.Input(size=(int(0.7*35),1), font=('Helvetica',12), key='input_move5',readonly= True, text_color='navy',enable_events= True),
                #sg.FolderBrowse(size=(int(0.7*12),1), font=('Helvetica',10),key= 'directory_move5',enable_events=True) ],     
                [sg.T('Then Move folder', font='Any 15', text_color = 'orange')],
                [sg.Button('Move', size=(int(0.7*12),1), font=('Helvetica',10),key= 'move5')],       
                [sg.T('2.Enter all name labels', font='Any 15', text_color = 'orange')],
                [sg.Multiline('',size=(int(0.7*40),int(0.7*10)),text_color='navy' ,key='input_classes5')],
                [sg.Button('OK', size=(int(0.7*12),1), font=('Helvetica',10),key= 'button_classes5')],
                [sg.T('3.Choose epoch', font='Any 15', text_color = 'orange')],
                [sg.Slider(range=(1,500),orientation='h',size=(int(0.7*48),20),font=('Helvetica',11),default_value=300,key= 'input_epoch5')],
                [sg.T('4.Choose folder save model', font='Any 15', text_color = 'orange')],
                [sg.Input(size=(int(0.7*35),1), font=('Helvetica',12), key='input_save5',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(int(0.7*12),1), font=('Helvetica',10),key= 'directory_image5',enable_events=True) ],
                [sg.T('5.Start auto training', font='Any 15', text_color = 'orange')],
                [sg.Combo((416,608),default_value=416,key='sz5'),sg.Button('Start', size=(int(0.7*12),1), font=('Helvetica',10),key= 'start5')],
            ]

    Step_2a = [
                [sg.Frame('',
                [   
                    [sg.Text('Step 2a: Auto filter image, label', font='Any 20', text_color='yellow')],
                    [sg.T(' 1.Choose folder labels (txt)', font='Any 15', text_color = 'orange')],
                    [sg.Input(size=(36,1), font=('Helvetica',12), key='input_folder2a',readonly= True, text_color='navy',enable_events= True),
                    sg.FolderBrowse(size=(8,1), font=('Helvetica',10),key= 'directory_label2a',enable_events=True)],
                ], relief=sg.RELIEF_FLAT),
                ],
                [sg.T(' 2.Choose Parameter', font='Any 15', text_color = 'orange')],
                [sg.Frame('',[
                    [sg.Text('Name',size=(int(0.7*20),1),font=('Helvetica',15), text_color='orange'), 
                    sg.Text('Join',size=(int(0.7*7),1),font=('Helvetica',15), text_color='orange'), 
                    sg.Text('OK',size=(int(0.7*7),1),font=('Helvetica',15), text_color='orange'), 
                    sg.Text('Num',size=(int(0.7*7),1),font=('Helvetica',15), text_color='orange'), 
                    sg.Text('NG',size=(int(0.7*8),1),font=('Helvetica',15), text_color='orange')],
                ], relief=sg.RELIEF_FLAT)],
                [sg.Frame('',[
                    [
                        sg.Text(f'{cls_names[i2]}_2',size=(int(0.7*20),1),font=('Helvetica',15), text_color='pink'), 
                        sg.Checkbox('',size=(int(0.7*3),5),default=True,font=('Helvetica',15),  key=f'{cls_names[i2]}_chon2',enable_events=True), 
                        sg.Checkbox('',size=(int(0.7*3),5),default=True,font=('Helvetica',15),  key=f'{cls_names[i2]}_OK_2',enable_events=True), 
                        sg.Input('1',size=(int(0.7*2),1),font=('Helvetica',15),key= f'{cls_names[i2]}_Num_2',text_color='navy',enable_events=True), 
                        sg.Text('',size=(int(0.7*3),1),font=('Helvetica',15), text_color='red'), 
                        sg.Checkbox('',size=(int(0.7*5),5),font=('Helvetica',15),  key=f'{cls_names[i2]}_NG_2',enable_events=True, disabled=True),  
                    ] for i2 in range(len(cls_names))
                ], relief=sg.RELIEF_FLAT)],
                [sg.T(' 3.Choose type filter   ', font='Any 15', text_color = 'orange'),
                sg.Radio('OK',group_id='chon2a',size=(5,5),default=True,font=('Helvetica',15),  key='chon_OK',enable_events=True, disabled=False),
                sg.Radio('NG',group_id='chon2a',size=(5,5),font=('Helvetica',15),  key='chon_NG',enable_events=True, disabled=False)],
                [sg.T(' 4.Choose folder save ', font='Any 15', text_color = 'orange')],
                [sg.Input(size=(36,1), font=('Helvetica',12), key='input_move2a',readonly= True, text_color='navy',enable_events= True),
                sg.FolderBrowse(size=(8,1), font=('Helvetica',10),key= 'directory_save2a',enable_events=True) ],
                [sg.T(' 5.Start auto filter', font='Any 15', text_color = 'orange'),
                sg.Button('Start', size=(8,1), font=('Helvetica',10),key= 'start2b')],
                ]


    layout_1 = [
        [  
            sg.Column(Step_1, size=(int(0.8*450),int(0.9*700)),  pad=MYBPAD),
            sg.Column(Step_2, scrollable = True, vertical_scroll_only = True, expand_y = True, pad=MYBPAD),
        ]
    ]

    layout_1n = [
        [  
            sg.Column(Step_1n, size=(int(0.8*425),int(0.9*700)),  pad=MYBPAD),
            sg.Column(Step_2n, scrollable = True, vertical_scroll_only = True, expand_y = True, pad=MYBPAD),
        ]
    ]

    layout_2 = [
                [sg.Column(Step_3, size=(int(0.7*480),int(0.9*700)),  pad=MYBPAD),
                sg.Column([ 
                            [sg.Column(Step_4, size=(int(0.9*480),int(0.9*350)), pad=MYBPAD)],
                            [sg.Column(Step_5, size=(int(0.9*480),int(0.9*320)), pad=MYBPAD)]], pad=MYBPAD, background_color=BORDER_COLOR),
                sg.Column(Step_6, size=(int(0.8*480),int(0.9*700)),  pad=MYBPAD),
                ]]
    layout_3 = [
                [sg.Column(Step_3a, size=(int(0.7*480),int(0.9*700))),
                sg.Column([
                            [sg.Column(Step_5a, size=(int(0.9*460),int(0.9*320)))],
                            [sg.Column(Step_5b, size=(int(0.9*460),int(0.9*320)))]], pad=MYBPAD, background_color=BORDER_COLOR),
                sg.Column(Step_2a, scrollable = True, vertical_scroll_only = True, expand_y = True, pad=MYBPAD),
                ]]
        
    layout = [[
                sg.TabGroup([[  
                            sg.Tab('Page 1', layout_1,background_color=BORDER_COLOR),
                            sg.Tab('Page 1+', layout_1n,background_color=BORDER_COLOR),
                            sg.Tab('Page 2', layout_2,background_color=BORDER_COLOR),
                            sg.Tab('Page 3', layout_3,background_color=BORDER_COLOR),
                            ]],background_color=BORDER_COLOR,selected_background_color= BORDER_COLOR,selected_title_color='black')
                ]]
    window = sg.Window('Huynh Le Vu', layout, margins=(0,0), background_color=BORDER_COLOR, grab_anywhere=True)
    return window

mypath1 = "best.pt"

model1 = torch.hub.load('./levu','custom', path= mypath1, source='local',force_reload =False)
model2 = torch.hub.load('./levu','custom', path= mypath1, source='local',force_reload =False)
window = make_window()

while True:             # Event Loop
    event, values = window.read(timeout=20)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break


#1
    if event == 'button_names0' :
        if (values['input_names0'] != ''):
            mynames0 = []
            texts = values['input_names0'].split('\n')
            for text in texts:
                mynames0.append(text)
        else:
            sg.popup_error('Error')


    if event == 'start0':
        if (values['input_image0'] != '') & (values['input_save0'] !='') & (values['input_names0'] != ''):
            i=0
            filedir0= values['input_image0'] + '/*'
            for filename0 in glob.glob(filedir0):
                for path0 in glob.glob(filename0 + '/*'):
                    for myname0 in mynames0:
                        length = -len(myname0)
                        if myname0 == str(path0[length:]):
                            img0 = cv2.imread(path0)
                            cv2.imwrite(values['input_save0'] + '/' + time_to_name() + '.jpg',img0)
                            i+=1
                            print(i)
            window['input_image0'].update(value='')
            window['input_save0'].update(value='')
            window['input_names0'].update(value='')
        else:
            sg.popup_error('Error')

 
#2

    for i1 in range(len(model1.names)):
        if event == f'{model1.names[i1]}_OK_1':
            if values[f'{model1.names[i1]}_OK_1'] == True:
                window[f'{model1.names[i1]}_NG_1'].update(disabled=True)
            else:
                window[f'{model1.names[i1]}_NG_1'].update(disabled=False)
        if event == f'{model1.names[i1]}_NG_1':
            if values[f'{model1.names[i1]}_NG_1'] == True:
                window[f'{model1.names[i1]}_OK_1'].update(disabled=True)
            else:
                window[f'{model1.names[i1]}_OK_1'].update(disabled=False)

    for i2 in range(len(cls_names)):
        if event == f'{cls_names[i2]}_OK_2':
            if values[f'{cls_names[i2]}_OK_2'] == True:
                window[f'{cls_names[i2]}_NG_2'].update(disabled=True)
            else:
                window[f'{cls_names[i2]}_NG_2'].update(disabled=False)
        if event == f'{cls_names[i2]}_NG_2':
            if values[f'{cls_names[i2]}_NG_2'] == True:
                window[f'{cls_names[i2]}_OK_2'].update(disabled=True)
            else:
                window[f'{cls_names[i2]}_OK_2'].update(disabled=False)


    if event == 'input_weight1':
        mypath = values['input_weight1']
        model1 =torch.hub.load('./levu','custom', path= mypath, source='local', force_reload =False)
        window.close()
        window = make_window()
        event, values = window.read(timeout=20)
        window['input_weight1'].update(value=mypath)

    if event == 'input_weight2n':
        mypath = values['input_weight2n']
        model2 =torch.hub.load('./levu','custom', path= mypath, source='local', force_reload =False)        
        window.close()
        window = make_window()
        event, values = window.read(timeout=20)
        window['input_weight2n'].update(value=mypath)


    if event == 'input_folder2a':
        mypath2a= values['input_folder2a']
        mypath = mypath2a + '/classes.txt'
        if not os.path.exists(mypath):
            ftext = sg.PopupGetFile(message='Please find file classes.txt',title='This is PopupFileBrowser ',file_types=(("Classes File", "classes.txt"),))
            if ftext != '' and ftext != None:
               mypath = ftext
            elif ftext == '':
                sg.popup_error('Bạn chưa chọn file classes.txt',title='Lỗi')
        
        if os.path.basename(mypath)=='classes.txt':
            cls_names = []
            f=open(mypath,'r')
            while True:
                line = f.readline()
                if not line:
                    break
                tmp = line.split()
                cls_names.append(tmp[0])
            os.remove('./classes.txt')
            shutil.copyfile(mypath,'./classes.txt') 
            window.close()
            window = make_window()
            event, values = window.read(timeout=20)
            window['input_folder2a'].update(value=mypath2a)
    
    if event == 'start2b':
        NG=[]
        OK={}
        for i1 in range(len(cls_names)):
            if values[f'{cls_names[i1]}_chon2']:
                if values[f'{cls_names[i1]}_OK_2']:
                    OK[str(i1)]=int(values[f'{cls_names[i1]}_Num_2'])
            if values[f'{cls_names[i1]}_chon2']:
                if values[f'{cls_names[i1]}_NG_2']:
                    NG.append(str(i1))
        txt = glob.glob(values['input_folder2a'] + '/*.txt')
        for filename in txt:
            flg1=flg2=False
            lstOK = []
            tenf = os.path.basename(filename)
            with open(filename, 'r') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    tmp = line.split()
                    if tmp[0] in NG:
                        flg1=True
                        break
                    else:
                        lstOK.append(tmp[0])
            if (values['chon_OK']):
                if not flg1:
                    if Counter(lstOK)==OK:
                        flg2=True
            if (values['chon_NG']):
                if flg1:
                    flg2=True
                elif Counter(lstOK)!=OK:
                    flg2=True
            f.close()
            if flg2:
                TM = (values['input_move2a']).replace('\\','/') + '/'
                shutil.move(filename,TM + tenf)
                if os.path.exists(filename[:-3] + 'jpg'):
                    shutil.move(filename[:-3] + 'jpg',TM + tenf[:-3] + 'jpg')
        print('Finished')

    if event == 'start1':
        if (values['input_weight1'] != '') & (values['input_image1'] !='') & (values['input_save1'] != ''):
            #mypath = values['input_weight1']
            #model1 =torch.hub.load('./levu','custom', path= mypath, source='local', force_reload =False)
            size = int(values['sz2b'])
            conf = values['input_conf1']/100
            mydir = values['input_image1'] + '/*.jpg'

            def time_to_name():
                current_time = datetime.datetime.now() 
                name_folder = str(current_time)
                name_folder = list(name_folder)
                for i in range(len(name_folder)):
                    if name_folder[i] == ':':
                        name_folder[i] = '-'
                    if name_folder[i] == ' ':
                        name_folder[i] ='_'
                    if name_folder[i] == '.':
                        name_folder[i] ='-'
                name_folder = ''.join(name_folder)
                return name_folder

            for i,a in zip(reversed(range(len(mydir))),reversed(mydir)):
                if a == '/':
                    index = i
                    break
            hinh=glob.glob(mydir)
            cnt = len(hinh)
            for path1 in hinh:
                name = path1[index+1:-4]

                img1 = cv2.imread(path1)
                
                result1 = model1(path1,size= size,conf = conf) 

                table1 = result1.pandas().xyxy[0]
                #print(table1)
                area_remove1 = []

                myresult1 =0 

                for item in range(len(table1.index)):
                    width1 = table1['xmax'][item] - table1['xmin'][item]
                    height1 = table1['ymax'][item] - table1['ymin'][item]
                    #area1 = width1*height1
                    conf1 = table1['confidence'][item]*100
                    label_name = table1['name'][item]
                    for i1 in range(len(model1.names)):
                        if values[f'{model1.names[i1]}_1'] == True:
                            #if values[f'{model1.names[i1]}_WH'] == True:
                            if label_name == model1.names[i1]:
                                if conf1 < int(values[f'{model1.names[i1]}_conf_1']): 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)
                                if width1 < int(values[f'{model1.names[i1]}_Wn_1']): 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)
                                elif width1 > int(values[f'{model1.names[i1]}_Wx_1']): 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)
                                elif height1 < int(values[f'{model1.names[i1]}_Hn_1']): 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)
                                elif height1 > int(values[f'{model1.names[i1]}_Hx_1']): 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove1.append(item)

                names1 = list(table1['name'])

                for i1 in range(len(model1.names)):
                    if values[f'{model1.names[i1]}_OK_1'] == True:
                        len_name1 = 0
                        for name1 in names1:
                            if name1 == model1.names[i1]:
                                len_name1 +=1
                        if len_name1 != int(values[f'{model1.names[i1]}_Num_1']):
                            print(name,'NG')
                            if values['CHK_MOVE']:
                                shutil.move(path1,values['input_save1'] +'/' + name + '.jpg')
                            elif values['CHK_COPY']:
                                shutil.copy(path1,values['input_save1'] +'/' + name + '.jpg')
                            myresult1 = 1
                            break

                    if values[f'{model1.names[i1]}_NG_1'] == True:
                        if model1.names[i1] in names1:
                            print(name, 'NG')
                            if values['CHK_MOVE']:
                                shutil.move(path1,values['input_save1'] +'/' + name + '.jpg')
                                if os.path.exists(path1[:-3]+'txt'):
                                    shutil.move(path1[:-3] + 'txt',values['input_save1'] +'/' + name + '.txt')
                            elif values['CHK_COPY']:
                                shutil.copy(path1,values['input_save1'] +'/' + name + '.jpg') 
                                if os.path.exists(path1[:-3]+'txt'):
                                    shutil.copy(path1[:-3] + 'txt',values['input_save1'] +'/' + name + '.txt')  
                            myresult1 = 1         
                            break    

                #if myresult1 == 0:
                #    print(name,'OK')
                cnt -= 1
                print(cnt)
    
            window['input_weight1'].update(value='')
            window['input_image1'].update(value='')
            window['input_save1'].update(value='')
        else:
            sg.popup_error('Error')
    


#3
    if event == 'button_classes2' :
        if (values['input_classes2'] != ''):
            #print(values['input_classes2'])
            myclasses2 = []
            texts = values['input_classes2'].split('\n')
            for text in texts:
                myclasses2.append(text)
            print('myclasses2 =',myclasses2)
        else:
            ftext = sg.PopupGetFile(message='Please find file classes.txt',title='This is PopupFileBrowser ',file_types=(("Classes File", "classes.txt"),("Text Files", "*.txt"),))
            if ftext != '' and ftext != None:
                f = open(ftext, 'r')
                txt=''
                myclasses2 = []
                while True: 
                    # Get next line from file
                    line = f.readline()
                    text = line.split('\n')
                    if text[0] != '':
                        myclasses2.append(text[0])
                    txt = txt + line
                    if not line:
                        break   
                print('myclasses2 =',myclasses2)
                window['input_classes2'].update(value=txt)
            elif ftext == '':
                sg.popup_error('Bạn chưa chọn file classes.txt', title='Lỗi')

    if event == 'start2':
        if (values['input_weight2'] != '') & (values['input_image2'] !='') & (values['input_save2'] != '') & (values['input_classes2'] != ''):
            mypath = values['input_weight2']
            model = torch.hub.load('./levu','custom', path= mypath, source='local',force_reload =False)

            mydir = values['input_image2'] + "/*.jpg"
            mysave = values['input_save2'] + '/'
            #print(mysave)
            for i,a in zip(reversed(range(len(mydir))),reversed(mydir)):
                if a == '/':
                    index = i
                    break
            cnt = 0
            for path in glob.glob(mydir):
                name = path[index+1:-4]
                cnt += 1
                print(cnt, name)
                result = model(path,size=int(values['sz2']),conf=values['input_conf2']/100)
                f = open(mysave + name + '.txt', "w")
                for detect in result.xywhn:
                    mydetects = detect.tolist()
                    for item in range(len(mydetects)):
                        mydetect = mydetects[item]
                        number_label = mydetect[-1]
                        number_label = int(number_label)
                        name_label = result.names[number_label]
                        for i,myclass2 in zip(range(len(myclasses2)),myclasses2):
                            if name_label == myclass2:
                                label_text = str(i)
                        f.write(label_text + " " + str(mydetect[0]) + " " + str(mydetect[1]) + " " + str(mydetect[2]) + " " + str(mydetect[3]))
                        f.write("\n")
                    f.close()
            with open(mysave + 'classes.txt', "w") as f: 
                for myclass in myclasses2:
                    f.write(myclass)
                    if myclass != myclasses2[-1]:
                        f.write('\n')
                    
            window['input_weight2'].update(value='')
            window['input_image2'].update(value='')
            window['input_save2'].update(value='')
            window['input_classes2'].update(value='')
        else:
            sg.popup_error('Error')

    if event == 'start2n':
        if (values['input_weight2n'] != '') & (values['input_image2n'] !='') & (values['input_save2n'] != ''):
            mypath = values['input_weight2n']
            #model = torch.hub.load('./levu','custom', path= mypath, source='local',force_reload =False)

            mydir = values['input_image2n'] + "/*.jpg"
            mysave = values['input_save2n'] + '/'
           
            for i,a in zip(reversed(range(len(mydir))),reversed(mydir)):
                if a == '/':
                    index = i
                    break
            myclasses2n=[]
            for i3 in range(len(model2.names)):
                if values[f'{model2.names[i3]}_3']:
                    myclasses2n.append(model2.names[i3])
            
            lastclasses=[]
            if os.path.exists(mysave + 'lastclasses.txt'):
                with open(mysave + 'lastclasses.txt','r') as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        tmp = line.split()
                        lastclasses.append(tmp[0])
                f.close()
            print('lastclasses',lastclasses)
            cnt = 0
            
            for path in glob.glob(mydir):
                fileSize = getsize(path)
                if fileSize < 100000:
                    continue
                name = path[index+1:-4]
                cnt += 1
                print(cnt, name)
                result = model2(path,size=int(values['sz2n']),conf=values['input_conf2n']/100)
                f = open(mysave + name + '.txt', "a")
                for detect in result.xywhn:
                    mydetects = detect.tolist()
                    #print(detect)
                    for item in range(len(mydetects)):
                        mydetect = mydetects[item]
                        number_label = mydetect[-1]
                        i3 = int(number_label)
                        conf_child = float(mydetect[-2])
                        d = float(mydetect[-3])*1200
                        w = float(mydetect[-4])*1600
                        cls=''
                        #Conditions for each label
                        if conf_child > int(values[f'{model2.names[i3]}_conf_3'])/100 and w > float(values[ f'{model2.names[i3]}_Wn_3']) and w < float(values[ f'{model2.names[i3]}_Wx_3']) and d > float(values[ f'{model2.names[i3]}_Hn_3']) and d < float(values[ f'{model2.names[i3]}_Hx_3']):
                            name_label = result.names[i3]
                            for myclass in myclasses2n:
                                if myclass == name_label and myclass not in lastclasses:
                                    cls = str(i3)

                        # if conf_child > int(values[f'{model2.names[i3]}_conf_3'])/100 and w > float(values[ f'{model2.names[i3]}_Wn_3']) and w < float(values[ f'{model2.names[i3]}_Wx_3']) and d > float(values[ f'{model2.names[i3]}_Hn_3']) and d < float(values[ f'{model2.names[i3]}_Hx_3']):
                        #     name_label = result.names[i3]
                        #     for i,myclass in zip(range(len(myclasses2n)),myclasses2n):
                        #         if name_label == myclass:
                        #             label_text = str(i)
                        if cls!='':
                            # f.write(cls + " " + str(mydetect[0]) + " " + str(mydetect[1]) + " " + str(mydetect[2]) + " " + str(mydetect[3]) + '\n')
                            if cls != '1':
                                f.write(cls + " " + str(mydetect[0]) + " " + str(mydetect[1]) + " " + str(mydetect[2]) + " " + str(mydetect[3]) + '\n')
                            else:
                                w = max(float(mydetect[2]),0.013125)
                                h = max(float(mydetect[3]),0.0175)
                                f.write(cls + " " + str(mydetect[0]) + " " + str(mydetect[1]) + " " + str(w) + " " + str(h) + '\n')
                    f.close()

            with open(mysave + 'lastclasses.txt', "a") as f: 
                for myclass in myclasses2n:
                    f.write(myclass)
                    if myclass != myclasses2n[-1]:
                        f.write('\n')
            f.close()

            #class from weight file
            with open(mysave + 'classes.txt', "w") as f: 
                myclasses2n = model2.names
                for myclass in myclasses2n:
                    f.write(myclass)
                    if myclass != myclasses2n[-1]:
                        f.write('\n')
            f.close()
            # window['input_weight2n'].update(value='')
            # window['input_image2n'].update(value='')
            # window['input_save2n'].update(value='')
        else:
            sg.popup_error('Error')
    
    
    if event == 'button_classes2a' :
        if (values['input_classes2a'] != ''):
            #print(values['input_classes2'])
            myclasses2a = []
            texts = values['input_classes2a'].split('\n')
            for text in texts:
                myclasses2a.append(text)
            print('myclasses2a =',myclasses2a)
        else:
            ftext = sg.PopupGetFile(message='Please find file classes.txt',title='This is PopupFileBrowser ',file_types=(("Classes File", "classes.txt"),("Text Files", "*.txt"),))
            if ftext != '' and ftext != None:
                f = open(ftext, 'r')
                txt=''
                myclasses2a = []
                while True: 
                    # Get next line from file
                    line = f.readline()
                    text = line.split('\n')
                    if text[0] != '':
                        myclasses2a.append(text[0])
                    txt = txt + line
                    if not line:
                        break   
                print('myclasses2a =',myclasses2a)
                window['input_classes2a'].update(value=txt)
            elif ftext == '':
                sg.popup_error('Bạn chưa chọn file classes.txt', title='Lỗi')        
            
    if event == 'start2a':
        if (values['input_weight2a'] != '') & (values['input_image2a'] !='') & (values['input_save2a'] != '') & (values['input_classes2a'] != ''):
            mypath = values['input_weight2a']
            model = torch.hub.load('./levu','custom', path= mypath, source='local',force_reload =False)

            mydir = values['input_image2a'] + "/*.jpg"
            mysave = values['input_save2a'] + '/'

            for i,a in zip(reversed(range(len(mydir))),reversed(mydir)):
                if a == '/':
                    index = i
                    break
            cnt = 0
            for path in glob.glob(mydir):
                name = path[index+1:-4]
                cnt += 1
                print(cnt, name)
                result = model(path,size=416,conf=values['input_conf2a']/100)
                f = open(mysave + name + '.txt', "a")
                for detect in result.xywhn:
                    mydetects = detect.tolist()
                    for item in range(len(mydetects)):
                        mydetect = mydetects[item]
                        number_label = mydetect[-1]
                        number_label = int(number_label)
                        name_label = result.names[number_label]
                        for i,myclass2a in zip(range(len(myclasses2a)),myclasses2a):
                            if name_label == myclass2a:
                                label_text = str(i + int(values['sz2a']))
                        f.write(label_text + " " + str(mydetect[0]) + " " + str(mydetect[1]) + " " + str(mydetect[2]) + " " + str(mydetect[3]))
                        f.write("\n")
                    f.close()
            with open(mysave + 'classes.txt', "w") as f: 
                for myclass in myclasses2a:
                    f.write(myclass)
                    if myclass != myclasses2a[-1]:
                        f.write('\n')
                    
            window['input_weight2a'].update(value='')
            window['input_image2a'].update(value='')
            window['input_save2a'].update(value='')
            window['input_classes2a'].update(value='')
        else:
            sg.popup_error('Error')
            
    if event == 'button_classes3':
        if values['input_classes3'] != '':
            myclasses3 = []
            texts = values['input_classes3'].split('\n')
            for text in texts:
                myclasses3.append(text)
            #print('myclasses3 =',myclasses3)
            with open(os.getcwd() + '/labelImg/data/predefined_classes.txt', "w") as f:
                for myclass3 in myclasses3:
                    f.write(myclass3)
                    f.write('\n')
        else:
            ftext = sg.PopupGetFile(message='Please find file classes.txt',title='This is PopupFileBrowser ',file_types=(("Classes File", "classes.txt"),("Text Files", "*.txt"),))
            if ftext != '' and ftext != None:
                f = open(ftext, 'r')
                txt=''
                myclasses3 = []
                while True: 
                    # Get next line from file
                    line = f.readline()
                    text = line.split('\n')
                    if text[0] != '':
                        myclasses3.append(text[0])
                    txt = txt + line
                    if not line:
                        break
                print('myclasses3 =',myclasses3)
                with open(os.getcwd() + '/labelImg/data/predefined_classes.txt', "w") as f:
                    for myclass3 in myclasses3:
                        f.write(myclass3)
                        f.write('\n')
                window['input_classes3'].update(value=txt)
                #sg.popup('Xác nhận nội dung Labels rồi bấm OK lần nữa',title='Nhắc nhở')
            elif ftext == '':
                sg.popup_error('Bạn chưa chọn file classes.txt',title='Lỗi')

    if event == 'program3':
        if values['input_classes3'] != '':
            program_dir = os.path.join(os.getcwd(), 'labelImg' , 'labelImg.py')
            #ExecuteCommandSubprocess('python', program_dir)
            subprocess.call(['python',program_dir])
            #subprocess.call(['python', 'test2.py'])

            window['input_classes3'].update(value='')
        else:
            sg.popup_error('Error')  

#5
    if event == 'start4':
        if (values['input_image4'] !='') & (values['input_save4'] != ''):
            dir = values['input_image4']
            dir = dir + ('/*')
            list_image = [] 
            list_text = []
            list_image_train = []
            list_image_valid = []
            list_text_train = []
            list_text_valid = []

            list_name_train = []
            list_name_valid = []

            os.mkdir(values['input_save4'] + '/train')
            os.mkdir(values['input_save4'] + '/train/images')
            os.mkdir(values['input_save4'] + '/train/labels')

            os.makedirs(values['input_save4'] + '/valid')
            os.mkdir(values['input_save4'] + '/valid/images')
            os.mkdir(values['input_save4'] + '/valid/labels')

            for i,a in zip(reversed(range(len(dir))),reversed(dir)):
                if a == '/':
                    index = i
                    break

            for path in glob.glob(dir):
                name = path[index+1:-4]
                extension = path[-3:]

                if extension == 'jpg':
                    list_image.append(path)

                if extension == 'txt':
                    list_text.append(path)

            #random.seed(0)
            list_image_train = random.sample(list_image,int(len(list_image)*0.85))


            for item_image in list_image:
                if item_image not in list_image_train: 
                    list_image_valid.append(item_image)


            for path_train in list_image_train:
                name_train = path_train[index+1:-4]
                list_name_train.append(name_train)

            for path_valid in list_image_valid:
                name_valid = path_valid[index+1:-4]
                list_name_valid.append(name_valid)

            for item_text in list_text:
                name_text = item_text[index+1:-4]
                if name_text in list_name_train:
                    list_text_train.append(item_text)
                elif name_text in list_name_valid:
                    list_text_valid.append(item_text)

            for path1 in list_image_train:
                name1 = path1[index+1:-4]
                shutil.copyfile(path1,values['input_save4']+'/train/images/'+ name1 + '.jpg')

            for path2 in list_text_train:
                name2 = path2[index+1:-4]
                shutil.copyfile(path2,values['input_save4']+'/train/labels/'+ name2 + '.txt')

            for path3 in list_image_valid:
                name3 = path3[index+1:-4]
                shutil.copyfile(path3,values['input_save4']+'/valid/images/'+ name3 + '.jpg')

            for path4 in list_text_valid:
                name4 = path4[index+1:-4]
                shutil.copyfile(path4,values['input_save4']+'/valid/labels/'+ name4 + '.txt')

            window['input_image4'].update(value='')
            window['input_save4'].update(value='')

        else:
            sg.popup_error('Error')  

    # if event == 'start4a':
    #     tatca = glob.glob(values['input_image4a'] + '/**/*.*', recursive=True)
    #     i=ht=0
    #     #window['speed5a'].update(disabled=False)
    #     for f in tatca:
    #         tenf = os.path.basename(f)
    #         print(tenf)
    #         shutil.copyfile(f,values['directory_save4a'] + '/' + tenf)
    #         i += 1
    #         if ht != int(100 * (i/len(tatca))):
    #             ht = int(100 * (i/len(tatca)))
    #             window['speed5a'].update(value=ht)
    #             window.refresh()
                
    #     sg.popup('Xong')

    if event == 'start4a':
        print('a')
        test()
        # tatca = glob.glob(values['input_image4a'] + '/**/*.*', recursive=True)
        # i=ht=0
        # if len(tatca) > 0:
        #     window['progress1'].update(visible=True)
        #     for f in tatca:
        #         tenf = os.path.basename(f)
        #         print(tenf)
        #         shutil.copyfile(f,values['directory_save4a'] + '/' + tenf)
        #         i += 1
        #         if ht != int(100 * (i/len(tatca))):
        #             ht = int(100 * (i/len(tatca)))
        #             window['progress1'].UpdateBar(ht)
        #             window.refresh()
                    
        # sg.popup('Xong')

    if event == 'start4b':
        txt = glob.glob(values['input_image4b'] + '/*.txt')
        cls = int(values['cls5'])
        for filename in txt:
            tenf = os.path.basename(filename)
            if tenf != 'classes.txt' and tenf != 'lastclasses.txt':
                with open(filename, 'r') as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        tmp = line.split()
                        if int(tmp[0])==cls:
                            print(tenf)
                            f.close()
                            if os.path.exists(filename[:-3] + 'jpg'):
                                if values['CHK_MOVE1']:
                                    shutil.move(filename,values['directory_save4b'] + '/' + tenf)
                                    shutil.move(filename[:-3] + 'jpg',values['directory_save4b'] + '/' + tenf[:-3] + 'jpg')
                                elif values['CHK_COPY1']:
                                    shutil.copyfile(filename,values['directory_save4b'] + '/' + tenf)
                                    shutil.copyfile(filename[:-3] + 'jpg',values['directory_save4b'] + '/' + tenf[:-3] + 'jpg') 
                            else:
                                os.remove(filename)
                            break   
            else:
                shutil.copyfile(filename,values['directory_save4b'] + '/' + tenf)
        sg.popup('Xong')
#6

    if event == 'move5':
        cpath = sg.PopupGetFolder(message='Hãy chọn thư mục đã chia train//valid',title='Select Folder')
        if cpath != '' and cpath != None:
            try:
                if os.path.isdir(os.getcwd() + '/train'):
                    shutil.rmtree(os.getcwd() + '/train')
                if os.path.isdir(os.getcwd() + '/valid'):
                    shutil.rmtree(os.getcwd() + '/valid')
                #shutil.copytree(values['input_move5'] + '/train',os.getcwd() + '/train')
                #shutil.copytree(values['input_move5'] + '/valid',os.getcwd() + '/valid')
                shutil.copytree(cpath + '/train',os.getcwd() + '/train')
                shutil.copytree(cpath + '/valid',os.getcwd() + '/valid')
                sg.popup('Xong')
            except:
                sg.popup_error('Lỗi')   
                print(traceback.format_exc())
        elif cpath == '':
            sg.popup_error('Bạn chưa nhập folder')

    if event == 'button_classes5' :
        if (values['input_classes5'] != ''):
            myclasses5 = []
            texts5 = values['input_classes5'].split('\n')
            for text in texts5:
                myclasses5.append(text)

            with open(os.getcwd() + '/levu/data.yaml', "w") as f:
                f.write('train: ' + os.getcwd() + '/train/images')
                f.write('\n')
                f.write('val: ' + os.getcwd() + '/valid/images')
                f.write('\n')
                f.write('nc: '  + str(len(myclasses5)))     
                f.write('\n')
                f.write('names: '  + str(myclasses5))     

            with open(os.getcwd() + '/levu/models/levu.yaml', "w") as f:
                f.write('nc: ' +  str(len(myclasses5)) + '\n' + 
                        'depth_multiple: 0.33  # model depth multiple' + '\n' + 
                        'width_multiple: 0.50  # layer channel multiple' + '\n' + 
                        'anchors:' + '\n' + 
                        '  - [10,13, 16,30, 33,23]  # P3/8' + '\n' + 
                        '  - [30,61, 62,45, 59,119]  # P4/16' + '\n' + 
                        '  - [116,90, 156,198, 373,326]  # P5/32' + '\n' + 

                        'backbone:' + '\n' + 

                        '  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2' + '\n' + 
                        '   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4' + '\n' + 
                        '   [-1, 3, C3, [128]],' + '\n' + 
                        '   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8' + '\n' + 
                        '   [-1, 6, C3, [256]],' + '\n' + 
                        '   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16' + '\n' + 
                        '   [-1, 9, C3, [512]],' + '\n' + 
                        '   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32' + '\n' + 
                        '   [-1, 3, C3, [1024]],' + '\n' + 
                        '   [-1, 1, SPPF, [1024, 5]],  # 9' + '\n' + 
                        '  ]' + '\n' + 

                        'head:' + '\n' + 
                        '  [[-1, 1, Conv, [512, 1, 1]],' + '\n' + 
                        "   [-1, 1, nn.Upsample, [None, 2, 'nearest']]," + '\n' + 
                        '   [[-1, 6], 1, Concat, [1]],  # cat backbone P4' + '\n' + 
                        '   [-1, 3, C3, [512, False]],  # 13' + '\n' + 

                        '   [-1, 1, Conv, [256, 1, 1]],' + '\n' + 
                        "   [-1, 1, nn.Upsample, [None, 2, 'nearest']]," + '\n' + 
                        '   [[-1, 4], 1, Concat, [1]],  # cat backbone P3' + '\n' + 
                        '   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)' + '\n' + 

                        '   [-1, 1, Conv, [256, 3, 2]],' + '\n' + 
                        '   [[-1, 14], 1, Concat, [1]],  # cat head P4' + '\n' + 
                        '   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)' + '\n' + 

                        '   [-1, 1, Conv, [512, 3, 2]],' + '\n' + 
                        '   [[-1, 10], 1, Concat, [1]],  # cat head P5' + '\n' + 
                        '   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)' + '\n' + 

                        '   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)' + '\n' + 
                        '  ]'
                        )
            window['button_classes5'].update('OK')

        else:
            ftext = sg.PopupGetFile(message='Please find file classes.txt',title='This is PopupFileBrowser ',file_types=(("Classes File", "classes.txt"),("Text Files", "*.txt"),))
            if ftext != '' and ftext != None:
                f = open(ftext, 'r')
                txt=''
                while True: 
                    # Get next line from file
                    line = f.readline()
                    txt = txt + line
                    if not line:
                        break
                window['input_classes5'].update(value=txt)
                window['button_classes5'].update('Confirm')
                sg.popup('Xác nhận nội dung Labels rồi bấm Confirm',title='Nhắc nhở')     
            elif ftext == '':
                sg.popup_error('Bạn chưa chọn file classes.txt',title='Lỗi')

    if event == 'start5':
        if values['input_classes5'] != '':
            #program_dir5 = os.path.join(os.getcwd()  + '/levu/' , 'train.py')
            dir_py5 = os.path.join(os.getcwd(), 'levu', 'hlvtrain.py')
            dir_data5 = os.path.join(os.getcwd(), 'levu', 'data.yaml')
            dir_model5 = os.path.join(os.getcwd(), 'levu', 'models', 'levu.yaml')
            name_folder = time_to_name()
            program_dir5 = [ dir_py5, ' --img ','{}'.format(int(values['sz5'])), ' --batch ', '1' ,' --epochs ', '{}'.format(int(values['input_epoch5'])) , ' --data ', dir_data5 , ' --cfg ', dir_model5, ' --weights ', '""', ' --name ', 'my_results' + '{}'.format(name_folder),  ' --cache']
   
            subprocess.call(['python', program_dir5])
            #subprocess.Popen(program_dir5)
            try:
                if values['input_classes5'] != '':
                    filebest = time_to_name() + '.pt'
                    shutil.copyfile(os.getcwd() + '/levu/runs/train/my_results'+ name_folder +'/weights/best.pt',values['input_save5'] + '/' + filebest)
                    window['input_classes5'].update(value='')
                    #window['input_move5'].update(value='')
                    window['input_save5'].update(value='')
                else:
                    sg.popup_error('Error') 

            except:
                sg.popup_error('Error') 
        else:
            sg.popup_error('Error')  
    

    # if event == 'get5':
    #     try:
    #         if values['input_classes5'] != '':
    #             shutil.copyfile(os.getcwd() + '/levu/runs/train/my_results'+ name_folder +'/weights/best.pt',values['input_save5'] + '/best.pt')
    #             window['input_classes5'].update(value='')
    #             #window['input_move5'].update(value='')
    #             window['input_save5'].update(value='')
    #         else:
    #             sg.popup_error('Error') 

    #     except:
    #         sg.popup_error('Error')  