import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import glob
import torch
import mysql.connector
from tkinter import messagebox
import threading
# import stapipy as st
import numpy as np
import cv2
import socket
import os
import time
import PySimpleGUI as sg
from PIL import Image,ImageTk
from PIL import Image
from yaml import load
import socket
import threading
from udp import UDPFinsConnection
from initialization import FinsPLCMemoryAreas
import shutil
soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SCALE_X_CAM1 = 1#1280/2048
SCALE_Y_CAM1 = 1#960/1536


SCALE_X_CAM2 = 1#640/1440
SCALE_Y_CAM2 = 1#480/1080

class CMyCallback:
    """
    Class that contains a callback function.
    """
    
    def __init__(self):
        self._image = None
        self._lock = threading.Lock()

    @property
    def image(self):
        """Property: return PyIStImage of the grabbed image."""
        duplicate = None
        self._lock.acquire()
        if self._image is not None:
            duplicate = self._image.copy()
        self._lock.release()
        return duplicate

    def datastream_callback1(self, handle=None, context=None):
        """
        Callback to handle events from DataStream.

        :param handle: handle that `trigger` the callback.
        :param context: user data passed on during callback registration.
        """
        st_datastream = handle.module
        if st_datastream:
            with st_datastream.retrieve_buffer() as st_buffer:
                # Check if the acquired data contains image data.
                if st_buffer.info.is_image_present:
                    # Create an image object.
                    st_image = st_buffer.get_image()

                    # Check the pixelformat of the input image.
                    pixel_format = st_image.pixel_format
                    pixel_format_info = st.get_pixel_format_info(pixel_format)

                    # Only mono or bayer is processed.
                    if not(pixel_format_info.is_mono or pixel_format_info.is_bayer):
                        return

                    # Get raw image data.
                    data = st_image.get_image_data()

                    # Perform pixel value scaling if each pixel component is
                    # larger than 8bit. Example: 10bit Bayer/Mono, 12bit, etc.
                    if pixel_format_info.each_component_total_bit_count > 8:
                        nparr = np.frombuffer(data, np.uint16)
                        division = pow(2, pixel_format_info
                                       .each_component_valid_bit_count - 8)
                        nparr = (nparr / division).astype('uint8')
                    else:
                        nparr = np.frombuffer(data, np.uint8)

                    # Process image for display.
                    nparr = nparr.reshape(st_image.height, st_image.width, 1)

                    # Perform color conversion for Bayer.
                    if pixel_format_info.is_bayer:
                        bayer_type = pixel_format_info.get_pixel_color_filter()
                        if bayer_type == st.EStPixelColorFilter.BayerRG:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_RG2RGB)
                        elif bayer_type == st.EStPixelColorFilter.BayerGR:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_GR2RGB)
                        elif bayer_type == st.EStPixelColorFilter.BayerGB:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_GB2RGB)
                        elif bayer_type == st.EStPixelColorFilter.BayerBG:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_BG2RGB)

                    # Resize image and store to self._image.
                    nparr = cv2.resize(nparr, None,
                                       fx=SCALE_X_CAM1,
                                       fy=SCALE_Y_CAM1)
                    self._lock.acquire()
                    self._image = nparr
                    self._lock.release()


    def datastream_callback2(self, handle=None, context=None):
        """
        Callback to handle events from DataStream.

        :param handle: handle that trigger the callback.
        :param context: user data passed on during callback registration.
        """
        st_datastream = handle.module
        if st_datastream:
            with st_datastream.retrieve_buffer() as st_buffer:
                # Check if the acquired data contains image data.
                if st_buffer.info.is_image_present:
                    # Create an image object.
                    st_image = st_buffer.get_image()

                    # Check the pixelformat of the input image.
                    pixel_format = st_image.pixel_format
                    pixel_format_info = st.get_pixel_format_info(pixel_format)

                    # Only mono or bayer is processed.
                    if not(pixel_format_info.is_mono or pixel_format_info.is_bayer):
                        return

                    # Get raw image data.
                    data = st_image.get_image_data()

                    # Perform pixel value scaling if each pixel component is
                    # larger than 8bit. Example: 10bit Bayer/Mono, 12bit, etc.
                    if pixel_format_info.each_component_total_bit_count > 8:
                        nparr = np.frombuffer(data, np.uint16)
                        division = pow(2, pixel_format_info
                                       .each_component_valid_bit_count - 8)
                        nparr = (nparr / division).astype('uint8')
                    else:
                        nparr = np.frombuffer(data, np.uint8)

                    # Process image for display.
                    nparr = nparr.reshape(st_image.height, st_image.width, 1)

                    # Perform color conversion for Bayer.
                    if pixel_format_info.is_bayer:
                        bayer_type = pixel_format_info.get_pixel_color_filter()
                        if bayer_type == st.EStPixelColorFilter.BayerRG:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_RG2RGB)
                        elif bayer_type == st.EStPixelColorFilter.BayerGR:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_GR2RGB)
                        elif bayer_type == st.EStPixelColorFilter.BayerGB:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_GB2RGB)
                        elif bayer_type == st.EStPixelColorFilter.BayerBG:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_BG2RGB)

                    # Resize image and store to self._image.
                    nparr = cv2.resize(nparr, None,
                                       fx=SCALE_X_CAM2,
                                       fy=SCALE_Y_CAM2)
                    self._lock.acquire()
                    self._image = nparr
                    self._lock.release()


def set_enumeration(nodemap, enum_name, entry_name):
    enum_node = st.PyIEnumeration(nodemap.get_node(enum_name))
    entry_node = st.PyIEnumEntry(enum_node[entry_name])
    enum_node.set_entry_value(entry_node)



def setup_camera_stc(num_camera,st_system,cb_func):
    st_device = st_system.create_first_device()
    print(f'Device{num_camera}=', st_device.info.display_name)
    st_datastream = st_device.create_datastream()
    callback = st_datastream.register_callback(cb_func)
    st_datastream.start_acquisition()
    st_device.acquisition_start()
    remote_nodemap = st_device.remote_port.nodemap
    set_enumeration(remote_nodemap,"TriggerMode", "Off")
    return  st_datastream, st_device,remote_nodemap




def setup_camera1_stc(st_system,cb_func1):
 
    st_device1 = st_system.create_first_device()
    print('Device1=', st_device1.info.display_name)
    st_datastream1 = st_device1.create_datastream()
    callback1 = st_datastream1.register_callback(cb_func1)
    st_datastream1.start_acquisition()
    st_device1.acquisition_start()
    remote_nodemap = st_device1.remote_port.nodemap
    set_enumeration(remote_nodemap,"TriggerMode", "Off")
    return  st_datastream1, st_device1,remote_nodemap

  


def setup_camera2_stc(st_system,cb_func2):
    st_device2 = st_system.create_first_device()
    print('Device2=', st_device2.info.display_name)
    st_datastream2 = st_device2.create_datastream()
    callback2 = st_datastream2.register_callback(cb_func2)
    st_datastream2.start_acquisition()
    st_device2.acquisition_start()
    remote_nodemap2 = st_device2.remote_port.nodemap
    set_enumeration(remote_nodemap2,"TriggerMode", "Off")
    return  st_datastream2, st_device2,remote_nodemap2



def config_off_auto(remote_nodemap):
    # Configure the ExposureMode
    node_name_eps = "ExposureMode"

    node_eps = remote_nodemap.get_node(node_name_eps)

    if not node_eps.is_writable:
        print('not node ExposureMode')
    enum_node_eps = st.PyIEnumeration(node_eps)
    enum_entries_eps = enum_node_eps.entries
    selection_eps = 1

    if selection_eps < len(enum_entries_eps):
        enum_entry_eps = enum_entries_eps[selection_eps]
        enum_node_eps.set_int_value(enum_entry_eps.value)


    #Configure the BalanceWhiteAuto
    node_name_bwa = "BalanceWhiteAuto"

    node_bwa = remote_nodemap.get_node(node_name_bwa)

    if not node_bwa.is_writable:
        print('not node BalanceWhiteAuto')
    enum_node_bwa = st.PyIEnumeration(node_bwa)
    enum_entries_bwa = enum_node_bwa.entries
    selection_bwa = 0

    if selection_bwa < len(enum_entries_bwa):
        enum_entry_bwa = enum_entries_bwa[selection_bwa]
        enum_node_bwa.set_int_value(enum_entry_bwa.value)
        



def BalanceWhiteAuto(remote_nodemap,choose_value_br,index):

    enum_name_brs = "BalanceRatioSelector"
    numeric_name_br = "BalanceRatio"
    node_brs = remote_nodemap.get_node(enum_name_brs)
    if not node_brs.is_writable:
        print('not node '+ enum_name_brs)

    enum_node_brs = st.PyIEnumeration(node_brs)
    enum_entries_brs = enum_node_brs.entries

    enum_entry_brs = enum_entries_brs[index]
    if enum_entry_brs.is_available:
        enum_node_brs.value = enum_entry_brs.value
        #print(st.PyIEnumEntry(enum_entry).symbolic_value)
        node_name_br = numeric_name_br
        node_br = remote_nodemap.get_node(node_name_br)

        if not node_br.is_writable:
            # print('not node '+ name)
            pass
        else:
            if node_br.principal_interface_type == st.EGCInterfaceType.IFloat:
                node_value_br = st.PyIFloat(node_br)
            elif node_br.principal_interface_type == st.EGCInterfaceType.IInteger:
                node_value_br = st.PyIInteger(node_br)
            value_br = choose_value_br

            if node_br.principal_interface_type == st.EGCInterfaceType.IFloat:
                value_br = float(value_br)
                pass
            else:
                value_br = int(value_br)
                pass
            if node_value_br.min <= value_br <= node_value_br.max:
                node_value_br.value = value_br


def set_exposure_or_gain(remote_nodemap,node_name,value):
    if remote_nodemap.get_node(node_name):
        node = remote_nodemap.get_node(node_name)
        if not node.is_writable:
            print('not node' + node_name)
        else:
            if node.principal_interface_type == st.EGCInterfaceType.IFloat:
                node_value = st.PyIFloat(node)
            elif node.principal_interface_type == st.EGCInterfaceType.IInteger:
                node_value = st.PyIInteger(node)
            value = float(value)
            if node.principal_interface_type == st.EGCInterfaceType.IFloat:
                value = float(value)
            else:
                value = int(value)
            if node_value.min <= value <= node_value.max:
                node_value.value = value
            


def set_balance_white_auto(remote_nodemap,index,value):
    enum_name_brs = "BalanceRatioSelector"
    numeric_name_br = "BalanceRatio"
    node_brs = remote_nodemap.get_node(enum_name_brs)
    if not node_brs.is_writable:
        print('not node  '+ enum_name_brs)

    enum_node_brs = st.PyIEnumeration(node_brs)
    enum_entries_brs = enum_node_brs.entries

    enum_entry_brs = enum_entries_brs[index]
    if enum_entry_brs.is_available:
        enum_node_brs.value = enum_entry_brs.value

        node_name_br = numeric_name_br
        node_br = remote_nodemap.get_node(node_name_br)

        if not node_br.is_writable:
            print('not node '+ numeric_name_br + index)
        else:
            if node_br.principal_interface_type == st.EGCInterfaceType.IFloat:
                node_value_br = st.PyIFloat(node_br)
            elif node_br.principal_interface_type == st.EGCInterfaceType.IInteger:
                node_value_br = st.PyIInteger(node_br)
            value_br = value

            if node_br.principal_interface_type == st.EGCInterfaceType.IFloat:
                value_br = float(value_br)
            else:
                value_br = int(value_br)
            if node_value_br.min <= value_br <= node_value_br.max:
                node_value_br.value = value_br
            print(st.PyIEnumEntry(enum_entry_brs).symbolic_value + ' : ' + str(node_value_br.value))


def set_config_init_camera(remote_nodemap,value_exposure_time, value_gain,select_balance, value_balance_red,value_balance_green, value_balance_blue):

    set_exposure_or_gain(remote_nodemap,"ExposureTime",value_exposure_time)
    set_exposure_or_gain(remote_nodemap,"ExposureTimeRaw",value_exposure_time)

    set_exposure_or_gain(remote_nodemap,"Gain",value_gain)
    set_exposure_or_gain(remote_nodemap,"GainRaw",value_gain)
    if select_balance == 'off':
        pass
    else:
        set_balance_white_auto(remote_nodemap,0,value_balance_red)
        set_balance_white_auto(remote_nodemap,3,value_balance_green)
        set_balance_white_auto(remote_nodemap,4,value_balance_blue)

class Main_Display():
    

    def display_images(self, camera_frame, camera_number):
        # global time_processing
        handle_image = Model_Camera_1()
        for widget in camera_frame.winfo_children():
            widget.destroy()
        image_paths = glob.glob(f"C:/Users/CCSX009/Documents/yolov5/test_image/camera{camera_number}/*.jpg")
        if len(image_paths) == 0:
            print(f'Folder CAM{camera_number} empty')
        else:
            for filename1 in image_paths:
                img1_orgin = cv2.imread(filename1)
                if img1_orgin is None:
                    print('loading img 1...')
                    continue
                image_result,time_processing = handle_image.processing_handle_image(img1_orgin)
                # time_processing_output.delete(0, tk.END)
                # time_processing_output.insert(0, f'{time_processing}')
                time_processing_output.config(text=f'{time_processing}')
                img_pil = Image.fromarray(image_result)
                photo = ImageTk.PhotoImage(img_pil)
                image_label = tk.Label(camera_frame, image=photo)
                image_label.image = photo
                image_label.pack()
                # removefile()

    def update_images(self,window, camera1_frame, camera2_frame):
        self.display_images(camera1_frame, 1)
        # self.display_images(camera2_frame, 2)
        window.after(100, self.update_images,window,camera1_frame, camera2_frame)

    def create_camera_frame_cam1(self,notebook, camera_number):
        global time_processing_output
        camera_frame = ttk.LabelFrame(notebook, text=f"Camera {camera_number}", width=800, height=800)
        camera_frame.grid(row=0, column=camera_number-1, padx=80, pady=20, sticky="nws")
        time_frame = ttk.LabelFrame(notebook, text=f"Time Processing Camera {camera_number}", width=400, height=100)
        time_frame.grid(row=1, column=camera_number-1, padx=80, pady=10, sticky="nws")
        # time_processing_output = tk.Entry(time_frame, fg='black', font=('Segoe UI', 30), width=20)
        # time_processing_output.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        time_processing_output = tk.Label(time_frame, text='0 ms', fg='black', font=('Segoe UI', 30), width=20, anchor='center')
        time_processing_output.grid(row=0, column=0, padx=10, pady=10, sticky='ew')
        return camera_frame
    
    def create_camera_frame_cam2(self,notebook, camera_number):
        # global time_processing_output
        camera_frame = ttk.LabelFrame(notebook, text=f"Camera {camera_number}", width=800, height=800)
        camera_frame.grid(row=0, column=camera_number-1, padx=80, pady=20, sticky="nws")
        time_frame = ttk.LabelFrame(notebook, text=f"Time Processing Camera {camera_number}", width=400, height=100)
        time_frame.grid(row=1, column=camera_number-1, padx=80, pady=10, sticky="nws")
        # time_processing_output = tk.Entry(time_frame, fg='black', font=('Segoe UI', 30), width=20)
        # time_processing_output.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        time_processing_output = tk.Label(time_frame, text='0 ms', fg='black', font=('Segoe UI', 30), width=20, anchor='w')
        time_processing_output.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        return camera_frame

    def Display_Camera(self,notebook):
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="Display Camera")

        camera1_frame = self.create_camera_frame_cam1(tab1, 1)
        camera2_frame = self.create_camera_frame_cam2(tab1, 2)

        return camera1_frame, camera2_frame


class Model_Camera_1():
    
    def __init__(self):
        # self.notebook = notebook
        self.item_code_cfg = "ABC2D3F"

    def connect_database(self):
        database = MySQL_Connection("127.0.0.1","root1","987654321","model_1")
        cursor, db_connection  = database.Connect_MySQLServer()
        check_connection = database.check_connection()
        reconnect = database.reconnect()
        return cursor,db_connection,check_connection,reconnect

    def save_data_model_1(self, weights, scale_conf_all, item_code, model_name_labels, join, ok_vars, ng_vars, num_inputs, wn_inputs, wx_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales):
        confirm_save_data = messagebox.askokcancel("Confirm", "Are you sure you want to save the data?")
        
        if confirm_save_data:
            cursor, db_connection,check_connection,reconnect = self.connect_database()
            if not check_connection and not reconnect:
                print( check_connection, reconnect)
                return
            try:
                cursor.execute("SELECT COUNT(*) FROM test_model_cam1_model1 WHERE item_code = %s", (str(item_code.get()),))
                exists = cursor.fetchone()[0] > 0

                weight = weights.get()
                confidence_all = int(scale_conf_all.get())
                item_code_value = str(item_code.get())
                
                for i1 in range(len(model_name_labels)):
                    label_name = model_name_labels[i1].cget("text")
                    join_detect = join[i1].get()
                    OK_jont = ok_vars[i1].get()
                    NG_jont = ng_vars[i1].get()
                    num_labels = int(num_inputs[i1].get())
                    width_min = int(wn_inputs[i1].get())
                    width_max = int(wx_inputs[i1].get())
                    height_min = int(hn_inputs[i1].get())
                    height_max = int(hx_inputs[i1].get())
                    PLC_value = int(plc_inputs[i1].get())
                    cmpnt_conf = int(conf_scales[i1].get())
                
                    print(label_name)
                    print(join_detect)
                    print(OK_jont)
                    print(NG_jont)
                    print(num_labels)
                    print(width_min)
                    print(width_max)
                    print(height_min)
                    print(height_max)
                    print(cmpnt_conf)
                    print('-----------------------')

                    if exists:
                        query_sql = """
                        UPDATE test_model_cam1_model1
                        SET weight = %s, confidence_all = %s, label_name = %s, join_detect = %s, OK = %s, NG = %s, 
                            num_labels = %s, width_min = %s, width_max = %s, height_min = %s, height_max = %s, 
                            PLC_value = %s, cmpnt_conf = %s
                        WHERE item_code = %s AND label_name = %s
                        """
                        values = (weight, confidence_all, label_name, join_detect, OK_jont, NG_jont, num_labels, 
                                  width_min, width_max, height_min, height_max, PLC_value, cmpnt_conf, self.item_code_cfg, label_name)                      
                    else:
                        query_sql = """
                        INSERT INTO test_model_cam1_model1
                        (item_code, weight, confidence_all, label_name, join_detect, OK, NG, num_labels, width_min, width_max, 
                         height_min, height_max, PLC_value, cmpnt_conf)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        values = (item_code_value, weight, confidence_all, label_name, join_detect, OK_jont, NG_jont, num_labels, 
                                  width_min, width_max, height_min, height_max, PLC_value, cmpnt_conf)
                    cursor.execute(query_sql, values)
                db_connection.commit()
                cursor.close()
                db_connection.close()
                messagebox.showinfo("Notification", "Saved parameters successfully!")
            except Exception as e:
                cursor.close()
                db_connection.close()
                messagebox.showinfo("Notification", f"Data saved failed! Error: {str(e)}")
        else:
            pass

    def load_data_model_1(self):
        cursor, db_connection,_,_ = self.connect_database()
        cursor.execute("SELECT * FROM test_model_cam1_model1 WHERE item_code = %s", (self.item_code_cfg,))
        records = cursor.fetchall()
        cursor.close()
        db_connection.close()
        if records:
            first_record = records[0]
            load_item_code = first_record[1]
            load_path_weight = first_record[2]
            load_confidence_all_scale = first_record[3]
        return records,load_path_weight,load_item_code,load_confidence_all_scale

    def load_parameters_model_1(self,model1,records,load_path_weight,load_item_code,load_confidence_all_scale):
        weights.delete(0, tk.END)
        weights.insert(0, load_path_weight)
        item_code.delete(0, tk.END)
        item_code.insert(0, load_item_code)
        scale_conf_all.set(load_confidence_all_scale)
        for i1 in range(len(model1.names)):
            for record in records:
                if record[4] == model1.names[i1]:

                    join[i1].set(bool(record[5]))

                    ok_vars[i1].set(bool(record[6]))

                    ng_vars[i1].set(bool(record[7]))

                    num_inputs[i1].delete(0, tk.END)
                    num_inputs[i1].insert(0, record[8])

                    wn_inputs[i1].delete(0, tk.END)
                    wn_inputs[i1].insert(0, record[9])

                    wx_inputs[i1].delete(0, tk.END)
                    wx_inputs[i1].insert(0, record[10])

                    hn_inputs[i1].delete(0, tk.END)
                    hn_inputs[i1].insert(0, record[11])
                    
                    hx_inputs[i1].delete(0, tk.END)
                    hx_inputs[i1].insert(0, record[12])

                    plc_inputs[i1].delete(0, tk.END)
                    plc_inputs[i1].insert(0, record[13])

                    conf_scales[i1].set(record[14])
        
    def change_model_1(self,Frame_2,weights,scale_conf_all,item_code):
        global model1
        selected_file = filedialog.askopenfilename(title="Choose a file", filetypes=[("Model Files", "*.pt")])
        if selected_file:
            weights.delete(0,tk.END)
            weights.insert(0,selected_file)
            item_code.delete(0,tk.END)
            scale_conf_all.set(15)
            model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5','custom', path=selected_file, source='local', force_reload=False)
            for widget in Frame_2.grid_slaves():
                widget.grid_forget()
            self.option_1_parameters(Frame_2,model1)    

    def Camera_1_Settings(self,notebook):
        global model1
        records,load_path_weight,load_item_code,load_confidence_all_scale = self.load_data_model_1()
        
        model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5','custom', path=load_path_weight, source='local', force_reload=False)
        
        camera_settings_tab = ttk.Frame(notebook)
        notebook.add(camera_settings_tab, text="Camera 1 Settings")

        frame_width = 1500
        frame_height = 2000

        Frame_1 = ttk.LabelFrame(camera_settings_tab, text="Frame 1", width=frame_width, height=frame_height)
        Frame_2 = ttk.LabelFrame(camera_settings_tab, text="Frame 2", width=frame_width, height=frame_height)

        Frame_1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  
        Frame_2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        

        self.option_1_layout_models(Frame_1,Frame_2)
        self.option_1_parameters(Frame_2,model1)
        self.load_parameters_model_1(model1,records,load_path_weight,load_item_code,load_confidence_all_scale)
       
    def option_1_layout_models(self,Frame_1, Frame_2):

        global weights,scale_conf_all,size_model,item_code
        
        ttk.Label(Frame_1, text='1. File train detect model', font=('Segoe UI', 12)).grid(column=0, row=0, padx=10, pady=5, sticky="nws")

        weights = ttk.Entry(Frame_1, width=50)
        weights.grid(row=1, column=0, columnspan=5, padx=30, pady=5, sticky="w", ipadx=100, ipady=2)

        change_model_button = tk.Button(Frame_1, text="Change Model", command=lambda: self.change_model_1(Frame_2, weights,scale_conf_all,item_code))
        change_model_button.grid(row=1, column=1, padx=10, pady=5, sticky="w", ipadx=5, ipady=2)

        label_scale_conf_all = ttk.Label(Frame_1, text='2. Confidence all', font=('Segoe UI', 12))
        label_scale_conf_all.grid(row=2, column=0, padx=10, pady=5, sticky="nws")
        
        scale_conf_all = tk.Scale(Frame_1, from_=1, to=100, orient='horizontal', length=500)
        scale_conf_all.grid(row=3, column=0, padx=30, pady=5, sticky="nws")

        label_size_model = ttk.Label(Frame_1, text='2. Size Detection', font=('Segoe UI', 12))
        label_size_model.grid(row=4, column=0, padx=10, pady=5, sticky="nws")
        
        options = [468, 608, 832]
        size_model = ttk.Combobox(Frame_1, values=options)
        size_model.grid(row=5, column=0, padx=30, pady=5, sticky="nws")
        size_model.set(608)

        name_item_code = ttk.Label(Frame_1, text='3. Item code', font=('Segoe UI', 12))
        name_item_code.grid(row=6, column=0, padx=10, pady=5, sticky="nws")

        item_code = ttk.Entry(Frame_1, width=50)
        item_code.grid(row=7, column=0, columnspan=5, padx=30, pady=5, sticky="w", ipadx=20, ipady=2)

        save_data_to_database = ttk.Button(Frame_1, text='Apply', command=lambda: self.save_data_model_1(weights,scale_conf_all,item_code,model_name_labels, join, ok_vars, ng_vars, num_inputs,  wn_inputs, wx_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales))
        save_data_to_database.grid(row=8, column=0, padx=30, pady=5, sticky="w", ipadx=5, ipady=2)

        camera_frame_display = ttk.Label(Frame_1, text='4. Display Camera', font=('Segoe UI', 12))
        camera_frame_display.grid(row=9, column=0, padx=10, pady=5, sticky="nws")

        camera_frame = ttk.LabelFrame(Frame_1, text=f"Camera 1", width=500, height=500)
        camera_frame.grid(row=10, column=0, padx=30, pady=5, sticky="nws")

    def option_1_parameters(self,Frame_2,model1):
        
        global model_name_labels, join, ok_vars, ng_vars, num_inputs, wn_inputs, wx_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales
        
        def ng_selected(row_widgets):
            ok_checkbox_var = row_widgets[2].var
            ng_checkbox_var = row_widgets[3].var
            if ng_checkbox_var.get() == True:
                ok_checkbox_var.set(False)

        def ok_selected(row_widgets):
            ok_checkbox_var = row_widgets[2].var
            ng_checkbox_var = row_widgets[3].var
            if ok_checkbox_var.get() == True:
                ng_checkbox_var.set(False)
    
        model_name_labels = []
        join = []
        ok_vars = []
        ng_vars = []
        num_inputs = []
        wn_inputs = []
        wx_inputs = []
        hn_inputs = []
        hx_inputs = []
        plc_inputs = []
        conf_scales = []
        widgets = []

        for i1 in range(len(model1.names)):
            row_widgets = []

            model_name_label = tk.Label(Frame_2, text=f'{model1.names[i1]}', fg='black', font=('Segoe UI', 12), width=20, anchor='w')
            row_widgets.append(model_name_label)
            model_name_labels.append(model_name_label)

            join_checkbox_var = tk.BooleanVar()
            join_checkbox = tk.Checkbutton(Frame_2, variable=join_checkbox_var, onvalue=True, offvalue=False)
            join_checkbox.grid()
            join_checkbox.var = join_checkbox_var
            row_widgets.append(join_checkbox)
            join.append(join_checkbox_var)

            ok_checkbox_var = tk.BooleanVar()
            ok_checkbox = tk.Checkbutton(Frame_2, variable=ok_checkbox_var, onvalue=True, offvalue=False, command=lambda rw=row_widgets:ok_selected(rw))
            ok_checkbox.grid()
            ok_checkbox.var = ok_checkbox_var
            row_widgets.append(ok_checkbox)
            ok_vars.append(ok_checkbox_var)

            ng_checkbox_var = tk.BooleanVar()
            ng_checkbox = tk.Checkbutton(Frame_2, variable=ng_checkbox_var, onvalue=True, offvalue=False, command=lambda rw=row_widgets:ng_selected(rw))
            ng_checkbox.grid()
            ng_checkbox.var = ng_checkbox_var
            row_widgets.append(ng_checkbox)
            ng_vars.append(ng_checkbox_var)

            num_input = tk.Entry(Frame_2, width=7)
            num_input.insert(0, '1')
            row_widgets.append(num_input)
            num_inputs.append(num_input)

            wn_input = tk.Entry(Frame_2, width=7)
            wn_input.insert(0, '0')
            row_widgets.append(wn_input)
            wn_inputs.append(wn_input)

            wx_input = tk.Entry(Frame_2, width=7)
            wx_input.insert(0, '1600')
            row_widgets.append(wx_input)
            wx_inputs.append(wx_input)

            hn_input = tk.Entry(Frame_2, width=7)
            hn_input.insert(0, '0')
            row_widgets.append(hn_input)
            hn_inputs.append(hn_input)

            hx_input = tk.Entry(Frame_2, width=7)
            hx_input.insert(0, '1200')
            row_widgets.append(hx_input)
            hx_inputs.append(hx_input)

            plc_input = tk.Entry(Frame_2, width=7)
            plc_input.insert(0, '0')
            row_widgets.append(plc_input)
            plc_inputs.append(plc_input)

            conf_scale = tk.Scale(Frame_2, from_=1, to=100, orient='horizontal', length=280)
            row_widgets.append(conf_scale)
            conf_scales.append(conf_scale)

            widgets.append(row_widgets)

        for i, row in enumerate(widgets):
            for j, widget in enumerate(row):
                widget.grid(row=i+1, column=j, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)
        
    def processing_handle_image(self,input_image):
        label_remove = []
        size_model_all = int(size_model.get())
        conf_all = int(scale_conf_all.get())/100
        t1 = time.time()
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        results = model1(input_image,size_model_all,conf_all)
        table_results = results.pandas().xyxy[0]
        for item in range(len(table_results.index)):
            width = table_results['xmax'][item] - table_results['xmin'][item]
            height = table_results['ymax'][item] - table_results['ymin'][item]
            conf = table_results['confidence'][item] * 100
            label_name_tables = table_results['name'][item]
            # print(f'name label : {label_name_tables}--width : {width}-- height : {height}-- conf : {conf}')
            # print('-------------------------')
            for j1 in range(len(model1.names)):
                for i1 in range(len(model_name_labels)):
                    label_name = model_name_labels[i1].cget("text")
                    join_detect = join[i1].get()
                    OK_jont = ok_vars[i1].get()
                    NG_jont = ng_vars[i1].get()
                    num_labels = int(num_inputs[i1].get())
                    width_min = int(wn_inputs[i1].get())
                    width_max = int(wx_inputs[i1].get())
                    height_min = int(hn_inputs[i1].get())
                    height_max = int(hx_inputs[i1].get())
                    PLC_value = int(plc_inputs[i1].get())
                    cmpnt_conf = int(conf_scales[i1].get())
                    if label_name == model1.names[j1]:  
                        # print(f'{label_name}--{model1.names[j1]}------{join_detect}')
                        if join_detect == True:
                            if label_name_tables == model1.names[j1]:
                                # print(f'label_name_tables :{label_name_tables} & model1.names :{model1.names[j1]}--during models: {width}-{height}-{conf}- during settings {width_min}-{width_max}-{height_min}-{height_max}\{cmpnt_conf}')
                                if (width < int(width_min))\
                                    or (width > width_max)\
                                    or (height < height_min)\
                                    or (height > height_max)\
                                    or (conf < cmpnt_conf):                                  
                                    label_remove.append(item)
                        if join_detect == False:
                            if label_name_tables == model1.names[j1]:
                                label_remove.append(item)

        table_results.drop(index=label_remove, inplace=True)   
        names = list(table_results['name'])  
        print(names)               
        show1 = np.squeeze(results.render(label_remove))
        show1 = cv2.resize(show1, (800,800), interpolation=cv2.INTER_AREA)
        t2 = time.time() - t1
        time_processing = str(int(t2*1000)) + 'ms'
        return show1,time_processing
        

class PLC_Connection():

    def __init__(self,host,port):
        self.host = host
        self.port = port
      
    def connect_plc_keyence(self):
        try:
            soc.connect((self.host, self.port))
            return True
        except OSError:
            print("Can't connect to PLC")
            time.sleep(3)
            print("Reconnecting....")
            return False

    def run_plc_keyence(self):
        connected = False
        while connected == False:
            connected = self.connect_plc_keyence(self.host,self.port)
        print("connected") 

    def read_plc_keyence(self,data):
        a = 'RD '
        c = '\x0D'
        d = a+ data +c
        datasend = d.encode("UTF-8")
        soc.sendall(datasend)
        data = soc.recv(1024)
        datadeco = data.decode("UTF-8")
        data1 = int(datadeco)
        return data1

    def write_plc_keyence(self,register,data):
        a = 'WR '
        b = ' '
        c = '\x0D'
        d = a+register+b+str(data)+c
        datasend  = d.encode("UTF-8")
        soc.sendall(datasend)
        datares = soc.recv(1024)

    def connect_plc_omron(self):
        global fins_instance
        try:
            fins_instance = UDPFinsConnection()
            fins_instance.connect(self.host)
            fins_instance.dest_node_add=1
            fins_instance.srce_node_add=25
            return True
        except:
            print("Can't connect to PLC")
            for i in range(100000000):
                pass
            print("Reconnecting....")
            return False

    def run_plc_omron(self):
        connected = False
        while connected == False:
            connected = self.connect_plc_omron(self.host)
            print('connecting ....')
        print("connected plc") 

    def read_plc_omron(self,register):
        register = (register).to_bytes(2, byteorder='big') + b'\x00'
        read_var = fins_instance.memory_area_read(FinsPLCMemoryAreas().DATA_MEMORY_WORD,register)
        read_var = int.from_bytes(read_var[-2:], byteorder='big')  
        return read_var

    def write_plc_omron(self,register,data):
        register = (register).to_bytes(2, byteorder='big') + b'\x00'
        fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,register,b'\x00\x00',data)


class MySQL_Connection():

    def __init__(self,host,user,passwd,database):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.database = database

    def Connect_MySQLServer(self):
            db_connection = mysql.connector.connect(
            host= self.host,
            user=self.user, 
            passwd=self.passwd,
            database=self.database)                    
            cursor = db_connection.cursor()
            return cursor,db_connection

    def check_connection(self):
        _,db_connection = self.Connect_MySQLServer()
        try:
            db_connection.ping(reconnect=True, attempts=3, delay=5)
            return True
        except mysql.connector.Error as err:
            messagebox.showinfo("Notification", f"Error connecting to the database: {str(err)}")
            return False

    def reconnect(self):
        _,db_connection = self.Connect_MySQLServer()
        try:
            db_connection.reconnect(attempts=3, delay=5)
            cursor = db_connection.cursor()  
            return True
        except mysql.connector.Error as err:
            messagebox.showinfo("Notification", f"Failed to reconnect to the database: {str(err)}")
            return False
def removefile():
    directory1 = 'C:/Users/CCSX009/Documents/yolov5/test_image/camera1/*.jpg'
    directory2 = 'C:/Users/CCSX009/Documents/yolov5/test_image/camera2/*.jpg'
    chk1 = glob.glob(directory1)
    for f1 in chk1:
        os.remove(f1)
        print('already delete folder 1')
    chk2 = glob.glob(directory2)
    for f2 in chk2:
        os.remove(f2)
        print('already delete folder 2')

def main():
    window = tk.Tk()
    window.title("Utralytics Yolov5 ft Tkinter")
    window.state('zoomed')
    notebook = ttk.Notebook(window)
    notebook.pack(fill="both", expand=True)
    removefile()
        #tab 1
    Display = Main_Display()
    camera1_frame, camera2_frame = Display.Display_Camera(notebook)
    Display.update_images(window,camera1_frame, camera2_frame)

    #tab 2
    Camera_1_Settings = Model_Camera_1()
    Camera_1_Settings.Camera_1_Settings(notebook)

    window.mainloop()

if __name__ == "__main__":
    main()
