
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import glob
import torch
import mysql.connector
from tkinter import messagebox,simpledialog
import threading
# import stapipy as st
import numpy as np
import cv2
import socket
import os
import time
from PIL import Image,ImageTk
from PIL import Image
from yaml import load
import socket
import threading
from udp import UDPFinsConnection
from initialization import FinsPLCMemoryAreas
import threading
import time
from tkinter import ttk
from tkinter import ttk, messagebox
import threading
from subprocess import Popen, PIPE
import tkinter as tk
import shutil
from pathlib import Path
soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SCALE_X_CAM1 = 1#1280/2048
SCALE_Y_CAM1 = 1#960/1536


SCALE_X_CAM2 = 1#640/1440
SCALE_Y_CAM2 = 1#480/1080


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
        
    @staticmethod    
    def Connect_to_MySQLServer(host,user,passwd,database):
            db_connection = mysql.connector.connect(
            host= host,
            user=user, 
            passwd=passwd,
            database=database)                    
            cursor = db_connection.cursor()
            return cursor,db_connection
      

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

# from auto_training.labelImg.labelImg import main


class Model_Camera_1(PLC_Connection,MySQL_Connection):
    
    def __init__(self,settings_notebook):
        self.settings_notebook = settings_notebook
        self.name_table = 'test_model_cam1_model1'
        self.item_code_cfg = "EDFWTA"
        self.image_files = []
        self.current_image_index = -1
        self.state = 1
        self.password = "123"
        self.lockable_widgets = [] 
        self.lock_params = []
        self.cursor,self.db_connection = MySQL_Connection.Connect_to_MySQLServer(host="127.0.0.1",user="root1",passwd="987654321",database="model_1")  
        self.Camera_Settings()

    def connect_database(self):
        database = MySQL_Connection("127.0.0.1","root1","987654321","model_1")
        cursor, db_connection  = database.Connect_MySQLServer()
        check_connection = database.check_connection()
        reconnect = database.reconnect()
        return cursor,db_connection,check_connection,reconnect
    
    def save_params_model(self, weights, scale_conf_all, item_code, model_name_labels, join, ok_vars, ng_vars, num_inputs, wn_inputs, wx_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales):
        confirm_save_data = messagebox.askokcancel("Confirm", "Are you sure you want to save the data?")
        
        if confirm_save_data:
            cursor, db_connection, check_connection, reconnect = self.connect_database()
            if not check_connection and not reconnect:                
                return
            try:
                weight = weights.get()
                confidence_all = int(scale_conf_all.get())
                item_code_value = str(item_code.get())
                self.cursor.execute(f"DELETE FROM {self.name_table} WHERE item_code = %s", (item_code_value,))
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
                    query_sql = f"""
                    INSERT INTO {self.name_table}
                    (item_code, weight, confidence_all, label_name, join_detect, OK, NG, num_labels, width_min, width_max, 
                    height_min, height_max, PLC_value, cmpnt_conf)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    values = (item_code_value, weight, confidence_all, label_name, join_detect, OK_jont, NG_jont, num_labels, 
                            width_min, width_max, height_min, height_max, PLC_value, cmpnt_conf)
                    self.cursor.execute(query_sql, values)
                self.db_connection.commit()
                self.cursor.close()
                self.db_connection.close()
                messagebox.showinfo("Notification", "Saved parameters successfully!")
            except Exception as e:
                self.cursor.close()
                self.db_connection.close()
                messagebox.showinfo("Notification", f"Data saved failed! Error: {str(e)}")
        else:
            pass

    def save_params_model_bk(self, weights, scale_conf_all, item_code, model_name_labels, join, ok_vars, ng_vars, num_inputs, wn_inputs, wx_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales):
        confirm_save_data = messagebox.askokcancel("Confirm", "Are you sure you want to save the data?")
        
        if confirm_save_data:
            cursor, db_connection,check_connection,reconnect = self.connect_database()
            if not check_connection and not reconnect:
                print( check_connection, reconnect)
                return
            try:
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

                    cursor.execute(f"SELECT COUNT(*) FROM {self.name_table} WHERE item_code = %s AND label_name = %s", (item_code_value, label_name))
                    exists = cursor.fetchone()[0] > 0

                    if exists:
                        query_sql = f"""
                        UPDATE {self.name_table}
                        SET weight = %s, confidence_all = %s, label_name = %s, join_detect = %s, OK = %s, NG = %s, 
                            num_labels = %s, width_min = %s, width_max = %s, height_min = %s, height_max = %s, 
                            PLC_value = %s, cmpnt_conf = %s
                        WHERE item_code = %s AND label_name = %s
                        """
                        values = (weight, confidence_all, label_name, join_detect, OK_jont, NG_jont, num_labels, 
                                  width_min, width_max, height_min, height_max, PLC_value, cmpnt_conf, self.item_code_cfg, label_name)                      
                    else:
                        query_sql = f"""
                        INSERT INTO {self.name_table}
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

    def load_data_model(self):
        cursor, db_connection,_,_ = self.connect_database()
        cursor.execute(f"SELECT * FROM {self.name_table} WHERE item_code = %s", (self.item_code_cfg,))
        records = cursor.fetchall()
        cursor.close()
        db_connection.close()
        if records:
            first_record = records[0]
            load_item_code = first_record[1]
            load_path_weight = first_record[2]
            load_confidence_all_scale = first_record[3]
        return records,load_path_weight,load_item_code,load_confidence_all_scale

    def load_parameters_model(self,model1,records,load_path_weight,load_item_code,load_confidence_all_scale):
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
        
    def change_model(self,Frame_2,weights):
        global model1
        selected_file = filedialog.askopenfilename(title="Choose a file", filetypes=[("Model Files", "*.pt")])
        if selected_file:
            weights.delete(0,tk.END)
            weights.insert(0,selected_file)
            model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5','custom', path=selected_file, source='local', force_reload=False)
            for widget in Frame_2.grid_slaves():
                widget.grid_forget()
            self.option_layout_parameters(Frame_2,model1)  
        else:
            messagebox.showinfo("Notification","Please select the correct training file!")

    def load_params_child(self,weights,item_code):
        weight = weights.get()
        item_code_value = str(item_code.get())
        cursor, db_connection,_,_ = self.connect_database()
        cursor.execute("SELECT * FROM test_model_cam1_model1 WHERE item_code = %s", (item_code_value,))
        cursor.close()
        db_connection.close()
        records = cursor.fetchall()
        model = torch.hub.load('C:/Users/CCSX009/Documents/yolov5','custom', path=weight, source='local', force_reload=False)
        return records,model

    def load_parameters_from_weight(self,weights, item_code):
        confirm_load_parameters = messagebox.askokcancel("Confirm", "Are you sure you want to load the parameters?")
        # _,_ = self.load_params_child(weights, item_code)
        if confirm_load_parameters :
            try: 
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

                messagebox.showinfo("Notification", "Loaded parameters successfully!")
            except Exception as e:
                messagebox.showinfo("Notification", f"Parameters Loaded failed! Error: {str(e)}")
        else:
            pass

    def handle_image(self,img1_orgin, width, height,camera_frame):
        for widget in camera_frame.winfo_children():
            widget.destroy()
        image_result,time_processing,results_detect,label_ng = self.processing_handle_image_local(img1_orgin, width, height,cls=make_cls_var.get())
        img_pil = Image.fromarray(image_result)
        photo = ImageTk.PhotoImage(img_pil)
        canvas = tk.Canvas(camera_frame, width=width, height=height)
        canvas.grid(row=2, column=0, padx=10, pady=10, sticky='ew')
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo
        canvas.create_text(10, 10, anchor=tk.NW, text=f'Time: {time_processing}', fill='black', font=('Segoe UI', 20))
        canvas.create_text(10, 40, anchor=tk.NW, text=f'Result: {results_detect}', fill='green' if results_detect == 'OK' else 'red', font=('Segoe UI', 20))
        if not label_ng:
            canvas.create_text(10, 70, anchor=tk.NW, text=f'No Label', fill='green', font=('Segoe UI', 20))
        else:
            label_ng = ','.join(label_ng)
            canvas.create_text(10, 70, anchor=tk.NW, text=f'Label: {label_ng}', fill='red', font=('Segoe UI', 20))
        return results_detect

    def detect_single_img(self, camera_frame):
        selected_file = filedialog.askopenfilename(title="Choose a file", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if selected_file:
            for widget in camera_frame.winfo_children():
                widget.destroy()
            width = 480
            height = 450
            self.handle_image(selected_file, width, height,camera_frame)
        else: 
            pass
           
    def detect_multi_img(self,camera_frame):
        selected_folder = filedialog.askdirectory(title="Choose a folder")
        if selected_folder:
            self.image_files = [os.path.join(selected_folder, f) for f in os.listdir(selected_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.current_image_index = 0
            if self.image_files:
                for widget in camera_frame.winfo_children():
                    widget.destroy()
                self.show_image(self.current_image_index,camera_frame)
            else:
                messagebox.showinfo("No Images", "The selected folder contains no images.")
        else:
            pass

    def show_image(self,index,camera_frame):  
        width = 480
        height = 450
        image_path = self.image_files[index]
        self.handle_image(image_path, width, height,camera_frame)

    def detect_next_img(self,camera_frame):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_image(self.current_image_index,camera_frame)
        else:
            messagebox.showinfo("End of Images", "No more images in the folder.")

    def detect_previos_img(self,camera_frame):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image(self.current_image_index,camera_frame)
        else:
            messagebox.showinfo("Start of Images", "This is the first image in the folder.")

    def detect_auto(self, camera_frame):
        global selected_folder_original
        selected_folder_original = filedialog.askdirectory(title="Choose a folder")
        if selected_folder_original: 
            selected_folder = glob.glob(selected_folder_original + '/*.jpg')
            if not selected_folder:
                pass
            self.image_index = 0
            self.selected_folder_detect_auto = selected_folder
            self.camera_frame = camera_frame
            self.image_path_mks_cls = []
            self.results_detect = None
            def process_next_image():
                if self.image_index < len(self.selected_folder_detect_auto):
                    self.image_path_mks_cls = self.selected_folder_detect_auto[self.image_index]
                    width = 480
                    height = 450
                    self.results_detect = self.handle_image(self.image_path_mks_cls, width, height,camera_frame)
                    self.image_index += 1
                    self.camera_frame.after(500, process_next_image)
                else:
                    print("Processing auto detect complete")
            process_next_image()
        else:
            pass

    
    def logging(self, folder_ok, folder_ng, logging_ok_checkbox_var, logging_ng_checkbox_var, camera_frame, percent_entry,logging_frame):
        selected_folder_original = filedialog.askdirectory(title="Choose a folder")
        selected_folder = glob.glob(os.path.join(selected_folder_original,'*.jpg'))
        width = 480
        height = 450
        self.image_index = 0
        self.selected_folder_logging = selected_folder
        self.logging_frame = logging_frame
        self.percent_entry = percent_entry
        total_images = len(self.selected_folder_logging)
        if not self.selected_folder_logging:
            return
        
        def update_progress(current_index, total_images):
            percent_value = int((current_index + 1) / total_images * 100)
            self.percent_entry.delete(0, tk.END)
            self.percent_entry.insert(0, f"{percent_value}%")

        def process_images():
            for img in self.selected_folder_logging:
                basename = os.path.basename(img)
                if self.image_index < total_images:
                    results_detect = self.handle_image(img, width, height, camera_frame)
                    if results_detect == 'OK':
                        if logging_ok_checkbox_var.get():
                            shutil.move(img, os.path.join(folder_ok.get(), basename))
                    else:
                        if logging_ng_checkbox_var.get():
                            shutil.move(img, os.path.join(folder_ng.get(), basename))
                    self.logging_frame.after(10, update_progress, self.image_index, total_images)
                    self.image_index += 1
                else:
                    messagebox.showinfo("End of Images", "No more images in the folder.")
                    break
            self.percent_entry.delete(0, tk.END)
            self.percent_entry.insert(0, "0%")
        threading.Thread(target=process_images).start()
        
             
    def toggle_state_layout_model(self): 
        if self.state == 1:
            password = simpledialog.askstring("Administrator", "Enter password:", show="*")
            if password == self.password:
                self.state = 0
                permisson_btn.config(text="Lock")
                self.toggle_widgets_state("normal")
            else:
                messagebox.showerror("Error", "Incorrect password!")
        else:
            self.state = 1
            permisson_btn.config(text="Unlock")
            self.toggle_widgets_state("disabled")

    def toggle_widgets_state(self,state):
        for widget in self.lockable_widgets:
            widget.config(state=state)

    def toggle_state_option_layout_parameters(self):
        for widget in self.lock_params:
            widget.config(state='disabled')

    def pick_folder_ok(self,folder_ok):
        file_path = filedialog.askdirectory(title="Choose a folder")
        if file_path:
            folder_ok.delete(0,tk.END)
            folder_ok.insert(0,file_path)
           
    def pick_folder_ng(self,folder_ng):
        file_path = filedialog.askdirectory(title="Choose a folder")
        if file_path:
            folder_ng.delete(0,tk.END)
            folder_ng.insert(0,file_path)
                
    def Camera_Settings(self):
        global model1,records
        records,load_path_weight,load_item_code,load_confidence_all_scale = self.load_data_model()
        
        model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5','custom', path=load_path_weight, source='local', force_reload=False)
        
        camera_settings_tab = ttk.Frame(self.settings_notebook)
        self.settings_notebook.add(camera_settings_tab, text="Camera 1")

        canvas = tk.Canvas(camera_settings_tab)
        scrollbar = ttk.Scrollbar(camera_settings_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        camera_settings_tab.grid_columnconfigure(0, weight=1)
        camera_settings_tab.grid_rowconfigure(0, weight=1)

        frame_width = 1500
        frame_height = 2000

        Frame_1 = ttk.LabelFrame(scrollable_frame, text="Frame 1", width=frame_width, height=frame_height)
        Frame_2 = ttk.LabelFrame(scrollable_frame, text="Frame 2", width=frame_width, height=frame_height)

        Frame_1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  
        Frame_2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.option_layout_models(Frame_1,Frame_2)
        self.option_layout_parameters(Frame_2,model1)
        self.load_parameters_model(model1,records,load_path_weight,load_item_code,load_confidence_all_scale)
        self.toggle_state_option_layout_parameters()
        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=1)
        scrollable_frame.grid_rowconfigure(0, weight=1)
        scrollable_frame.grid_rowconfigure(1, weight=1)

    def option_layout_models(self, Frame_1, Frame_2):
        
        global weights, scale_conf_all, size_model, item_code,make_cls_var,permisson_btn

        ttk.Label(Frame_1, text='1. File train detect model', font=('Segoe UI', 12)).grid(column=0, row=0, padx=10, pady=5, sticky="nws")

        weights = ttk.Entry(Frame_1, width=60)
        weights.grid(row=1, column=0, columnspan=5, padx=(30, 5), pady=5, sticky="w", ipadx=20, ipady=2)

        button_frame = ttk.Frame(Frame_1)
        button_frame.grid(row=2, column=0, columnspan=2, padx=(30, 30), pady=5, sticky="w")

        change_model_button = tk.Button(button_frame, text="Change Model", command=lambda: self.change_model(Frame_2, weights))
        change_model_button.grid(row=0, column=0, padx=(0, 8), pady=5, sticky="w", ipadx=5, ipady=2)
        change_model_button.config(state="disabled")
        self.lockable_widgets.append(change_model_button)

        load_parameters = tk.Button(button_frame, text="Load Parameters", command=lambda: self.load_parameters_from_weight(weights, item_code))
        load_parameters.grid(row=0, column=1, padx=(0, 8), pady=5, sticky="w", ipadx=5, ipady=2)
        load_parameters.config(state="disabled")
        self.lockable_widgets.append(load_parameters)

        custom_para = tk.Button(button_frame, text="Custom Parameters")
        custom_para.grid(row=0, column=2, padx=(0, 8), pady=5, sticky="w", ipadx=5, ipady=2)
        custom_para.config(state="disabled")
        self.lockable_widgets.append(custom_para)

        permisson_btn = tk.Button(button_frame, text="Unlock", command=lambda: self.toggle_state_layout_model())
        permisson_btn.grid(row=0, column=3, padx=(0, 8), pady=5, sticky="w", ipadx=5, ipady=2)

        label_scale_conf_all = ttk.Label(Frame_1, text='2. Confidence Threshold', font=('Segoe UI', 12))
        label_scale_conf_all.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="nws")

        scale_conf_all = tk.Scale(Frame_1, from_=1, to=100, orient='horizontal', length=400)
        scale_conf_all.grid(row=4, column=0, columnspan=2, padx=30, pady=5, sticky="nws")
        scale_conf_all.config(state="disabled")
        self.lockable_widgets.append(scale_conf_all)

        label_size_model = ttk.Label(Frame_1, text='2. Size Model', font=('Segoe UI', 12))
        label_size_model.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="nws")

        options = [468, 608, 832]
        size_model = ttk.Combobox(Frame_1, values=options, width=7)
        size_model.grid(row=6, column=0, columnspan=2, padx=30, pady=5, sticky="nws", ipadx=5, ipady=2)
        size_model.set(608)
        size_model.config(state="disabled")
        self.lockable_widgets.append(size_model)

        name_item_code = ttk.Label(Frame_1, text='3. Item Code', font=('Segoe UI', 12))
        name_item_code.grid(row=7, column=0, columnspan=2, padx=10, pady=5, sticky="nws")

        item_code = ttk.Entry(Frame_1, width=10)
        item_code.grid(row=8, column=0, columnspan=2, padx=30, pady=5, sticky="w", ipadx=5, ipady=2)
        self.lockable_widgets.append(item_code)

        save_data_to_database = ttk.Button(Frame_1, text='Apply', command=lambda: self.save_params_model(weights, scale_conf_all, item_code, model_name_labels, join, ok_vars, ng_vars, num_inputs, wn_inputs, wx_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales))
        save_data_to_database.grid(row=9, column=0, columnspan=2, padx=30, pady=5, sticky="w", ipadx=5, ipady=2)
        save_data_to_database.config(state="disabled")
        self.lockable_widgets.append(save_data_to_database)

        camera_frame_display = ttk.Label(Frame_1, text='4. Modify Image', font=('Segoe UI', 12))
        camera_frame_display.grid(row=10, column=0, columnspan=2, padx=10, pady=5, sticky="nws")

        camera_frame = ttk.LabelFrame(Frame_1, text=f"Camera 1", width=500, height=500)
        camera_frame.grid(row=11, column=0, columnspan=2, padx=30, pady=5, sticky="nws")

        camera_custom_setup = ttk.Frame(Frame_1)
        camera_custom_setup.grid(row=12, column=0, columnspan=2, padx=(30, 30), pady=5, sticky="w") 

        single_img = tk.Button(camera_custom_setup, text="Only Image", command=lambda: self.detect_single_img(camera_frame))
        single_img.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        single_img.config(state="disabled")
        self.lockable_widgets.append(single_img)

        multi_img = tk.Button(camera_custom_setup, text="Multi Image", command=lambda: self.detect_multi_img(camera_frame))
        multi_img.grid(row=0, column=1, padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        multi_img.config(state="disabled")
        self.lockable_widgets.append(multi_img)

        previos_img = tk.Button(camera_custom_setup, text="Prev...", command=lambda: self.detect_previos_img(camera_frame))
        previos_img.grid(row=0, column=2, padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        previos_img.config(state="disabled")
        self.lockable_widgets.append(previos_img)
        
        next_img = tk.Button(camera_custom_setup, text="Next...", command=lambda: self.detect_next_img(camera_frame))
        next_img.grid(row=0, column=3, padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        next_img.config(state="disabled")
        self.lockable_widgets.append(next_img)

        auto_detect = tk.Button(camera_custom_setup, text="Auto Detect", command=lambda: self.detect_auto(camera_frame))
        auto_detect.grid(row=0, column=4, padx=(0, 10), pady=5, sticky="w", ipadx=7, ipady=2)
        auto_detect.config(state="disabled")
        self.lockable_widgets.append(auto_detect)

        make_cls_var = tk.BooleanVar()
        make_cls = tk.Checkbutton(camera_custom_setup,text='Make class',variable=make_cls_var, onvalue=True, offvalue=False,anchor='w')
        make_cls.grid(row=0, column=5, padx=(0, 10), pady=5, sticky="w", ipadx=2, ipady=2)
        make_cls.config(state="disabled")
        self.lockable_widgets.append(make_cls)
        make_cls.var = make_cls_var

        logging_frame = ttk.Frame(Frame_1)
        logging_frame.grid(row=13, column=0, columnspan=2, padx=(30, 30), pady=10, sticky="w") 

        logging = tk.Button(logging_frame, text="Logging Image", command=lambda: self.logging(folder_ok,folder_ng,logging_ok_checkbox_var,logging_ng_checkbox_var,camera_frame,percent_entry,logging_frame))
        logging.grid(row=0, column=0, padx=(0,10), pady=5, sticky="w", ipadx=7, ipady=2)
        logging.config(state="disabled")
        self.lockable_widgets.append(logging)


        default_text_var = tk.StringVar()
        default_text_var.set("0%")
        percent_entry = ttk.Entry(logging_frame,textvariable=default_text_var,width=5)
        percent_entry.grid(row=0, column=1, padx=(0, 10), pady=3, sticky="w", ipadx=5, ipady=2)

        logging_ok_checkbox_var = tk.BooleanVar()
        logging_ok_checkbox = tk.Checkbutton(logging_frame,text='OK', variable=logging_ok_checkbox_var, onvalue=True, offvalue=False)
        logging_ok_checkbox.grid(row=1, column=2, padx=(0,10), pady=5, sticky="w", ipadx=7, ipady=2)
        logging_ok_checkbox.var = logging_ok_checkbox_var
        self.lock_params.append(logging_ok_checkbox)
        self.lockable_widgets.append(logging_ok_checkbox)

        logging_ng_checkbox_var = tk.BooleanVar()
        logging_ng_checkbox = tk.Checkbutton(logging_frame,text='NG', variable=logging_ng_checkbox_var, onvalue=True, offvalue=False)
        logging_ng_checkbox.grid(row=2, column=2, padx=(0,10), pady=5, sticky="w", ipadx=7, ipady=2)
        logging_ng_checkbox.var = logging_ng_checkbox_var
        self.lock_params.append(logging_ng_checkbox)
        self.lockable_widgets.append(logging_ng_checkbox)

        folder_ok = ttk.Entry(logging_frame, width=45)
        folder_ok.grid(row=1, column=0, padx=(0, 10), pady=3, sticky="w", ipadx=15, ipady=2)

        folder_ng = ttk.Entry(logging_frame ,width=45)
        folder_ng.grid(row=2, column=0, padx=(0, 10), pady=3, sticky="w", ipadx=15, ipady=2)

        folder_ok_button = tk.Button(logging_frame, text="Folder OK", command=lambda: self.pick_folder_ok(folder_ok))
        folder_ok_button.grid(row=1, column=1, padx=(0, 8), pady=3, sticky="w", ipadx=5, ipady=2)
        folder_ok_button.config(state="disabled")
        self.lockable_widgets.append(folder_ok_button)

        folder_ng_button = tk.Button(logging_frame, text="Folder NG", command=lambda: self.pick_folder_ng(folder_ng))
        folder_ng_button.grid(row=2, column=1, padx=(0, 8), pady=3, sticky="w", ipadx=5, ipady=2)
        folder_ng_button.config(state="disabled")
        self.lockable_widgets.append(folder_ng_button)

    def option_layout_parameters(self,Frame_2,model1):
        
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

        label = tk.Label(Frame_2, text='LABEL', fg='red', font=('Ubuntu', 12), width=12, anchor='center', relief="groove", borderwidth=2)
        label.grid(row=0, column=0, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        joint_detect = tk.Label(Frame_2, text='JOIN', fg='red', font=('Ubuntu', 12), anchor='center', relief="groove", borderwidth=2)
        joint_detect.grid(row=0, column=1, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        ok_joint = tk.Label(Frame_2, text='OK', fg='red', font=('Ubuntu', 12), anchor='center', relief="groove", borderwidth=2)
        ok_joint.grid(row=0, column=2, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        ng_joint = tk.Label(Frame_2, text='NG', fg='red', font=('Ubuntu', 12), anchor='center', relief="groove", borderwidth=2)
        ng_joint.grid(row=0, column=3, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        num_lb = tk.Label(Frame_2, text='NUM', fg='red', font=('Ubuntu', 12), width=7, anchor='center', relief="groove", borderwidth=2)
        num_lb.grid(row=0, column=4, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        width_n = tk.Label(Frame_2, text='W_N', fg='red', font=('Ubuntu', 12), width=7, anchor='center', relief="groove", borderwidth=2)
        width_n.grid(row=0, column=5, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        wight_x= tk.Label(Frame_2, text='W_X', fg='red', font=('Ubuntu', 12), width=7, anchor='center', relief="groove", borderwidth=2)
        wight_x.grid(row=0, column=6, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        height_n = tk.Label(Frame_2, text='H_N', fg='red', font=('Ubuntu', 12), width=7, anchor='center', relief="groove", borderwidth=2)
        height_n.grid(row=0, column=7, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        height_x = tk.Label(Frame_2, text='H_X', fg='red', font=('Ubuntu', 12), width=7, anchor='center', relief="groove", borderwidth=2)
        height_x.grid(row=0, column=8, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        plc_var = tk.Label(Frame_2, text='PLC', fg='red', font=('Ubuntu', 12), width=7, anchor='center', relief="groove", borderwidth=2)
        plc_var.grid(row=0, column=9, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        conf = tk.Label(Frame_2, text='CONFIDENCE THRESHOLD', fg='red', font=('Ubuntu', 12), width=25, anchor='center', relief="groove", borderwidth=2)
        conf.grid(row=0, column=10, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        for i1 in range(len(model1.names)):
            row_widgets = []

            model_name_label = tk.Label(Frame_2, text=f'{model1.names[i1]}', fg='black', font=('Segoe UI', 12), width=15, anchor='w')
            row_widgets.append(model_name_label)
            model_name_labels.append(model_name_label)

            join_checkbox_var = tk.BooleanVar()
            join_checkbox = tk.Checkbutton(Frame_2, variable=join_checkbox_var, onvalue=True, offvalue=False,anchor='w')
            join_checkbox.grid()
            join_checkbox.var = join_checkbox_var
            row_widgets.append(join_checkbox)
            join.append(join_checkbox_var)
            self.lock_params.append(join_checkbox)
            self.lockable_widgets.append(join_checkbox)

            ok_checkbox_var = tk.BooleanVar()
            ok_checkbox = tk.Checkbutton(Frame_2, variable=ok_checkbox_var, onvalue=True, offvalue=False, command=lambda rw=row_widgets:ok_selected(rw), anchor='w')
            ok_checkbox.grid()
            ok_checkbox.var = ok_checkbox_var
            row_widgets.append(ok_checkbox)
            ok_vars.append(ok_checkbox_var)
            self.lock_params.append(ok_checkbox)
            self.lockable_widgets.append(ok_checkbox)

            ng_checkbox_var = tk.BooleanVar()
            ng_checkbox = tk.Checkbutton(Frame_2, variable=ng_checkbox_var, onvalue=True, offvalue=False, command=lambda rw=row_widgets:ng_selected(rw), anchor='w')
            ng_checkbox.grid()
            ng_checkbox.var = ng_checkbox_var
            row_widgets.append(ng_checkbox)
            ng_vars.append(ng_checkbox_var)
            self.lock_params.append(ng_checkbox)
            self.lockable_widgets.append(ng_checkbox)

            num_input = tk.Entry(Frame_2, width=7,)
            num_input.insert(0, '1')
            row_widgets.append(num_input)
            num_inputs.append(num_input)
            self.lock_params.append(num_input)
            self.lockable_widgets.append(num_input)

            wn_input = tk.Entry(Frame_2, width=7, )
            wn_input.insert(0, '0')
            row_widgets.append(wn_input)
            wn_inputs.append(wn_input)
            self.lock_params.append(wn_input)
            self.lockable_widgets.append(wn_input)

            wx_input = tk.Entry(Frame_2, width=7, )
            wx_input.insert(0, '1600')
            row_widgets.append(wx_input)
            wx_inputs.append(wx_input)
            self.lock_params.append(wx_input)
            self.lockable_widgets.append(wx_input)

            hn_input = tk.Entry(Frame_2, width=7, )
            hn_input.insert(0, '0')
            row_widgets.append(hn_input)
            hn_inputs.append(hn_input)
            self.lock_params.append(hn_input)
            self.lockable_widgets.append(hn_input)

            hx_input = tk.Entry(Frame_2, width=7, )
            hx_input.insert(0, '1200')
            row_widgets.append(hx_input)
            hx_inputs.append(hx_input)
            self.lock_params.append(hx_input)
            self.lockable_widgets.append(hx_input)

            plc_input = tk.Entry(Frame_2, width=7,)
            plc_input.insert(0, '0')
            row_widgets.append(plc_input)
            plc_inputs.append(plc_input)
            self.lock_params.append(plc_input)
            self.lockable_widgets.append(plc_input)

            conf_scale = tk.Scale(Frame_2, from_=1, to=100, orient='horizontal', length=250)
            row_widgets.append(conf_scale)
            conf_scales.append(conf_scale)
            self.lock_params.append(conf_scale)
            self.lockable_widgets.append(conf_scale)

            widgets.append(row_widgets)

        for i, row in enumerate(widgets):
            for j, widget in enumerate(row):
                widget.grid(row=i+1, column=j, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

    @staticmethod
    def processing_handle_image_customize(input_image,width,height):
        label_remove = []
        size_model_all = int(size_model.get())
        conf_all = int(scale_conf_all.get())/100
        t1 = time.time()
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        results = model1(input_image,size_model_all,conf_all)
        table_results = results.pandas().xyxy[0]
        model_names = model1.names
        model_settings = []
        for i1 in range(len(model_name_labels)):
            model_settings.append({
                'label_name': model_name_labels[i1].cget("text"),
                'join_detect': join[i1].get(),
                'OK_jont': ok_vars[i1].get(),
                'NG_jont': ng_vars[i1].get(),
                'num_labels': int(num_inputs[i1].get()),
                'width_min': int(wn_inputs[i1].get()),
                'width_max': int(wx_inputs[i1].get()),
                'height_min': int(hn_inputs[i1].get()),
                'height_max': int(hx_inputs[i1].get()),
                'PLC_value': int(plc_inputs[i1].get()),
                'cmpnt_conf': int(conf_scales[i1].get()),
            })
        for i in range(len(table_results.index)):
            width_result = table_results['xmax'][i] - table_results['xmin'][i]
            height_result = table_results['ymax'][i] - table_results['ymin'][i]
            conf_result = table_results['confidence'][i] * 100
            label_name_tables_result = table_results['name'][i]
            for i1 in range(len(model_names)):
                for setting in model_settings:
                    if label_name_tables_result == model_names[i1] == setting['label_name']:
                        if setting['join_detect']:
                            if width_result < setting['width_min'] or width_result > setting['width_max'] \
                                    or height_result < setting['height_min'] or height_result > setting['height_max'] \
                                    or conf_result < setting['cmpnt_conf']:
                                label_remove.append(i)
                        else:
                            label_remove.append(i)
        table_results.drop(index=label_remove, inplace=True)   
        name_rest = list(table_results['name'])                
        results_detect = 'ERROR'
        ok_variable = False
        list_label_ng = []
        for i1 in range(len(model_names)):
            for j1 in model_settings:
                if model_names[i1] == j1['label_name']:
                    if j1['join_detect']:
                        if j1['OK_jont']:
                            number_of_labels = name_rest.count(model_names[i1])
                            if number_of_labels != j1['num_labels']:
                                results_detect = 'NG'
                                ok_variable = True
                                list_label_ng.append(model_names[i1])
                                # self.write_plc_keyence(setting['PLC_value'], 1)
                        if j1['NG_jont']:
                            if j1['label_name'] in name_rest:
                                results_detect = 'NG'
                                ok_variable = True
                                list_label_ng.append(model_names[i1])
        if not ok_variable:
            results_detect = 'OK'
            # self.write_plc_keyence(PLC_value,2)
        show_img = np.squeeze(results.render(label_remove))
        show_img = cv2.resize(show_img, (width,height), interpolation=cv2.INTER_AREA)
        t2 = time.time() - t1
        time_processing = str(int(t2*1000)) + 'ms'
        return show_img,time_processing,results_detect,list_label_ng
    
    def _make_cls(self,image_path_mks_cls,results,model_settings):  
        with open(image_path_mks_cls[:-3] + 'txt', "a") as file:
            for params in results.xywhn:
                params = params.tolist()
                for item in range(len(params)):
                    param = params[item]
                    param = [round(i,6) for i in param]
                    number_label = int(param[5])
                    conf_result = float(param[4])
                    width_result = float(param[2])*1200
                    height_result = float(param[3])*1600
                    for setting in model_settings:
                        if results.names[int(number_label)] == setting['label_name']:
                            if setting['join_detect']:
                                if width_result < setting['width_min'] or width_result > setting['width_max'] \
                                        or height_result < setting['height_min'] or height_result > setting['height_max'] \
                                        or conf_result < setting['cmpnt_conf']: 
                                    formatted_values = ["{:.6f}".format(value) for value in param[:4]]
                                    output_line = "{} {}\n".format(str(number_label),' '.join(formatted_values))
                                    file.write(output_line)
                                    
        path = Path(image_path_mks_cls).parent
        path = os.path.join(path,'classes.txt')
        with open(path, "w") as file:
            for i1 in range(len(results.names)):
                file.write(str(results.names[i1])+'\n')
      
    def classify_imgs(self):
        pass    

    def processing_handle_image_local(self,input_image_original,width,height,cls=False):
        label_remove = []
        size_model_all = int(size_model.get())
        conf_all = int(scale_conf_all.get())/100
        t1 = time.time()
        results = model1(input_image_original,size_model_all,conf_all)
        table_results = results.pandas().xyxy[0]
        model_names = model1.names
        model_settings = []
        for i1 in range(len(model_name_labels)):
            model_settings.append({
                'label_name': model_name_labels[i1].cget("text"),
                'join_detect': join[i1].get(),
                'OK_jont': ok_vars[i1].get(),
                'NG_jont': ng_vars[i1].get(),
                'num_labels': int(num_inputs[i1].get()),
                'width_min': int(wn_inputs[i1].get()),
                'width_max': int(wx_inputs[i1].get()),
                'height_min': int(hn_inputs[i1].get()),
                'height_max': int(hx_inputs[i1].get()),
                'PLC_value': int(plc_inputs[i1].get()),
                'cmpnt_conf': int(conf_scales[i1].get()),
            })
        for i in range(len(table_results.index)):
            width_result = table_results['xmax'][i] - table_results['xmin'][i]
            height_result = table_results['ymax'][i] - table_results['ymin'][i]
            conf_result = table_results['confidence'][i] * 100
            label_name_tables_result = table_results['name'][i]
            for i1 in range(len(model_names)):
                for setting in model_settings:
                    if label_name_tables_result == model_names[i1] == setting['label_name']:
                        if setting['join_detect']:
                            if width_result < setting['width_min'] or width_result > setting['width_max'] \
                                    or height_result < setting['height_min'] or height_result > setting['height_max'] \
                                    or conf_result < setting['cmpnt_conf']:
                                label_remove.append(i)
                        else:
                            label_remove.append(i)
        table_results.drop(index=label_remove, inplace=True)   
        name_rest = list(table_results['name'])                
        results_detect = 'ERROR'
        ok_variable = False
        list_label_ng = []
        for i1 in range(len(model_names)):
            for j1 in model_settings:
                if model_names[i1] == j1['label_name']:
                    if j1['join_detect']:
                        if j1['OK_jont']:
                            number_of_labels = name_rest.count(model_names[i1])
                            if number_of_labels != j1['num_labels']:
                                results_detect = 'NG'
                                ok_variable = True
                                list_label_ng.append(model_names[i1])
                        if j1['NG_jont']:
                            if j1['label_name'] in name_rest:
                                results_detect = 'NG'
                                ok_variable = True
                                list_label_ng.append(model_names[i1])
        if not ok_variable:
            results_detect = 'OK'
        show_img = np.squeeze(results.render(label_remove))
        show_img = cv2.resize(show_img, (width,height), interpolation=cv2.INTER_AREA)
        t2 = time.time() - t1
        time_processing = str(int(t2*1000)) + 'ms'
        if cls:
            self._make_cls(input_image_original,results,model_settings)
        return show_img,time_processing,results_detect,list_label_ng


class Model_Camera_2(PLC_Connection,MySQL_Connection):
    
    def __init__(self,settings_notebook):
        self.settings_notebook = settings_notebook
        self.name_table = 'test_model_cam1_model1'
        self.item_code_cfg = "EDFWTA"
        self.image_files = []
        self.current_image_index = -1
        self.state = 1
        self.password = "123"
        self.lockable_widgets = [] 
        self.lock_params = []
        self.cursor,self.db_connection = MySQL_Connection.Connect_to_MySQLServer(host="127.0.0.1",user="root1",passwd="987654321",database="model_1")  
        self.Camera_Settings()

    def connect_database(self):
        database = MySQL_Connection("127.0.0.1","root1","987654321","model_1")
        cursor, db_connection  = database.Connect_MySQLServer()
        check_connection = database.check_connection()
        reconnect = database.reconnect()
        return cursor,db_connection,check_connection,reconnect
    
    def save_params_model(self, weights, scale_conf_all, item_code, model_name_labels, join, ok_vars, ng_vars, num_inputs, wn_inputs, wx_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales):
        confirm_save_data = messagebox.askokcancel("Confirm", "Are you sure you want to save the data?")
        
        if confirm_save_data:
            cursor, db_connection, check_connection, reconnect = self.connect_database()
            if not check_connection and not reconnect:                
                return
            try:
                weight = weights.get()
                confidence_all = int(scale_conf_all.get())
                item_code_value = str(item_code.get())
                self.cursor.execute(f"DELETE FROM {self.name_table} WHERE item_code = %s", (item_code_value,))
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
                    query_sql = f"""
                    INSERT INTO {self.name_table}
                    (item_code, weight, confidence_all, label_name, join_detect, OK, NG, num_labels, width_min, width_max, 
                    height_min, height_max, PLC_value, cmpnt_conf)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    values = (item_code_value, weight, confidence_all, label_name, join_detect, OK_jont, NG_jont, num_labels, 
                            width_min, width_max, height_min, height_max, PLC_value, cmpnt_conf)
                    self.cursor.execute(query_sql, values)
                self.db_connection.commit()
                self.cursor.close()
                self.db_connection.close()
                messagebox.showinfo("Notification", "Saved parameters successfully!")
            except Exception as e:
                self.cursor.close()
                self.db_connection.close()
                messagebox.showinfo("Notification", f"Data saved failed! Error: {str(e)}")
        else:
            pass

    def save_params_model_bk(self, weights, scale_conf_all, item_code, model_name_labels, join, ok_vars, ng_vars, num_inputs, wn_inputs, wx_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales):
        confirm_save_data = messagebox.askokcancel("Confirm", "Are you sure you want to save the data?")
        
        if confirm_save_data:
            cursor, db_connection,check_connection,reconnect = self.connect_database()
            if not check_connection and not reconnect:
                print( check_connection, reconnect)
                return
            try:
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

                    cursor.execute(f"SELECT COUNT(*) FROM {self.name_table} WHERE item_code = %s AND label_name = %s", (item_code_value, label_name))
                    exists = cursor.fetchone()[0] > 0

                    if exists:
                        query_sql = f"""
                        UPDATE {self.name_table}
                        SET weight = %s, confidence_all = %s, label_name = %s, join_detect = %s, OK = %s, NG = %s, 
                            num_labels = %s, width_min = %s, width_max = %s, height_min = %s, height_max = %s, 
                            PLC_value = %s, cmpnt_conf = %s
                        WHERE item_code = %s AND label_name = %s
                        """
                        values = (weight, confidence_all, label_name, join_detect, OK_jont, NG_jont, num_labels, 
                                  width_min, width_max, height_min, height_max, PLC_value, cmpnt_conf, self.item_code_cfg, label_name)                      
                    else:
                        query_sql = f"""
                        INSERT INTO {self.name_table}
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

    def load_data_model(self):
        cursor, db_connection,_,_ = self.connect_database()
        cursor.execute(f"SELECT * FROM {self.name_table} WHERE item_code = %s", (self.item_code_cfg,))
        records = cursor.fetchall()
        cursor.close()
        db_connection.close()
        if records:
            first_record = records[0]
            load_item_code = first_record[1]
            load_path_weight = first_record[2]
            load_confidence_all_scale = first_record[3]
        return records,load_path_weight,load_item_code,load_confidence_all_scale

    def load_parameters_model(self,model1,records,load_path_weight,load_item_code,load_confidence_all_scale):
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
        
    def change_model(self,Frame_2,weights):
        global model1
        selected_file = filedialog.askopenfilename(title="Choose a file", filetypes=[("Model Files", "*.pt")])
        if selected_file:
            weights.delete(0,tk.END)
            weights.insert(0,selected_file)
            model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5','custom', path=selected_file, source='local', force_reload=False)
            for widget in Frame_2.grid_slaves():
                widget.grid_forget()
            self.option_layout_parameters(Frame_2,model1)  
        else:
            messagebox.showinfo("Notification","Please select the correct training file!")

    def load_params_child(self,weights,item_code):
        weight = weights.get()
        item_code_value = str(item_code.get())
        cursor, db_connection,_,_ = self.connect_database()
        cursor.execute("SELECT * FROM test_model_cam1_model1 WHERE item_code = %s", (item_code_value))
        cursor.close()
        db_connection.close()
        records = cursor.fetchall()
        model = torch.hub.load('C:/Users/CCSX009/Documents/yolov5','custom', path=weight, source='local', force_reload=False)
        return records,model

    def load_parameters_from_weight(self,weights, item_code):
        confirm_load_parameters = messagebox.askokcancel("Confirm", "Are you sure you want to load the parameters?")
        _,_ = self.load_params_child(weights, item_code)
        if confirm_load_parameters :
            try: 
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

                messagebox.showinfo("Notification", "Loaded parameters successfully!")
            except Exception as e:
                messagebox.showinfo("Notification", f"Parameters Loaded failed! Error: {str(e)}")
        else:
            pass

    def handle_image(self,img1_orgin, width, height,camera_frame):
        for widget in camera_frame.winfo_children():
            widget.destroy()
        image_result,time_processing,results_detect,label_ng = self.processing_handle_image_local(img1_orgin, width, height,cls=make_cls_var.get())
        img_pil = Image.fromarray(image_result)
        photo = ImageTk.PhotoImage(img_pil)
        canvas = tk.Canvas(camera_frame, width=width, height=height)
        canvas.grid(row=2, column=0, padx=10, pady=10, sticky='ew')
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo
        canvas.create_text(10, 10, anchor=tk.NW, text=f'Time: {time_processing}', fill='black', font=('Segoe UI', 20))
        canvas.create_text(10, 40, anchor=tk.NW, text=f'Result: {results_detect}', fill='green' if results_detect == 'OK' else 'red', font=('Segoe UI', 20))
        if not label_ng:
            canvas.create_text(10, 70, anchor=tk.NW, text=f'No Label', fill='green', font=('Segoe UI', 20))
        else:
            label_ng = ','.join(label_ng)
            canvas.create_text(10, 70, anchor=tk.NW, text=f'Label: {label_ng}', fill='red', font=('Segoe UI', 20))
        return results_detect

    def detect_single_img(self, camera_frame):
        selected_file = filedialog.askopenfilename(title="Choose a file", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if selected_file:
            for widget in camera_frame.winfo_children():
                widget.destroy()
            width = 480
            height = 450
            self.handle_image(selected_file, width, height,camera_frame)
        else: 
            pass
           
    def detect_multi_img(self,camera_frame):
        selected_folder = filedialog.askdirectory(title="Choose a folder")
        if selected_folder:
            self.image_files = [os.path.join(selected_folder, f) for f in os.listdir(selected_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.current_image_index = 0
            if self.image_files:
                for widget in camera_frame.winfo_children():
                    widget.destroy()
                self.show_image(self.current_image_index,camera_frame)
            else:
                messagebox.showinfo("No Images", "The selected folder contains no images.")
        else:
            pass

    def show_image(self,index,camera_frame):  
        width = 480
        height = 450
        image_path = self.image_files[index]
        self.handle_image(image_path, width, height,camera_frame)

    def detect_next_img(self,camera_frame):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_image(self.current_image_index,camera_frame)
        else:
            messagebox.showinfo("End of Images", "No more images in the folder.")

    def detect_previos_img(self,camera_frame):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image(self.current_image_index,camera_frame)
        else:
            messagebox.showinfo("Start of Images", "This is the first image in the folder.")

    def detect_auto(self, camera_frame):
        global selected_folder_original
        selected_folder_original = filedialog.askdirectory(title="Choose a folder")
        if selected_folder_original: 
            selected_folder = glob.glob(selected_folder_original + '/*.jpg')
            if not selected_folder:
                pass
            self.image_index = 0
            self.selected_folder = selected_folder
            self.camera_frame = camera_frame
            self.image_path_mks_cls = []
            self.results_detect = None
            def process_next_image():
                if self.image_index < len(self.selected_folder):
                    self.image_path_mks_cls = self.selected_folder[self.image_index]
                    width = 480
                    height = 450
                    self.results_detect = self.handle_image(self.image_path_mks_cls, width, height,camera_frame)
                    self.image_index += 1
                    self.camera_frame.after(500, process_next_image)
                else:
                    print("Processing auto detect complete")
            process_next_image()
        else:
            pass

    # def logging(self,folder_ok,folder_ng,logging_ok_checkbox_var,logging_ng_checkbox_var,camera_frame):
    #     selected_folder_original = filedialog.askdirectory(title="Choose a folder")
    #     selected_folder = glob.glob(selected_folder_original + '/*.jpg')
    #     width = 480
    #     height = 450
    #     self.image_index = 0
    #     self.selected_folder = selected_folder
    #     if not self.selected_folder:
    #         pass
    #     for img in self.selected_folder:
    #         basename = os.path.basename(img)
    #         if self.image_index < len(self.selected_folder):
    #             results_detect = self.handle_image(img, width, height,camera_frame)
    #             if results_detect == 'OK':
    #                 if logging_ok_checkbox_var.get():
    #                     shutil.move(img,os.path.join(folder_ok.get(),basename))
    #                 else:
    #                     pass
    #             else:
    #                 if logging_ng_checkbox_var.get():
    #                     shutil.move(img,os.path.join(folder_ng.get(),basename))
    #                 else:
    #                     pass
    #             self.image_index += 1
    #         else:
    #             messagebox.showinfo("End of Images", "No more images in the folder.")

    def logging(self, folder_ok, folder_ng, logging_ok_checkbox_var, logging_ng_checkbox_var, camera_frame):
        selected_folder_original = filedialog.askdirectory(title="Choose a folder")
        selected_folder = glob.glob(selected_folder_original + '/*.jpg')
        width = 480
        height = 450
        self.image_index = 0
        self.selected_folder = selected_folder
        if not self.selected_folder:
            return
        total_images = len(self.selected_folder)
        progress = ttk.Progressbar(camera_frame, orient="horizontal", length=300, mode="determinate")
        progress.grid(row=13, column=0, columnspan=2, padx=10, pady=10)
        progress_label = ttk.Label(camera_frame, text="Progress: 0%")
        progress_label.grid(row=14, column=0, columnspan=2, padx=10, pady=5)

        for img in self.selected_folder:
            basename = os.path.basename(img)
            if self.image_index < total_images:
                results_detect = self.handle_image(img, width, height, camera_frame)
                if results_detect == 'OK':
                    if logging_ok_checkbox_var.get():
                        shutil.move(img, os.path.join(folder_ok.get(), basename))
                else:
                    if logging_ng_checkbox_var.get():
                        shutil.move(img, os.path.join(folder_ng.get(), basename))
                progress['value'] = (self.image_index + 1) / total_images * 100
                progress_label.config(text=f"Progress: {int((self.image_index + 1) / total_images * 100)}%")
                camera_frame.update_idletasks()
                
                self.image_index += 1
            else:
                messagebox.showinfo("End of Images", "No more images in the folder.")
                break
      
    def toggle_state_layout_model(self): 
        if self.state == 1:
            password = simpledialog.askstring("Administrator", "Enter password:", show="*")
            if password == self.password:
                self.state = 0
                permisson_btn.config(text="Lock")
                self.toggle_widgets_state("normal")
            else:
                messagebox.showerror("Error", "Incorrect password!")
        else:
            self.state = 1
            permisson_btn.config(text="Unlock")
            self.toggle_widgets_state("disabled")

    def toggle_widgets_state(self,state):
        for widget in self.lockable_widgets:
            widget.config(state=state)

    def toggle_state_option_layout_parameters(self):
        for widget in self.lock_params:
            widget.config(state='disabled')

    def pick_folder_ok(self,folder_ok):
        file_path = filedialog.askdirectory(title="Choose a folder")
        if file_path:
            folder_ok.delete(0,tk.END)
            folder_ok.insert(0,file_path)
           
    def pick_folder_ng(self,folder_ng):
        file_path = filedialog.askdirectory(title="Choose a folder")
        if file_path:
            folder_ng.delete(0,tk.END)
            folder_ng.insert(0,file_path)
                
    def Camera_Settings(self):
        global model1,records
        records,load_path_weight,load_item_code,load_confidence_all_scale = self.load_data_model()
        
        model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5','custom', path=load_path_weight, source='local', force_reload=False)
        
        camera_settings_tab = ttk.Frame(self.settings_notebook)
        self.settings_notebook.add(camera_settings_tab, text="Camera 2")

        canvas = tk.Canvas(camera_settings_tab)
        scrollbar = ttk.Scrollbar(camera_settings_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        camera_settings_tab.grid_columnconfigure(0, weight=1)
        camera_settings_tab.grid_rowconfigure(0, weight=1)

        frame_width = 1500
        frame_height = 2000

        Frame_1 = ttk.LabelFrame(scrollable_frame, text="Frame 1", width=frame_width, height=frame_height)
        Frame_2 = ttk.LabelFrame(scrollable_frame, text="Frame 2", width=frame_width, height=frame_height)

        Frame_1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  
        Frame_2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.option_layout_models(Frame_1,Frame_2)
        self.option_layout_parameters(Frame_2,model1)
        self.load_parameters_model(model1,records,load_path_weight,load_item_code,load_confidence_all_scale)
        self.toggle_state_option_layout_parameters()
        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=1)
        scrollable_frame.grid_rowconfigure(0, weight=1)
        scrollable_frame.grid_rowconfigure(1, weight=1)

    def option_layout_models(self, Frame_1, Frame_2):
        
        global weights, scale_conf_all, size_model, item_code,make_cls_var,permisson_btn

        ttk.Label(Frame_1, text='1. File train detect model', font=('Segoe UI', 12)).grid(column=0, row=0, padx=10, pady=5, sticky="nws")

        weights = ttk.Entry(Frame_1, width=60)
        weights.grid(row=1, column=0, columnspan=5, padx=(30, 5), pady=5, sticky="w", ipadx=20, ipady=2)

        button_frame = ttk.Frame(Frame_1)
        button_frame.grid(row=2, column=0, columnspan=2, padx=(30, 30), pady=5, sticky="w")

        change_model_button = tk.Button(button_frame, text="Change Model", command=lambda: self.change_model(Frame_2, weights))
        change_model_button.grid(row=0, column=0, padx=(0, 8), pady=5, sticky="w", ipadx=5, ipady=2)
        change_model_button.config(state="disabled")
        self.lockable_widgets.append(change_model_button)

        load_parameters = tk.Button(button_frame, text="Load Parameters", command=lambda: self.load_parameters_from_weight(weights, item_code))
        load_parameters.grid(row=0, column=1, padx=(0, 8), pady=5, sticky="w", ipadx=5, ipady=2)
        load_parameters.config(state="disabled")
        self.lockable_widgets.append(load_parameters)

        custom_para = tk.Button(button_frame, text="Custom Parameters")
        custom_para.grid(row=0, column=2, padx=(0, 8), pady=5, sticky="w", ipadx=5, ipady=2)
        custom_para.config(state="disabled")
        self.lockable_widgets.append(custom_para)

        permisson_btn = tk.Button(button_frame, text="Unlock", command=lambda: self.toggle_state_layout_model())
        permisson_btn.grid(row=0, column=3, padx=(0, 8), pady=5, sticky="w", ipadx=5, ipady=2)

        label_scale_conf_all = ttk.Label(Frame_1, text='2. Confidence Threshold', font=('Segoe UI', 12))
        label_scale_conf_all.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="nws")

        scale_conf_all = tk.Scale(Frame_1, from_=1, to=100, orient='horizontal', length=400)
        scale_conf_all.grid(row=4, column=0, columnspan=2, padx=30, pady=5, sticky="nws")
        scale_conf_all.config(state="disabled")
        self.lockable_widgets.append(scale_conf_all)

        label_size_model = ttk.Label(Frame_1, text='2. Size Model', font=('Segoe UI', 12))
        label_size_model.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="nws")

        options = [468, 608, 832]
        size_model = ttk.Combobox(Frame_1, values=options, width=7)
        size_model.grid(row=6, column=0, columnspan=2, padx=30, pady=5, sticky="nws", ipadx=5, ipady=2)
        size_model.set(608)
        size_model.config(state="disabled")
        self.lockable_widgets.append(size_model)

        name_item_code = ttk.Label(Frame_1, text='3. Item Code', font=('Segoe UI', 12))
        name_item_code.grid(row=7, column=0, columnspan=2, padx=10, pady=5, sticky="nws")

        item_code = ttk.Entry(Frame_1, width=10)
        item_code.grid(row=8, column=0, columnspan=2, padx=30, pady=5, sticky="w", ipadx=5, ipady=2)
        self.lockable_widgets.append(item_code)

        save_data_to_database = ttk.Button(Frame_1, text='Apply', command=lambda: self.save_params_model(weights, scale_conf_all, item_code, model_name_labels, join, ok_vars, ng_vars, num_inputs, wn_inputs, wx_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales))
        save_data_to_database.grid(row=9, column=0, columnspan=2, padx=30, pady=5, sticky="w", ipadx=5, ipady=2)
        save_data_to_database.config(state="disabled")
        self.lockable_widgets.append(save_data_to_database)

        camera_frame_display = ttk.Label(Frame_1, text='4. Modify Image', font=('Segoe UI', 12))
        camera_frame_display.grid(row=10, column=0, columnspan=2, padx=10, pady=5, sticky="nws")

        camera_frame = ttk.LabelFrame(Frame_1, text=f"Camera 1", width=500, height=500)
        camera_frame.grid(row=11, column=0, columnspan=2, padx=30, pady=5, sticky="nws")

        camera_custom_setup = ttk.Frame(Frame_1)
        camera_custom_setup.grid(row=12, column=0, columnspan=2, padx=(30, 30), pady=5, sticky="w") 

        single_img = tk.Button(camera_custom_setup, text="Only Image", command=lambda: self.detect_single_img(camera_frame))
        single_img.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        single_img.config(state="disabled")
        self.lockable_widgets.append(single_img)

        multi_img = tk.Button(camera_custom_setup, text="Multi Image", command=lambda: self.detect_multi_img(camera_frame))
        multi_img.grid(row=0, column=1, padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        multi_img.config(state="disabled")
        self.lockable_widgets.append(multi_img)

        previos_img = tk.Button(camera_custom_setup, text="Prev...", command=lambda: self.detect_previos_img(camera_frame))
        previos_img.grid(row=0, column=2, padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        previos_img.config(state="disabled")
        self.lockable_widgets.append(previos_img)
        
        next_img = tk.Button(camera_custom_setup, text="Next...", command=lambda: self.detect_next_img(camera_frame))
        next_img.grid(row=0, column=3, padx=(0, 10), pady=5, sticky="w", ipadx=5, ipady=2)
        next_img.config(state="disabled")
        self.lockable_widgets.append(next_img)

        auto_detect = tk.Button(camera_custom_setup, text="Auto Detect", command=lambda: self.detect_auto(camera_frame))
        auto_detect.grid(row=0, column=4, padx=(0, 10), pady=5, sticky="w", ipadx=7, ipady=2)
        auto_detect.config(state="disabled")
        self.lockable_widgets.append(auto_detect)

        make_cls_var = tk.BooleanVar()
        make_cls = tk.Checkbutton(camera_custom_setup,text='Make class',variable=make_cls_var, onvalue=True, offvalue=False,anchor='w')
        make_cls.grid(row=0, column=5, padx=(0, 10), pady=5, sticky="w", ipadx=2, ipady=2)
        make_cls.config(state="disabled")
        self.lockable_widgets.append(make_cls)
        make_cls.var = make_cls_var

        logging_frame = ttk.Frame(Frame_1)
        logging_frame.grid(row=13, column=0, columnspan=2, padx=(30, 30), pady=10, sticky="w") 

        logging = tk.Button(logging_frame, text="Logging Image", command=lambda: self.logging(folder_ok,folder_ng,logging_ok_checkbox_var,logging_ng_checkbox_var,camera_frame))
        logging.grid(row=0, column=0, padx=(0,10), pady=5, sticky="w", ipadx=7, ipady=2)
        logging.config(state="disabled")
        self.lockable_widgets.append(logging)

        logging_ok_checkbox_var = tk.BooleanVar()
        logging_ok_checkbox = tk.Checkbutton(logging_frame,text='OK', variable=logging_ok_checkbox_var, onvalue=True, offvalue=False)
        logging_ok_checkbox.grid(row=1, column=2, padx=(0,10), pady=5, sticky="w", ipadx=7, ipady=2)
        logging_ok_checkbox.var = logging_ok_checkbox_var
        self.lock_params.append(logging_ok_checkbox)
        self.lockable_widgets.append(logging_ok_checkbox)

        logging_ng_checkbox_var = tk.BooleanVar()
        logging_ng_checkbox = tk.Checkbutton(logging_frame,text='NG', variable=logging_ng_checkbox_var, onvalue=True, offvalue=False)
        logging_ng_checkbox.grid(row=2, column=2, padx=(0,10), pady=5, sticky="w", ipadx=7, ipady=2)
        logging_ng_checkbox.var = logging_ng_checkbox_var
        self.lock_params.append(logging_ng_checkbox)
        self.lockable_widgets.append(logging_ng_checkbox)

        folder_ok = ttk.Entry(logging_frame, width=45)
        folder_ok.grid(row=1, column=0, padx=(0, 10), pady=3, sticky="w", ipadx=15, ipady=2)

        folder_ng = ttk.Entry(logging_frame ,width=45)
        folder_ng.grid(row=2, column=0, padx=(0, 10), pady=3, sticky="w", ipadx=15, ipady=2)

        folder_ok_button = tk.Button(logging_frame, text="Folder OK", command=lambda: self.pick_folder_ok(folder_ok))
        folder_ok_button.grid(row=1, column=1, padx=(0, 8), pady=3, sticky="w", ipadx=5, ipady=2)
        folder_ok_button.config(state="disabled")
        self.lockable_widgets.append(folder_ok_button)

        folder_ng_button = tk.Button(logging_frame, text="Folder NG", command=lambda: self.pick_folder_ng(folder_ng))
        folder_ng_button.grid(row=2, column=1, padx=(0, 8), pady=3, sticky="w", ipadx=5, ipady=2)
        folder_ng_button.config(state="disabled")
        self.lockable_widgets.append(folder_ng_button)

    

    def option_layout_parameters(self,Frame_2,model1):
        
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

        label = tk.Label(Frame_2, text='LABEL', fg='red', font=('Ubuntu', 12), width=12, anchor='center', relief="groove", borderwidth=2)
        label.grid(row=0, column=0, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        joint_detect = tk.Label(Frame_2, text='JOIN', fg='red', font=('Ubuntu', 12), anchor='center', relief="groove", borderwidth=2)
        joint_detect.grid(row=0, column=1, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        ok_joint = tk.Label(Frame_2, text='OK', fg='red', font=('Ubuntu', 12), anchor='center', relief="groove", borderwidth=2)
        ok_joint.grid(row=0, column=2, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        ng_joint = tk.Label(Frame_2, text='NG', fg='red', font=('Ubuntu', 12), anchor='center', relief="groove", borderwidth=2)
        ng_joint.grid(row=0, column=3, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        num_lb = tk.Label(Frame_2, text='NUM', fg='red', font=('Ubuntu', 12), width=7, anchor='center', relief="groove", borderwidth=2)
        num_lb.grid(row=0, column=4, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        width_n = tk.Label(Frame_2, text='W_N', fg='red', font=('Ubuntu', 12), width=7, anchor='center', relief="groove", borderwidth=2)
        width_n.grid(row=0, column=5, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        wight_x= tk.Label(Frame_2, text='W_X', fg='red', font=('Ubuntu', 12), width=7, anchor='center', relief="groove", borderwidth=2)
        wight_x.grid(row=0, column=6, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        height_n = tk.Label(Frame_2, text='H_N', fg='red', font=('Ubuntu', 12), width=7, anchor='center', relief="groove", borderwidth=2)
        height_n.grid(row=0, column=7, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        height_x = tk.Label(Frame_2, text='H_X', fg='red', font=('Ubuntu', 12), width=7, anchor='center', relief="groove", borderwidth=2)
        height_x.grid(row=0, column=8, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        plc_var = tk.Label(Frame_2, text='PLC', fg='red', font=('Ubuntu', 12), width=7, anchor='center', relief="groove", borderwidth=2)
        plc_var.grid(row=0, column=9, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        conf = tk.Label(Frame_2, text='CONFIDENCE THRESHOLD', fg='red', font=('Ubuntu', 12), width=25, anchor='center', relief="groove", borderwidth=2)
        conf.grid(row=0, column=10, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

        for i1 in range(len(model1.names)):
            row_widgets = []

            model_name_label = tk.Label(Frame_2, text=f'{model1.names[i1]}', fg='black', font=('Segoe UI', 12), width=15, anchor='w')
            row_widgets.append(model_name_label)
            model_name_labels.append(model_name_label)

            join_checkbox_var = tk.BooleanVar()
            join_checkbox = tk.Checkbutton(Frame_2, variable=join_checkbox_var, onvalue=True, offvalue=False,anchor='w')
            join_checkbox.grid()
            join_checkbox.var = join_checkbox_var
            row_widgets.append(join_checkbox)
            join.append(join_checkbox_var)
            self.lock_params.append(join_checkbox)
            self.lockable_widgets.append(join_checkbox)

            ok_checkbox_var = tk.BooleanVar()
            ok_checkbox = tk.Checkbutton(Frame_2, variable=ok_checkbox_var, onvalue=True, offvalue=False, command=lambda rw=row_widgets:ok_selected(rw), anchor='w')
            ok_checkbox.grid()
            ok_checkbox.var = ok_checkbox_var
            row_widgets.append(ok_checkbox)
            ok_vars.append(ok_checkbox_var)
            self.lock_params.append(ok_checkbox)
            self.lockable_widgets.append(ok_checkbox)

            ng_checkbox_var = tk.BooleanVar()
            ng_checkbox = tk.Checkbutton(Frame_2, variable=ng_checkbox_var, onvalue=True, offvalue=False, command=lambda rw=row_widgets:ng_selected(rw), anchor='w')
            ng_checkbox.grid()
            ng_checkbox.var = ng_checkbox_var
            row_widgets.append(ng_checkbox)
            ng_vars.append(ng_checkbox_var)
            self.lock_params.append(ng_checkbox)
            self.lockable_widgets.append(ng_checkbox)

            num_input = tk.Entry(Frame_2, width=7,)
            num_input.insert(0, '1')
            row_widgets.append(num_input)
            num_inputs.append(num_input)
            self.lock_params.append(num_input)
            self.lockable_widgets.append(num_input)

            wn_input = tk.Entry(Frame_2, width=7, )
            wn_input.insert(0, '0')
            row_widgets.append(wn_input)
            wn_inputs.append(wn_input)
            self.lock_params.append(wn_input)
            self.lockable_widgets.append(wn_input)

            wx_input = tk.Entry(Frame_2, width=7, )
            wx_input.insert(0, '1600')
            row_widgets.append(wx_input)
            wx_inputs.append(wx_input)
            self.lock_params.append(wx_input)
            self.lockable_widgets.append(wx_input)

            hn_input = tk.Entry(Frame_2, width=7, )
            hn_input.insert(0, '0')
            row_widgets.append(hn_input)
            hn_inputs.append(hn_input)
            self.lock_params.append(hn_input)
            self.lockable_widgets.append(hn_input)

            hx_input = tk.Entry(Frame_2, width=7, )
            hx_input.insert(0, '1200')
            row_widgets.append(hx_input)
            hx_inputs.append(hx_input)
            self.lock_params.append(hx_input)
            self.lockable_widgets.append(hx_input)

            plc_input = tk.Entry(Frame_2, width=7,)
            plc_input.insert(0, '0')
            row_widgets.append(plc_input)
            plc_inputs.append(plc_input)
            self.lock_params.append(plc_input)
            self.lockable_widgets.append(plc_input)

            conf_scale = tk.Scale(Frame_2, from_=1, to=100, orient='horizontal', length=250)
            row_widgets.append(conf_scale)
            conf_scales.append(conf_scale)
            self.lock_params.append(conf_scale)
            self.lockable_widgets.append(conf_scale)

            widgets.append(row_widgets)

        for i, row in enumerate(widgets):
            for j, widget in enumerate(row):
                widget.grid(row=i+1, column=j, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

    @staticmethod
    def processing_handle_image_customize(input_image,width,height):
        label_remove = []
        size_model_all = int(size_model.get())
        conf_all = int(scale_conf_all.get())/100
        t1 = time.time()
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        results = model1(input_image,size_model_all,conf_all)
        table_results = results.pandas().xyxy[0]
        model_names = model1.names
        model_settings = []
        for i1 in range(len(model_name_labels)):
            model_settings.append({
                'label_name': model_name_labels[i1].cget("text"),
                'join_detect': join[i1].get(),
                'OK_jont': ok_vars[i1].get(),
                'NG_jont': ng_vars[i1].get(),
                'num_labels': int(num_inputs[i1].get()),
                'width_min': int(wn_inputs[i1].get()),
                'width_max': int(wx_inputs[i1].get()),
                'height_min': int(hn_inputs[i1].get()),
                'height_max': int(hx_inputs[i1].get()),
                'PLC_value': int(plc_inputs[i1].get()),
                'cmpnt_conf': int(conf_scales[i1].get()),
            })
        for i in range(len(table_results.index)):
            width_result = table_results['xmax'][i] - table_results['xmin'][i]
            height_result = table_results['ymax'][i] - table_results['ymin'][i]
            conf_result = table_results['confidence'][i] * 100
            label_name_tables_result = table_results['name'][i]
            for i1 in range(len(model_names)):
                for setting in model_settings:
                    if label_name_tables_result == model_names[i1] == setting['label_name']:
                        if setting['join_detect']:
                            if width_result < setting['width_min'] or width_result > setting['width_max'] \
                                    or height_result < setting['height_min'] or height_result > setting['height_max'] \
                                    or conf_result < setting['cmpnt_conf']:
                                label_remove.append(i)
                        else:
                            label_remove.append(i)
        table_results.drop(index=label_remove, inplace=True)   
        name_rest = list(table_results['name'])                
        results_detect = 'ERROR'
        ok_variable = False
        list_label_ng = []
        for i1 in range(len(model_names)):
            for j1 in model_settings:
                if model_names[i1] == j1['label_name']:
                    if j1['join_detect']:
                        if j1['OK_jont']:
                            number_of_labels = name_rest.count(model_names[i1])
                            if number_of_labels != j1['num_labels']:
                                results_detect = 'NG'
                                ok_variable = True
                                list_label_ng.append(model_names[i1])
                                # self.write_plc_keyence(setting['PLC_value'], 1)
                        if j1['NG_jont']:
                            if j1['label_name'] in name_rest:
                                results_detect = 'NG'
                                ok_variable = True
                                list_label_ng.append(model_names[i1])
        if not ok_variable:
            results_detect = 'OK'
            # self.write_plc_keyence(PLC_value,2)
        show_img = np.squeeze(results.render(label_remove))
        show_img = cv2.resize(show_img, (width,height), interpolation=cv2.INTER_AREA)
        t2 = time.time() - t1
        time_processing = str(int(t2*1000)) + 'ms'
        return show_img,time_processing,results_detect,list_label_ng
    
    def _make_cls(self,image_path_mks_cls,results,model_settings):  
        with open(image_path_mks_cls[:-3] + 'txt', "a") as file:
            for params in results.xywhn:
                params = params.tolist()
                for item in range(len(params)):
                    param = params[item]
                    param = [round(i,6) for i in param]
                    number_label = int(param[5])
                    conf_result = float(param[4])
                    width_result = float(param[2])*1200
                    height_result = float(param[3])*1600
                    for setting in model_settings:
                        if results.names[int(number_label)] == setting['label_name']:
                            if setting['join_detect']:
                                if width_result < setting['width_min'] or width_result > setting['width_max'] \
                                        or height_result < setting['height_min'] or height_result > setting['height_max'] \
                                        or conf_result < setting['cmpnt_conf']: 
                                    formatted_values = ["{:.6f}".format(value) for value in param[:4]]
                                    output_line = "{} {}\n".format(str(number_label),' '.join(formatted_values))
                                    file.write(output_line)
        with open(selected_folder_original + '/classes.txt', "w") as file:
            for i1 in range(len(results.names)):
                file.write(str(results.names[i1])+'\n')
      
    def classify_imgs(self):
        pass    

    def processing_handle_image_local(self,input_image_original,width,height,cls=False):
        label_remove = []
        size_model_all = int(size_model.get())
        conf_all = int(scale_conf_all.get())/100
        t1 = time.time()
        results = model1(input_image_original,size_model_all,conf_all)
        table_results = results.pandas().xyxy[0]
        model_names = model1.names
        model_settings = []
        for i1 in range(len(model_name_labels)):
            model_settings.append({
                'label_name': model_name_labels[i1].cget("text"),
                'join_detect': join[i1].get(),
                'OK_jont': ok_vars[i1].get(),
                'NG_jont': ng_vars[i1].get(),
                'num_labels': int(num_inputs[i1].get()),
                'width_min': int(wn_inputs[i1].get()),
                'width_max': int(wx_inputs[i1].get()),
                'height_min': int(hn_inputs[i1].get()),
                'height_max': int(hx_inputs[i1].get()),
                'PLC_value': int(plc_inputs[i1].get()),
                'cmpnt_conf': int(conf_scales[i1].get()),
            })
        for i in range(len(table_results.index)):
            width_result = table_results['xmax'][i] - table_results['xmin'][i]
            height_result = table_results['ymax'][i] - table_results['ymin'][i]
            conf_result = table_results['confidence'][i] * 100
            label_name_tables_result = table_results['name'][i]
            for i1 in range(len(model_names)):
                for setting in model_settings:
                    if label_name_tables_result == model_names[i1] == setting['label_name']:
                        if setting['join_detect']:
                            if width_result < setting['width_min'] or width_result > setting['width_max'] \
                                    or height_result < setting['height_min'] or height_result > setting['height_max'] \
                                    or conf_result < setting['cmpnt_conf']:
                                label_remove.append(i)
                        else:
                            label_remove.append(i)
        table_results.drop(index=label_remove, inplace=True)   
        name_rest = list(table_results['name'])                
        results_detect = 'ERROR'
        ok_variable = False
        list_label_ng = []
        for i1 in range(len(model_names)):
            for j1 in model_settings:
                if model_names[i1] == j1['label_name']:
                    if j1['join_detect']:
                        if j1['OK_jont']:
                            number_of_labels = name_rest.count(model_names[i1])
                            if number_of_labels != j1['num_labels']:
                                results_detect = 'NG'
                                ok_variable = True
                                list_label_ng.append(model_names[i1])
                        if j1['NG_jont']:
                            if j1['label_name'] in name_rest:
                                results_detect = 'NG'
                                ok_variable = True
                                list_label_ng.append(model_names[i1])
        if not ok_variable:
            results_detect = 'OK'
        show_img = np.squeeze(results.render(label_remove))
        show_img = cv2.resize(show_img, (width,height), interpolation=cv2.INTER_AREA)
        t2 = time.time() - t1
        time_processing = str(int(t2*1000)) + 'ms'
        if cls:
            self._make_cls(input_image_original,results,model_settings)
        return show_img,time_processing,results_detect,list_label_ng

class Model_Camera_3(PLC_Connection,MySQL_Connection):
    
    def __init__(self):
        self.item_code_cfg = "EDFWTA"
        self.image_files = []
        self.current_image_index = -1
        self.cursor,self.db_connection = self.Connect_to_MySQLServer(host="127.0.0.1",user="root1",passwd="987654321",database="model_1")  

    def Camera_Settings(self,notebook):
        # global model1,records
        # records,load_path_weight,load_item_code,load_confidence_all_scale = self.load_data_model()
        
        # model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5','custom', path=load_path_weight, source='local', force_reload=False)
        
        camera_settings_tab = ttk.Frame(notebook)
        notebook.add(camera_settings_tab, text="Camera 3")

        canvas = tk.Canvas(camera_settings_tab)
        scrollbar = ttk.Scrollbar(camera_settings_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        camera_settings_tab.grid_columnconfigure(0, weight=1)
        camera_settings_tab.grid_rowconfigure(0, weight=1)

        frame_width = 900
        frame_height = 900

        Frame_1 = ttk.LabelFrame(scrollable_frame, text="Frame 1", width=frame_width, height=frame_height)
        Frame_2 = ttk.LabelFrame(scrollable_frame, text="Frame 2", width=frame_width, height=frame_height)

        Frame_1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  
        Frame_2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        

        # self.option_layout_models(Frame_1,Frame_2)
        # self.option_layout_parameters(Frame_2,model1)
        # self.load_parameters_model(model1,records,load_path_weight,load_item_code,load_confidence_all_scale)

        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=1)
        scrollable_frame.grid_rowconfigure(0, weight=1)
        scrollable_frame.grid_rowconfigure(1, weight=1)

class Train():
    pass 

class Training_Data:

    def __init__(self, parent):
        self.notebook = ttk.Notebook(parent)
        parent.add(self.notebook, text="Training Data")

        self.training_tab = ttk.Frame(self.notebook)
        self.training_timer_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.training_tab, text="Training")
        self.notebook.add(self.training_timer_tab, text="Training Timer")

        self.setup_training_tab()
        self.setup_training_timer_tab()

    def setup_training_tab(self):
        canvas = tk.Canvas(self.training_tab)
        scrollbar = ttk.Scrollbar(self.training_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.training_tab.grid_columnconfigure(0, weight=1)
        self.training_tab.grid_rowconfigure(0, weight=1)

        frame1_width = 400
        frame1_height = 1000
        frame2_width = 1500
        frame2_height = 1000

        self.Frame_1 = ttk.LabelFrame(scrollable_frame, text="Frame 1", width=frame1_width, height=frame1_height)
        self.Frame_2 = ttk.LabelFrame(scrollable_frame, text="Frame 2", width=frame2_width, height=frame2_height)

        self.Frame_1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  
        self.Frame_2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=1)
        scrollable_frame.grid_rowconfigure(0, weight=1)
        scrollable_frame.grid_rowconfigure(1, weight=1)


    def setup_training_timer_tab(self):
        canvas = tk.Canvas(self.training_timer_tab)
        scrollbar = ttk.Scrollbar(self.training_timer_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.training_timer_tab.grid_columnconfigure(0, weight=1)
        self.training_timer_tab.grid_rowconfigure(0, weight=1)

        frame1_width = 400
        frame1_height = 1000
        frame2_width = 1500
        frame2_height = 1000

        self.Frame_1 = ttk.LabelFrame(scrollable_frame, text="Frame 1", width=frame1_width, height=frame1_height)
        self.Frame_2 = ttk.LabelFrame(scrollable_frame, text="Frame 2", width=frame2_width, height=frame2_height)

        self.Frame_1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  
        self.Frame_2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=1)
        scrollable_frame.grid_rowconfigure(0, weight=1)
        scrollable_frame.grid_rowconfigure(1, weight=1)

class Main_Display(Model_Camera_1,Model_Camera_2):

    def __init__(self,window,notebook):
        self.window = window
        self.notebook = notebook
        self.camera1_frame,self.camera2_frame = self.Display_Camera()
        self.update_images(self.window, self.camera1_frame, self.camera2_frame)
        self.lock_process = False

    def display_images_c1(self, camera_frame, camera_number):
        width = 800
        height = 800
        image_paths = glob.glob(f"C:/Users/CCSX009/Documents/yolov5/test_image/camera{camera_number}/*.jpg")
        if len(image_paths) == 0:
            pass
        else:
            for filename in image_paths:
                img1_orgin = cv2.imread(filename)
                if img1_orgin is None:
                    print('loading img 1...')
                    continue
                for widget in camera_frame.winfo_children():
                    widget.destroy()
                image_result,time_processing,results_detect,list_label_ng = Model_Camera_1.processing_handle_image_customize(img1_orgin,width,height)
                time_processing_output.config(text=f'{time_processing}')
                if results_detect == 'OK':
                    result_detection.config(text=results_detect,fg='green')
                else:
                    result_detection.config(text=results_detect,fg='red')
                list_label_ng = ','.join(list_label_ng)
                img_pil = Image.fromarray(image_result)
                photo = ImageTk.PhotoImage(img_pil)
                canvas = tk.Canvas(camera_frame, width=width, height=height)
                canvas.grid(row=2, column=0, padx=10, pady=10, sticky='ew')
                canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                canvas.image = photo
                canvas.create_text(10, 10, anchor=tk.NW, text=f'Time: {time_processing}', fill='black', font=('Segoe UI', 20))
                canvas.create_text(10, 40, anchor=tk.NW, text=f'Result: {results_detect}', fill='green' if results_detect == 'OK' else 'red', font=('Segoe UI', 20))
                canvas.create_text(10, 70, anchor=tk.NW, text=f'Label: {list_label_ng}', fill='red', font=('Segoe UI', 20))

    def display_images_c2(self, camera_frame, camera_number):
        width = 800
        height = 800
        image_paths = glob.glob(f"C:/Users/CCSX009/Documents/yolov5/test_image/camera{camera_number}/*.jpg")
        if len(image_paths) == 0:
            pass
        else:
            for filename in image_paths:
                img1_orgin = cv2.imread(filename)
                if img1_orgin is None:
                    print('loading img 2...')
                    continue
                for widget in camera_frame.winfo_children():
                    widget.destroy()
                image_result,time_processing,results_detect,list_label_ng = Model_Camera_2.processing_handle_image_customize(img1_orgin,width,height)
                time_processing_output.config(text=f'{time_processing}')
                if results_detect == 'OK':
                    result_detection.config(text=results_detect,fg='green')
                else:
                    result_detection.config(text=results_detect,fg='red')
                list_label_ng = ','.join(list_label_ng)
                img_pil = Image.fromarray(image_result)
                photo = ImageTk.PhotoImage(img_pil)
                canvas = tk.Canvas(camera_frame, width=width, height=height)
                canvas.grid(row=2, column=0, padx=10, pady=10, sticky='ew')
                canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                canvas.image = photo
                canvas.create_text(10, 10, anchor=tk.NW, text=f'Time: {time_processing}', fill='black', font=('Segoe UI', 20))
                canvas.create_text(10, 40, anchor=tk.NW, text=f'Result: {results_detect}', fill='green' if results_detect == 'OK' else 'red', font=('Segoe UI', 20))
                canvas.create_text(10, 70, anchor=tk.NW, text=f'Label: {list_label_ng}', fill='red', font=('Segoe UI', 20))

    def update_images(self,window, camera1_frame, camera2_frame):
        self.display_images_c1(camera1_frame, 1)
        self.display_images_c2(camera2_frame, 2)
        window.after(50, self.update_images,window,camera1_frame, camera2_frame)

    def create_camera_frame_cam1(self, tab1, camera_number):

        global time_processing_output, result_detection

        style = ttk.Style()
        style.configure("Custom.TLabelframe", borderwidth=0)
        style.configure("Custom.TLabelframe.Label", background="white", foreground="white")

        frame_1 = ttk.LabelFrame(tab1, width=900, height=900, style="Custom.TLabelframe")
        frame_1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        camera_frame = ttk.LabelFrame(frame_1, text=f"Camera {camera_number}", width=800, height=800)
        camera_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

        time_frame = ttk.LabelFrame(frame_1, text=f"Time Processing Camera {camera_number}", width=300, height=100)
        time_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        time_processing_output = tk.Label(time_frame, text='0 ms', fg='black', font=('Segoe UI', 30), width=10, height=1, anchor='center')
        time_processing_output.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        result = ttk.LabelFrame(frame_1, text=f"Result Camera {camera_number}", width=300, height=100)
        result.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        result_detection = tk.Label(result, text='ERROR', fg='red', font=('Segoe UI', 30), width=10, height=1, anchor='center')
        result_detection.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        bonus = ttk.LabelFrame(frame_1, text=f"Bonus {camera_number}", width=300, height=100)
        bonus.grid(row=1, column=2, padx=10, pady=5, sticky="ew")

        bonus_test = tk.Label(bonus, text='Bonus', fg='red', font=('Segoe UI', 30), width=10, height=1, anchor='center')
        bonus_test.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        frame_1.grid_columnconfigure(0, weight=1)
        frame_1.grid_columnconfigure(1, weight=1)
        frame_1.grid_columnconfigure(2, weight=1)
        frame_1.grid_rowconfigure(0, weight=1)
        frame_1.grid_rowconfigure(1, weight=1)

        return camera_frame

    def create_camera_frame_cam2(self, tab1, camera_number):

        global time_processing_output_2, result_detection_2

        style = ttk.Style()
        style.configure("Custom.TLabelframe", borderwidth=0)
        style.configure("Custom.TLabelframe.Label", background="white", foreground="white")

        frame_2 = ttk.LabelFrame(tab1, width=900, height=900, style="Custom.TLabelframe")
        frame_2.grid(row=0, column=1, padx=200, pady=10, sticky="nsew")

        camera_frame = ttk.LabelFrame(frame_2, text=f"Camera {camera_number}", width=800, height=800)
        camera_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

        time_frame = ttk.LabelFrame(frame_2, text=f"Time Processing Camera {camera_number}", width=300, height=100)
        time_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        time_processing_output_2 = tk.Label(time_frame, text='0 ms', fg='black', font=('Segoe UI', 30), width=10, height=1, anchor='center')
        time_processing_output_2.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        result = ttk.LabelFrame(frame_2, text=f"Result Camera {camera_number}", width=300, height=100)
        result.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        result_detection_2 = tk.Label(result, text='ERROR', fg='red', font=('Segoe UI', 30), width=10, height=1, anchor='center')
        result_detection_2.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        bonus = ttk.LabelFrame(frame_2, text=f"Bonus {camera_number}", width=300, height=100)
        bonus.grid(row=1, column=2, padx=10, pady=5, sticky="ew")

        bonus_test = tk.Label(bonus, text='Bonus', fg='red', font=('Segoe UI', 30), width=10, height=1, anchor='center')
        bonus_test.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        frame_2.grid_columnconfigure(0, weight=1)
        frame_2.grid_columnconfigure(1, weight=1)
        frame_2.grid_columnconfigure(2, weight=1)
        frame_2.grid_rowconfigure(0, weight=1)
        frame_2.grid_rowconfigure(1, weight=1)

        return camera_frame

    def Display_Camera(self):
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="Display Camera")

        camera1_frame = self.create_camera_frame_cam1(tab1, 1)
        camera2_frame = self.create_camera_frame_cam2(tab1, 2)

        return camera1_frame, camera2_frame

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
    window.title("YOLOv5 by Utralytics ft Tkinter")
    window.state('zoomed')
    notebook = ttk.Notebook(window)
    notebook.pack(fill="both", expand=True)
    removefile()
    #tab1
    Main_Display(window,notebook)
    #tab2
    settings_notebook = ttk.Notebook(notebook)
    notebook.add(settings_notebook, text="Camera Configure Settup")
    Model_Camera_1(settings_notebook)
    Model_Camera_2(settings_notebook)
    #tab3
    Training_Data(notebook)

    window.mainloop()

if __name__ == "__main__":
    main()
