import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import glob
import torch
import mysql.connector
from tkinter import messagebox
def display_images(camera_frame, camera_number):
    for widget in camera_frame.winfo_children():
        widget.destroy()
    
    image_paths = glob.glob(f"C:/Users/CCSX009/Documents/yolov5/test_image/camera{camera_number}/*.jpg")
    if image_paths:
        image_path = image_paths[-1]
        image = Image.open(image_path)
        image = image.resize((800, 800), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        image_label = tk.Label(camera_frame, image=photo)
        image_label.image = photo
        image_label.pack()

def update_images(window, camera1_frame, camera2_frame):
    display_images(camera1_frame, 1)
    display_images(camera2_frame, 2)
    window.after(100, update_images,window,camera1_frame, camera2_frame)


def create_camera_frame(notebook, camera_number):
    camera_frame = ttk.LabelFrame(notebook, text=f"Camera {camera_number}", width=800, height=800)
    camera_frame.grid(row=0, column=camera_number-1, padx=80, pady=20, sticky="nws")
    time_frame = ttk.LabelFrame(notebook, text=f"Time Processing Camera {camera_number}", width=400, height=100)
    time_frame.grid(row=1, column=camera_number-1, padx=80, pady=10, sticky="nws")
    return camera_frame

def Display_Camera(notebook):
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="Display Camera")

    camera1_frame = create_camera_frame(tab1, 1)
    camera2_frame = create_camera_frame(tab1, 2)

    return camera1_frame, camera2_frame


class Model_Camera_1():
    
    def __init__(self,notebook,cursor,db_connection):
        self.notebook = notebook
        self.cursor = cursor
        self.db_connection = db_connection
        self.item_code = "ABC2D3F"

    def save_data_model_1(self,weights,scale_conf_all,item_code,model_name_labels, join, ok_vars, ng_vars, num_inputs,  wn_inputs, wx_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales):
        confirm_save_data = messagebox.askokcancel("Confirm", "Are you sure you want to save the data?")
        delete_record = "DELETE FROM test_model_cam1_model1 WHERE item_code = %s"
        self.cursor.execute(delete_record,(self.item_code,))
        if confirm_save_data:
            try:
                for i1 in range(len(model_name_labels)):
                    weight = (weights.get())
                    confidence_all = int(scale_conf_all.get())
                    item_code_value = str(item_code.get())
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
                    query_sql = "INSERT INTO test_model_cam1_model1 (item_code,weight,confidence_all,label_name,join_detect,OK,NG,num_labels,width_min,width_max,height_min,height_max,PLC_value,cmpnt_conf) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s)"
                    values = (item_code_value,weight,confidence_all,label_name,join_detect,OK_jont,NG_jont,num_labels,width_min,width_max,height_min,height_max,PLC_value,cmpnt_conf)
                    self.cursor.execute(query_sql, values)
                    messagebox.showinfo("Notification", "Data saved successfully!")
            except Exception as e:
                messagebox.showinfo("Notification", f"Data saved failed! Error: {str(e)}")
        else: 
            pass

    def load_data_model_1(self):
        item_code = "ABC2D3F"
        self.cursor.execute("SELECT * FROM test_model_cam1_model1 WHERE item_code = %s", (item_code,))
        load_item_code= self.cursor.fetchone()[1]
        load_path_weight = self.cursor.fetchone()[2]
        load_confidence_all_scale = self.cursor.fetchone()[3]
        records = self.cursor.fetchall()
        self.cursor.close()
        self.db_connection.close()
        return records,load_path_weight,load_item_code,load_confidence_all_scale

    def load_parameters_model_1(self,records,load_path_weight,load_item_code,load_confidence_all_scale):
        model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5', 'custom', path=load_path_weight, source='local', force_reload=False)
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
        

    def change_model_1(self,Frame_2,weights):
        selected_file = filedialog.askopenfilename(title="Choose a file", filetypes=[("Model Files", "*.pt")])
        if selected_file:
            weights.delete(0, tk.END)
            weights.insert(0, selected_file)
            model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5', 'custom', path=selected_file, source='local', force_reload=True)
            for widget in Frame_2.grid_slaves():
                widget.grid_forget()
            self.option_1_parameters(Frame_2,model1)

    def Camera_1_Settings(self):
        records,load_path_weight,load_item_code,load_confidence_all_scale = self.load_data_model_1()
        # load_path_weight = "V:/tamsat10/TAM_SAT_M100_A75_2024-05-20.pt"
        model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5', 'custom', path=load_path_weight, source='local', force_reload=False)
        
        camera_settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(camera_settings_tab, text="Camera 1 Settings")

        frame_width = 1500
        frame_height = 2000

        Frame_1 = ttk.LabelFrame(camera_settings_tab, text="Frame 1", width=frame_width, height=frame_height)
        Frame_2 = ttk.LabelFrame(camera_settings_tab, text="Frame 2", width=frame_width, height=frame_height)

        Frame_1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  
        Frame_2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        

        self.option_1_layout_models(Frame_1,Frame_2)
        self.option_1_parameters(Frame_2,model1)
        self.load_parameters_model_1(records,load_path_weight,load_item_code,load_confidence_all_scale)

    def option_1_layout_models(self,Frame_1, Frame_2):

        global weights,scale_conf_all,item_code
        
        ttk.Label(Frame_1, text='1. File train detect model', font=('Segoe UI', 12)).grid(column=0, row=0, padx=10, pady=5, sticky="nws")

        weights = ttk.Entry(Frame_1, width=50)
        weights.grid(row=1, column=0, columnspan=5, padx=30, pady=5, sticky="w", ipadx=100, ipady=2)

        change_model_button = tk.Button(Frame_1, text="Change Model", command=lambda: self.change_model_1(Frame_2, weights))
        change_model_button.grid(row=1, column=1, padx=10, pady=5, sticky="w", ipadx=5, ipady=2)

        label_scale_conf_all = ttk.Label(Frame_1, text='2. Confidence all', font=('Segoe UI', 12))
        label_scale_conf_all.grid(row=2, column=0, padx=10, pady=5, sticky="nws")
        
        scale_conf_all = tk.Scale(Frame_1, from_=1, to=100, orient='horizontal', length=500)
        scale_conf_all.grid(row=3, column=0, padx=30, pady=5, sticky="nws")
        
        name_item_code = ttk.Label(Frame_1, text='3. Item code', font=('Segoe UI', 12))
        name_item_code.grid(row=4, column=0, padx=10, pady=5, sticky="nws")

        item_code = ttk.Entry(Frame_1, width=50)
        item_code.grid(row=5, column=0, columnspan=5, padx=30, pady=5, sticky="w", ipadx=20, ipady=2)

        save_data_to_database = ttk.Button(Frame_1, text='Apply', command=lambda: self.save_data_model_1(weights,scale_conf_all,item_code,model_name_labels, join, ok_vars, ng_vars, num_inputs,  wn_inputs, wx_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales))
        save_data_to_database.grid(row=6, column=0, padx=30, pady=5, sticky="w", ipadx=5, ipady=2)

        camera_frame_display = ttk.Label(Frame_1, text='4. Display Camera', font=('Segoe UI', 12))
        camera_frame_display.grid(row=7, column=0, padx=10, pady=5, sticky="nws")

        camera_frame = ttk.LabelFrame(Frame_1, text=f"Camera 1", width=500, height=500)
        camera_frame.grid(row=8, column=0, padx=30, pady=5, sticky="nws")

        
    def option_1_parameters(self,Frame_2, model1):
        
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

    

def Connect_MySQLServer():
        db_connection = mysql.connector.connect(
        host="127.0.0.1",
        user="root1", 
        passwd="987654321",
        database="model_1")                    
        cursor = db_connection.cursor()
        return cursor,db_connection

def main():
    window = tk.Tk()
    window.title("Data Entry Form")
    window.state('zoomed')
    notebook = ttk.Notebook(window)
    notebook.pack(fill="both", expand=True)

    cursor,db_connection = Connect_MySQLServer()

    #tab 1
    camera1_frame, camera2_frame = Display_Camera(notebook)

    #tab 2
    Camera_1_Settings = Model_Camera_1(notebook,cursor,db_connection)
    Camera_1_Settings.Camera_1_Settings()


    update_images(window,camera1_frame, camera2_frame)
    window.mainloop()

if __name__ == "__main__":
    main()
