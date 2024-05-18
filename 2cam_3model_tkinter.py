import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import glob
import torch
import mysql.connector

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
    def __init__(self,notebook):
        self.notebook = notebook

    def Connect_MySQLServer(self,):
        db_connection = mysql.connector.connect(
        host="127.0.0.1",
        user="root1", 
        passwd="987654321",
        database="server_management")                    
        cursor = db_connection.cursor()
        return cursor,db_connection


    def print_widget_values(self,model_name_labels, join, ok_vars, num_inputs, ng_vars, wn_inputs, wx_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales):
        for i1 in range(len(model_name_labels)):
            print("Values for", model_name_labels[i1].cget("text"))
            print("Model Name Label Text:", model_name_labels[i1].cget("text"))
            print("Join:", join[i1].var.get()) 
            print("OK Radio Value:", ok_vars[i1].var.get())  
            print("Num Input Value:", num_inputs[i1].get())
            print("NG Radio Value:", ng_vars[i1].var.get())
            print("WN Input Value:", wn_inputs[i1].get())
            print("WX Input Value:", wx_inputs[i1].get())
            print("HN Input Value:", hn_inputs[i1].get())
            print("HX Input Value:", hx_inputs[i1].get())
            print("PLC Input Value:", plc_inputs[i1].get())
            print("Conf Scale Value:", conf_scales[i1].get())
            print('------------------------------')

    def open_image(self,model_name_labels, join, ok_vars, num_inputs, ng_vars, wn_inputs, wx_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales):
        self.print_widget_values(model_name_labels, join, ok_vars, num_inputs, ng_vars, wn_inputs, wx_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales)

    def change_model_1(self,Frame_2,entry):
        selected_file = filedialog.askopenfilename(title="Choose a file", filetypes=[("Model Files", "*.pt")])
        if selected_file:
            entry.delete(0, tk.END)
            entry.insert(0, selected_file)
            model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5', 'custom', path=selected_file, source='local', force_reload=True)
            for widget in Frame_2.grid_slaves():
                widget.grid_forget()
            self.option_1_parameters(Frame_2,model1)

    def Camera_1_Settings(self):
        model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5', 'custom', path="C:/Users/CCSX009/Documents/yolov5/file_train/X75_M100_DEN_CAM1_2024-03-11.pt", source='local', force_reload=False)
        
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

    def option_1_layout_models(self,Frame_1, Frame_2):
        ttk.Label(Frame_1, text='1. File detect model', font=('Segoe UI', 12)).grid(column=0, row=0, padx=10, pady=5, sticky="nws")

        model_file_entry = ttk.Entry(Frame_1, width=50)
        model_file_entry.grid(row=1, column=0, columnspan=5, padx=30, pady=5, sticky="w", ipadx=100, ipady=2)

        change_model_button = tk.Button(Frame_1, text="Change Model", command=lambda: self.change_model_1(Frame_2, model_file_entry))
        change_model_button.grid(row=1, column=1, padx=10, pady=5, sticky="w", ipadx=5, ipady=2)

        scale_label = ttk.Label(Frame_1, text='2. Confidence all:', font=('Segoe UI', 12))
        scale_label.grid(row=2, column=0, padx=10, pady=5, sticky="nws")

        scale = tk.Scale(Frame_1, from_=1, to=100, orient='horizontal', length=500)
        scale.grid(row=3, column=0, padx=30, pady=5, sticky="nws")

        camera_frame_display = ttk.Label(Frame_1, text='3. Display Camera', font=('Segoe UI', 12))
        camera_frame_display.grid(row=4, column=0, padx=10, pady=5, sticky="nws")

        camera_frame = ttk.LabelFrame(Frame_1, text=f"Camera 1", width=500, height=500)
        camera_frame.grid(row=5, column=0, padx=30, pady=5, sticky="nws")

        open_image_button = ttk.Button(Frame_1, text='Open Image', command=lambda: self.open_image(model_name_labels, join, ok_vars, num_inputs, ng_vars, wn_inputs, wx_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales))
        open_image_button.grid(row=6, column=0, padx=30, pady=5, sticky="w", ipadx=5, ipady=2)

        
    def option_1_parameters(self,Frame_2, model1):

        global model_name_labels, join, ok_vars, num_inputs, ng_vars, wn_inputs, wx_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales
        
        def ng_selected(self,row_widgets):
            ng_checkbox_var = row_widgets[4].var
            ok_checkbox_var = row_widgets[2].var
        
            if ng_checkbox_var.get() == 1:
                ok_checkbox_var.set(0)

        def ok_selected(self,row_widgets):
            ng_checkbox_var = row_widgets[4].var
            ok_checkbox_var = row_widgets[2].var
            
            if ok_checkbox_var.get() == 1:
                ng_checkbox_var.set(0)

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

            model_name_label = tk.Label(Frame_2, text=f'{model1.names[i1]}_1', fg='black', font=('Segoe UI', 12), width=20, anchor='w')
            row_widgets.append(model_name_label)
            model_name_labels.append(model_name_label)

            join_checkbox_var = tk.IntVar()
            join_checkbox = tk.Checkbutton(Frame_2, variable=join_checkbox_var, onvalue=1, offvalue=0)
            join_checkbox.var = join_checkbox_var
            join_checkbox.grid()
            row_widgets.append(join_checkbox)
            join.append(join_checkbox)

            ok_checkbox_var = tk.IntVar()
            ok_checkbox = tk.Checkbutton(Frame_2, variable=ok_checkbox_var, command=lambda rw=row_widgets: ok_selected(rw))
            ok_checkbox.var = ok_checkbox_var
            ok_checkbox.grid()
            row_widgets.append(ok_checkbox)
            ok_vars.append(ok_checkbox)

            num_input = tk.Entry(Frame_2, width=5)
            num_input.insert(0, '1')
            row_widgets.append(num_input)
            num_inputs.append(num_input)

            ng_checkbox_var = tk.IntVar()
            ng_checkbox = tk.Checkbutton(Frame_2, variable=ng_checkbox_var, command=lambda rw=row_widgets: ng_selected(rw))
            ng_checkbox.var = ng_checkbox_var
            ng_checkbox.grid()
            row_widgets.append(ng_checkbox)
            ng_vars.append(ng_checkbox)

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

def main():
    window = tk.Tk()
    window.title("Data Entry Form")
    window.state('zoomed')
    notebook = ttk.Notebook(window)
    notebook.pack(fill="both", expand=True)
    camera1_frame, camera2_frame = Display_Camera(notebook)
    Camera_1_Settings = Model_Camera_1(notebook)
    Camera_1_Settings.Camera_1_Settings()
    update_images(window,camera1_frame, camera2_frame)
    window.mainloop()

if __name__ == "__main__":
    main()
