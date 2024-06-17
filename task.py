# import tkinter as tk
# from tkinter import ttk, filedialog
# from PIL import Image, ImageTk
# import glob
# import torch
# import mysql.connector
# # def display_images(camera_frame, camera_number):
# #     for widget in camera_frame.winfo_children():
# #         widget.destroy()
    
# #     image_paths = glob.glob(f"C:/Users/CCSX009/Documents/yolov5/test_image/camera{camera_number}/*.jpg")
# #     if image_paths:
# #         image_path = image_paths[-1]
# #         image = Image.open(image_path)
# #         image = image.resize((800, 800), Image.LANCZOS)
# #         photo = ImageTk.PhotoImage(image)
# #         image_label = tk.Label(camera_frame, image=photo)
# #         image_label.image = photo
# #         image_label.pack()

# # def update_images(camera1_frame, camera2_frame):
# #     display_images(camera1_frame, 1)
# #     display_images(camera2_frame, 2)
# #     # window.after(100, update_images, camera1_frame, camera2_frame)

# # def print_widget_values(wx_inputs, model_name_labels, model_checkboxes, ok_radios, num_inputs, ng_radios, wn_inputs,hn_inputs, hx_inputs, plc_inputs, conf_scales):
# #     model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5', 'custom', path="C:/Users/CCSX009/Documents/yolov5/file_train/X75_M100_DEN_CAM1_2024-03-11.pt", source='local', force_reload=False)
# #     for i1 in range(len(model1.names)):
# #         print("Values for", model1.names[i1])
# #         print("Model Name Label Text:", model_name_labels[i1].cget("text"))
# #         print("Model Checkbox State:", model_checkboxes[i1].instate(["selected"]))
# #         print("OK Radio Value:", ok_radios[i1].cget("value"))
# #         print("Num Input Value:", num_inputs[i1].get())
# #         print("NG Radio Value:", ng_radios[i1].cget("value"))
# #         print("WN Input Value:", wn_inputs[i1].get())
# #         print("WX Input Value:", wx_inputs[i1].get())
# #         print("HN Input Value:", hn_inputs[i1].get())
# #         print("HX Input Value:", hx_inputs[i1].get())
# #         print("PLC Input Value:", plc_inputs[i1].get())
# #         print("Conf Scale Value:", conf_scales[i1].get())
# #         print()

# # def open_image(wx_inputs, model_name_labels, model_checkboxes, ok_radios, num_inputs, ng_radios, wn_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales):
# #     print_widget_values(wx_inputs, model_name_labels, model_checkboxes, ok_radios, num_inputs, ng_radios, wn_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales)

# # def create_camera_frame(notebook, camera_number):
# #     camera_frame = ttk.LabelFrame(notebook, text=f"Camera {camera_number}", width=800, height=800)
# #     camera_frame.grid(row=0, column=camera_number-1, padx=50, pady=20, sticky="news")
# #     time_frame = ttk.LabelFrame(notebook, text=f"Time Processing Camera {camera_number}", width=400, height=100)
# #     time_frame.grid(row=1, column=camera_number-1, padx=50, pady=20, sticky="news")
# #     return camera_frame

# # def create_tab1(notebook):
# #     tab1 = ttk.Frame(notebook)
# #     notebook.add(tab1, text="Display Cameras")

# #     camera1_frame = create_camera_frame(tab1, 1)
# #     camera2_frame = create_camera_frame(tab1, 2)

# #     return camera1_frame, camera2_frame

# # def create_tab2(notebook):
# #     global wx_inputs, model_name_labels, model_checkboxes, ok_radios, num_inputs, ng_radios, wn_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales
# #     tab2 = ttk.Frame(notebook)
# #     notebook.add(tab2, text="Configuration")

# #     open_button = tk.Button(tab2, text="Open File Browser", command=open_file_browser)
# #     open_button.pack(pady=20)

# #     open_image_button = ttk.Button(tab2, text='Open Image', command=lambda: open_image(wx_inputs, model_name_labels, model_checkboxes, ok_radios, num_inputs, ng_radios, wn_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales))
# #     open_image_button.pack(pady=10)

# #     open_button.place(relx=0.1, rely=0.1, anchor=tk.CENTER)
# #     global label_result
# #     label_result = ttk.Label(tab2, text="")
# #     label_result.pack(pady=10)
    
# #     confidence_label = ttk.Label(tab2, text='Confidence',  foreground='black')
# #     confidence_label.pack(pady=5)
# #     confidence_scale = ttk.Scale(tab2, from_=1, to=100, length=200)
# #     confidence_scale.set(30)
# #     confidence_scale.pack(pady=5)

# #     model1 = torch.hub.load('C:/Users/CCSX009/Documents/yolov5', 'custom', path="C:/Users/CCSX009/Documents/yolov5/file_train/X75_M100_DEN_CAM1_2024-03-11.pt", source='local', force_reload=False)
# #     wx_inputs = []
# #     model_name_labels = []
# #     model_checkboxes = []
# #     ok_radios = []
# #     num_inputs = []
# #     ng_radios = []
# #     wn_inputs = []
   
# #     hn_inputs = []
# #     hx_inputs = []
# #     plc_inputs = []
# #     conf_scales = []

# #     widgets = []

# #     for i1 in range(len(model1.names)):
# #         row_widgets = []
# #         model_frame = ttk.Frame(tab2)
# #         model_frame.pack(pady=5)
        
# #         model_name_label = ttk.Label(model_frame, text=f'{model1.names[i1]}_1', font=('Helvetica', 15), foreground='black')
# #         model_name_label.grid(row=0, column=0)
# #         row_widgets.append(model_name_label)
# #         model_name_labels.append(model_name_label)
        
# #         model_checkbox = ttk.Checkbutton(model_frame)
# #         model_checkbox.grid(row=0, column=1)
# #         model_checkboxes.append(model_checkbox)
# #         model_checkboxes.append(model_checkbox)
        
# #         ok_radio = ttk.Radiobutton(model_frame, value='OK')
# #         ok_radio.grid(row=0, column=2)
# #         row_widgets.append(ok_radio)
# #         ok_radios.append(ok_radio)
        
# #         num_input = ttk.Entry(model_frame, width=5)
# #         num_input.insert(0, '1')
# #         num_input.grid(row=0, column=3)
# #         row_widgets.append(num_input)
# #         num_inputs.append(num_input)
        
# #         ng_radio = ttk.Radiobutton(model_frame, value='NG')
# #         ng_radio.grid(row=0, column=4)
# #         row_widgets.append(ng_radio)
# #         ng_radios.append(ng_radio)
        
# #         wn_input = ttk.Entry(model_frame, width=7)
# #         wn_input.insert(0, '0')
# #         wn_input.grid(row=0, column=5)
# #         row_widgets.append(wn_input)
# #         wn_inputs.append(wn_input)
        
# #         wx_input = ttk.Entry(model_frame, width=7)
# #         wx_input.insert(0, '1600')
# #         wx_input.grid(row=0, column=6)
# #         row_widgets.append(wx_input)
# #         wx_inputs.append(wx_input)
        
# #         hn_input = ttk.Entry(model_frame, width=7)
# #         hn_input.insert(0, '0')
# #         hn_input.grid(row=0, column=7)
# #         row_widgets.append(hn_input)
# #         hn_inputs.append(hn_input)
        
# #         hx_input = ttk.Entry(model_frame, width=7)
# #         hx_input.insert(0, '1200')
# #         hx_input.grid(row=0, column=8)
# #         row_widgets.append(hx_input)
# #         hx_inputs.append(hx_input)
        
# #         plc_input = ttk.Entry(model_frame, width=7)
# #         plc_input.insert(0, '0')
# #         plc_input.grid(row=0, column=9)
# #         row_widgets.append(plc_input)
# #         plc_inputs.append(plc_input)
        
# #         conf_scale = ttk.Scale(model_frame, from_=1, to=100, length=200)
# #         conf_scale.grid(row=0, column=10)
# #         row_widgets.append(conf_scale)
# #         conf_scales.append(conf_scale)
        
# #     for i, row in enumerate(widgets):
# #         for j, widget in enumerate(row):
# #             widget.grid(row=i+1, column=j, padx=15, pady=5, sticky="w", ipadx=2, ipady=2)

# #     # open_image_button = ttk.Button(tab2, text='Open Image', command=lambda: open_image(wx_inputs, model_name_labels, model_checkboxes, ok_radios, num_inputs, ng_radios, wn_inputs, hn_inputs, hx_inputs, plc_inputs, conf_scales))
# #     # open_image_button.pack(pady=10)

    

# # def open_file_browser():
# #     selected_file = filedialog.askopenfilename(title="Choose a file", filetypes=[("All Files", "*.*")])
# #     label_result.config(text=selected_file) 

# # def main():
# #     window = tk.Tk()
# #     window.title("Data Entry Form")
# #     window.state('zoomed')
# #     notebook = ttk.Notebook(window)
# #     notebook.pack(fill="both", expand=True)

# #     camera1_frame, camera2_frame = create_tab1(notebook)
# #     create_tab2(notebook)

# #     update_images(camera1_frame, camera2_frame)

# #     window.mainloop()

# # if __name__ == "__main__":
# #     main()

# def Connect_MySQLServer():
#         db_connection = mysql.connector.connect(
#         host="127.0.0.1",
#         user="root1", 
#         passwd="987654321",
#         database="model_1")                    
#         cursor = db_connection.cursor()
#         return cursor,db_connection

# def load_data_model_1():
#     item_code = "ABC2D3F"
#     cursor, db_connection = Connect_MySQLServer()
    
#     # Lấy tất cả dữ liệu
#     cursor.execute("SELECT * FROM test_model_cam1_model1 WHERE item_code = %s", (item_code,))
    
#     # Lấy tất cả các hàng vào một danh sách
#     records = cursor.fetchall()
#     cursor.close()
#     db_connection.close()
    
#     if records:
#         # Truy cập hàng đầu tiên
#         first_record = records[0]
#         load_item_code = first_record[1]
#         load_path_weight = first_record[2]
#         load_confidence_all_scale = first_record[3]
        
   
        
#         # In các bản ghi còn lại (bao gồm cả hàng đầu tiên nếu cần)
#         for record in records:
#             print(record[4])



# print(load_data_model_1())
    
# # def all():
# #     result = load_data_model_1()
# #     for i1 in range(len(model1.names)): 
# #         for i2 in result:
# #             if i2[4] == i1
         


# #         for i1 in range(len(model1.names)):
# #             row_widgets = []

# #             model_name_label = tk.Label(Frame_2, text=f'{model1.names[i1]}', fg='black', font=('Segoe UI', 12), width=20, anchor='w')
# #             row_widgets.append(model_name_label)
# #             model_name_labels.append(model_name_label)

# #             join_checkbox_var = tk.BooleanVar()
# #             join_checkbox = tk.Checkbutton(Frame_2, variable=join_checkbox_var, onvalue=True, offvalue=False)
# #             join_checkbox.grid()
# #             join_checkbox.var = join_checkbox_var
# #             row_widgets.append(join_checkbox)
# #             join.append(join_checkbox_var)

# #             ok_checkbox_var = tk.BooleanVar()
# #             ok_checkbox = tk.Checkbutton(Frame_2, variable=ok_checkbox_var, onvalue=True, offvalue=False, command=lambda rw=row_widgets:ok_selected(rw))
# #             ok_checkbox.grid()
# #             ok_checkbox.var = ok_checkbox_var
# #             row_widgets.append(ok_checkbox)
# #             ok_vars.append(ok_checkbox_var)

# #             ng_checkbox_var = tk.BooleanVar()
# #             ng_checkbox = tk.Checkbutton(Frame_2, variable=ng_checkbox_var, onvalue=True, offvalue=False, command=lambda rw=row_widgets:ng_selected(rw))
# #             ng_checkbox.grid()
# #             ng_checkbox.var = ng_checkbox_var
# #             row_widgets.append(ng_checkbox)
# #             ng_vars.append(ng_checkbox_var)

# #             num_input = tk.Entry(Frame_2, width=7)
# #             num_input.insert(0, '1')
# #             row_widgets.append(num_input)
# #             num_inputs.append(num_input)

# #             wn_input = tk.Entry(Frame_2, width=7)
# #             wn_input.insert(0, '0')
# #             row_widgets.append(wn_input)
# #             wn_inputs.append(wn_input)

# #             wx_input = tk.Entry(Frame_2, width=7)
# #             wx_input.insert(0, '1600')
# #             row_widgets.append(wx_input)
# #             wx_inputs.append(wx_input)

# #             hn_input = tk.Entry(Frame_2, width=7)
# #             hn_input.insert(0, '0')
# #             row_widgets.append(hn_input)
# #             hn_inputs.append(hn_input)

# #             hx_input = tk.Entry(Frame_2, width=7)
# #             hx_input.insert(0, '1200')
# #             row_widgets.append(hx_input)
# #             hx_inputs.append(hx_input)

# #             plc_input = tk.Entry(Frame_2, width=7)
# #             plc_input.insert(0, '0')
# #             row_widgets.append(plc_input)
# #             plc_inputs.append(plc_input)

# #             conf_scale = tk.Scale(Frame_2, from_=1, to=100, orient='horizontal', length=280)
# #             row_widgets.append(conf_scale)
# #             conf_scales.append(conf_scale)

# #             widgets.append(row_widgets)

from subprocess import Popen, PIPE
from threading import Thread, Lock
import tkinter as tk


# class TkinterPopen(tk.Text):
#     def __init__(self, master, state="disabled", **kwargs):
#         super().__init__(master, state=state, **kwargs)
#         self.commands = []
#         self.proc = None
#         self.running = True
#         self.stdout_buffer = ""
#         self.stdout_buffer_lock = Lock()

#     def stdout_loop(self, last_loop:bool=False) -> None:
#         with self.stdout_buffer_lock:
#             # Get the data and clear the buffer:
#             data, self.stdout_buffer = self.stdout_buffer, ""
#         state = super().cget("state")
#         super().config(state="normal")
#         super().insert("end", data)
#         super().see("end")
#         super().config(state=state)
#         if self.proc is None:
#             if len(self.commands) == 0:
#                 # If we are done with all of the commands:
#                 if last_loop:
#                     return None
#                 super().after(100, self.stdout_loop, True)
#             else:
#                 # If we have more commands to do call `start_next_proc`
#                 self.start_next_proc()
#         else:
#             super().after(100, self.stdout_loop)

#     def start_next_proc(self) -> None:
#         command = self.commands.pop(0) # Take the first one from the list
#         self.proc = Popen(command, stdout=PIPE)
#         new_thread = Thread(target=self.read_stdout, daemon=True)
#         new_thread.start()
#         self.stdout_loop()

#     def run_commands(self, commands:list) -> None:
#         self.commands = commands
#         self.start_next_proc()

#     def read_stdout(self):
#         while self.proc.poll() is None:
#             self._read_stdout()
#         self._read_stdout()
#         self.proc = None

#     def _read_stdout(self) -> None:
#         line = self.proc.stdout.readline()
#         with self.stdout_buffer_lock:
#             self.stdout_buffer += line.decode()


# if __name__ == "__main__":
#     def start_echo():
#         command = ["echo", "hi"]
#         tkinter_popen.run_commands([command])

#     def start_ping():
        
#         command = ["ping", "1.1.1.1", "-n", "3"]
#         tkinter_popen.run_commands([command])

#     root = tk.Tk()

#     tkinter_popen = TkinterPopen(root)
#     tkinter_popen.pack()

#     button = tk.Button(root, text="Run echo", command=start_echo)
#     button.pack()

#     button = tk.Button(root, text="Run ping", command=start_ping)
#     button.pack()

#     root.mainloop()
import tkinter as tk
from tkinter import scrolledtext
import sys
import threading
import time

class ConsoleRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

class ConsoleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Embedded Console in Tkinter")

        # Create a frame
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create a scrolled text widget
        self.console = scrolledtext.ScrolledText(self.frame, wrap=tk.WORD, height=20, width=80)
        self.console.pack(fill=tk.BOTH, expand=True)

        # Redirect stdout and stderr
        sys.stdout = ConsoleRedirector(self.console)
        sys.stderr = ConsoleRedirector(self.console)

        # Button to run a test function
        self.run_button = tk.Button(self.frame, text="Run Test", command=self.run_test)
        self.run_button.pack()

    def run_test(self):
        def test_function():
            for i in range(10):
                print(f"Test message {i}")
                time.sleep(1)

        threading.Thread(target=test_function).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = ConsoleApp(root)
    root.mainloop()