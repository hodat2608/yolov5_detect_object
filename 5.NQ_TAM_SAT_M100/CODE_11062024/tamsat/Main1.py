# combine_sql_24in_omron_4_FH_mul_conf.py
from myutils1 import run_model, load_choosemodel, load_theme, make_window, load_all_sql, save_data, remove_file,write_plc,read_plc
from myutils1 import update_disabled_control, change_disabled_administrator, add_file_browser, layout_theme, choose_model_event
from myutils1 import check_model, camera, detect_model, picture, set_param_before_choose_model_cam, row_for_var, var_for_interface
from myutils1 import program_camera_FH_c4,update_value_model, login_administrator,change_model,load_kc,detect_model_4,program_camera_FH,run_plc_keyence
import PySimpleGUI as sg
import traceback
from myutils1 import read_plc, write_plc, excel_ppbydate, excel_hangmuc, excel_ray, excel_handle_pphangmuc, excel_handle_ppray, excel_handle_all
from myutils1 import collect_dict_date,logging, CMyCallback, setup_camera_stc,program_camera, kiem_guong
from myutils1 import config_off_auto, set_config_init_camera, program_camera_test
from datetime import date
import csv
import os
import stapipy as st
import keyboard

nums_camera = 2
dir_img1 = ''
dir_img2 = ''



models = []
for i in range(1,nums_camera+2):
    models.append(run_model(i))

choose_model = load_choosemodel()

plc_name = 'k'
themes = load_theme()
theme = themes[0]
window = make_window(theme, models, nums_camera, choose_model)
window['choose_model'].update(value=choose_model)

try:
    for num_camera in range(1,nums_camera+2):
        dict_date = collect_dict_date(num_camera,nums_camera+1)
        window[f'date_save{num_camera}'].update(values=dict_date[choose_model])
except:
    # print(traceback.format_exc())
    pass

for i in range(1, nums_camera +2):
    try:
        load_all_sql(i, choose_model,window, nums_camera+1)
    except:            
        print(traceback.format_exc())
        if i > 2:
            window[f'time_cam{i-2}'].update(value= "Error") 
        else:
            window[f'time_cam{i}'].update(value= "Error") 

# try:
#     load_kc(window)
# except:
#     pass
# for i in range(1, nums_camera+1):
#     remove_file(f"C:/FH/camera{i}/")

# moplc
run_plc_keyence("192.168.1.1", 8501)
# write_plc('k','3150',1)
################end

ready_c1 = '10005'
ready_c2 = '10015'
plc_logging_2c = '10020'
logging_cam1 = '10007' # ok:1, ng:2
logging_cam2 = '10017' # ok:1, ng:2
trigger_kiemguong_c1 = '10022'
trigger_kiemguong_c2 = '10024'
result_kiemguong_c1 = '10002' # ok:1, ng:2
result_kiemguong_c2 = '10012' # ok:1, ng:2
trigger_c1 = "10000"
trigger_c2 = "10010"
hoantat_c1 = "10030"
hoantat_c2 = "10040"
during_kiem_guong = "10060"
during_chay_hang = "10062"
do_sang_min = 0
do_sang_max = 255
do_net_min = 0
do_net_max = 256
center_x_min = 400 
center_x_max = 555
center_y_min = 300
center_y_max = 465



try:
    my_callback1 = CMyCallback()
    cb_func1 = my_callback1.datastream_callback1
    connect_camera1 = True
    write_plc('k',ready_c1,1)


except Exception as exception:
    print('Error 1: ',exception)
    window['result_cam1'].update(value= 'Error', text_color='red')
    # window['Reconnect1'].update(disabled= False)
    write_plc('k',ready_c1,0)


try:
    my_callback2 = CMyCallback()
    cb_func2 = my_callback2.datastream_callback2
    connect_camera2 = True
    write_plc('k',ready_c2,1)

except Exception as exception:
    print('Error 2: ',exception)
    window['result_cam2'].update(value= 'Error', text_color='red')
    # window['Reconnect2'].update(disabled= False)
    write_plc('k',ready_c2,0)





connect_total = False
 
try:
    st.initialize()
    st_system = st.create_system()
    connect_total = True

except Exception as exception:
    print('Error total: ',exception)
    window['result_cam1'].update(value= 'Error', text_color='red')
    window['result_cam2'].update(value= 'Error', text_color='red')

value_exposure_time = 20105.91
value_gain = 51.000000

value_balance_red = 0
value_balance_green = 0
value_balance_blue = 0
check_kiem_guong = 0
try:
    st_datastream1, st_device1, remote_nodemap1= setup_camera_stc("1",st_system,cb_func1,window)
    config_off_auto(remote_nodemap1)
    set_config_init_camera(remote_nodemap1,value_exposure_time, value_gain,'off', value_balance_red,value_balance_green, value_balance_blue)

    write_plc('k',ready_c1,1)
except:
    window['result_cam1'].update(value= 'Error', text_color='red')
    # window['Reconnect1'].update(disabled= False)
    write_plc('k',ready_c1,0)


try:
    st_datastream2, st_device2, remote_nodemap2= setup_camera_stc("2",st_system,cb_func2,window)
    config_off_auto(remote_nodemap2)
    set_config_init_camera(remote_nodemap2,value_exposure_time, value_gain,'off', value_balance_red,value_balance_green, value_balance_blue)
    # set_config_init_camera(remote_nodemap2,value_exposure_time, value_gain, value_balance_red,value_balance_green, value_balance_blue)

    write_plc('k',ready_c2,1)
except:
    window['result_cam2'].update(value= 'Error', text_color='red')
    # window['Reconnect2'].update(disabled= False)
    write_plc('k',ready_c2,0)


try:
    while True:
        event, values = window.read(timeout=1)

        if event == 'Change Theme':
            window = layout_theme(window, models, nums_camera, choose_model)

        #moplc
        # if read_plc('k',3160) == 1:  
        #     for i in range(1, nums_camera+1):
        #         remove_file(f"C:/FH/camera{i}/")
        #     write_plc('k',3160,0)
        #################end
            

        for i, model in enumerate(models):
            add_file_browser(event, window, values, i+1)   
            save_data(window,event, model, i+1, choose_model,nums_camera+1,values)
  
        if event == 'Change_1':
            window = change_model(window, values, 1, nums_camera, theme, models, values['choose_model'])
        if event == 'Change_2':
            window = change_model(window, values, 2, nums_camera, theme, models, values['choose_model'])
        if event == 'Change_3':
            window = change_model(window, values, 3, nums_camera, theme, models, values['choose_model'])
        login_administrator(window, event, values, models)


        if read_plc('k',plc_logging_2c) == 1:
            logging(window,values,models[0],"C:/logging/1/",'k',1,logging_cam1)
            logging(window,values,models[1],"C:/logging/2/",'k',2,logging_cam2)
            write_plc('k',plc_logging_2c,0) 

  
        if (read_plc('k',during_kiem_guong) == 1 and check_kiem_guong ==0) or keyboard.is_pressed('shift+c'): 
            check_kiem_guong = 1
            value_exposure_time = 8508.300000
            value_gain = 5.000000
            config_off_auto(remote_nodemap1)
            set_config_init_camera(remote_nodemap1,value_exposure_time, value_gain,'off', value_balance_red,value_balance_green, value_balance_blue)
            value_exposure_time = 9768.91
            config_off_auto(remote_nodemap2)
            set_config_init_camera(remote_nodemap2,value_exposure_time, value_gain,'off', value_balance_red,value_balance_green, value_balance_blue)

        if (read_plc('k',during_chay_hang) == 1 and check_kiem_guong ==1) or keyboard.is_pressed('shift+d'):  
            check_kiem_guong =0  
            value_exposure_time = 20105.91
            value_gain = 51.000000
            config_off_auto(remote_nodemap1)
            set_config_init_camera(remote_nodemap1,value_exposure_time, value_gain,'off', value_balance_red,value_balance_green, value_balance_blue)
            config_off_auto(remote_nodemap2)
            set_config_init_camera(remote_nodemap2,value_exposure_time, value_gain,'off', value_balance_red,value_balance_green, value_balance_blue)


        kiem_guong(window,values,models[2],plc_name,31,my_callback1,trigger_kiemguong_c1, result_kiemguong_c1,do_sang_min,do_sang_max, do_net_min,do_net_max,center_x_min,center_x_max,center_y_min,center_y_max)
        kiem_guong(window,values,models[2],plc_name,32,my_callback2,trigger_kiemguong_c2, result_kiemguong_c2,do_sang_min,do_sang_max, do_net_min,do_net_max,center_x_min,center_x_max,center_y_min,center_y_max)


        try:
            program_camera(window,values,models[0],'k',1,my_callback1,trigger_c1, hoantat_c1)
        except:
            write_plc('k',ready_c1,0) 
        try:
            program_camera(window,values,models[1],'k',2,my_callback2,trigger_c2, hoantat_c2)
        except:
            write_plc('k',ready_c2,0) 



        program_camera_test(window,values,models[0],'k',1,my_callback1,'shift+a')

        program_camera_test(window,values,models[1],'k',2,my_callback2,'shift+b')
    




        # program_camera(window,values,models[0],'k',2,"10000", "10002")


        #moplc
        # program_camera_FH(window,values,models[0],"C:/FH/camera1/",'k',1,"1930", "1950")
        # program_camera_FH(window,values,models[1],"C:/FH/camera2/",'k',2,"1932", "1952")
        # program_camera_FH(window,values,models[2],"C:/FH/camera3/",'k',3,"1934", "1954")
        # program_camera_FH_c4(window,values,models[3],"C:/FH/camera4/",'k',"1936", "1956")
        # program_camera_FH(window,values,models[4],"C:/FH/camera5/",'k',5,"1938", "1958")
        # program_camera_FH(window,values,models[5],"C:/FH/camera6/",'k',6,"1940", "1960")
        # program_camera_FH(window,values,models[6],"C:/FH/camera7/",'k',7,"1942", "1962")
        #################end


        if event == 'choose_model':
            window = choose_model_event(window, values, nums_camera,theme,values['choose_model'], False)

        try:
            for num_camera in range(1, nums_camera +1):
                if event == f'date_save{num_camera}':
                    get_date = values[f'date_save{num_camera}']
                    window,models = choose_model_event(window, values,nums_camera, theme,values['choose_model'], True)
                    window[f'date_save{num_camera}'].update(value=get_date)
        except: 
            pass

        #moplc
        # try:
        # if read_plc(plc_name,10052) == 1 or keyboard.is_pressed('shift+i'):
        #     excel_ray('Ngay')
        #     excel_hangmuc('Ngay')
        #     write_plc(plc_name,10052,0)
        # if read_plc(plc_name,10054) == 1  or keyboard.is_pressed('shift+j'):
        #     excel_ray('Dem')
        #     excel_hangmuc('Dem')
        #     write_plc(plc_name,10054,0)



            # excel_handle_all(plc_name, 6054)   
        # except:
        #     pass
            # print(traceback.print_exc())
        excel_handle_pphangmuc(plc_name, 10058)
        excel_handle_ppray(plc_name, 10056)    

        try:
            # excel_handle_pphangmuc(plc_name, 10058)
            # excel_handle_ppray(plc_name, 10056)
            if read_plc(plc_name,20400) == 1:
                for var_plc in range(0,3):
                    # tu 4100 den 4104
                    if read_plc(plc_name,20401 + var_plc) == 1:
                        excel_ppbydate(var_plc,"")
                        write_plc(plc_name,20401 + var_plc,0) 

                    # if read_plc(plc_name,4110 + var_plc) == 1:
                    #     excel_ppbydate(var_plc,"CAM 2")
                    #     write_plc(plc_name,4110 + var_plc,0) 

                    # if read_plc(plc_name,4120 + var_plc) == 1:
                    #     excel_ppbydate(var_plc,"CAM 3")
                    #     write_plc(plc_name,4120 + var_plc,0) 

                    # if read_plc(plc_name,4130 + var_plc) == 1:
                    #     excel_ppbydate(var_plc,"CAM 4")
                    #     write_plc(plc_name,4130 + var_plc,0) 

                    # if read_plc(plc_name,4140 + var_plc) == 1:
                    #     excel_ppbydate(var_plc,"CAM 5")
                    #     write_plc(plc_name,4140 + var_plc,0) 

                    # if read_plc(plc_name,4150 + var_plc) == 1:
                    #     excel_ppbydate(var_plc,"CAM 6")
                    #     write_plc(plc_name,4150 + var_plc,0) 

                    # if read_plc(plc_name,4160 + var_plc) == 1:
                    #     excel_ppbydate(var_plc,"CAM 7")
                    #     write_plc(plc_name,4160 + var_plc,0)      


                write_plc(plc_name,20400,0) 
 
        except:               
            pass
        #################end


        # check_model(window, event, values, directory, model,size,conf,num_camera)

        # camera(window, event,nums_camera)

        if event == 'Pic1':
            dir_img1 = picture(window, 1)

        if event == f'Detect1':
            detect_model(window, dir_img1, models[0],values['choose_size1'],values['conf_thres1']/100,1,values)

        if event == 'Pic2':
            dir_img2 = picture(window, 2)

        if event == f'Detect2':
            detect_model(window, dir_img2, models[1],values['choose_size2'],values['conf_thres2']/100,2,values)


        if event =='Exit' or event == sg.WINDOW_CLOSED :
            #moplc
            write_plc('k',ready_c1,0) 
            write_plc('k',ready_c2,0) 
            #################end

            break
    window.close() 

except Exception as e:

    print(traceback.print_exc())
    str_error = str(e)
    sg.popup(str_error,font=('Helvetica',15), text_color='red',keep_on_top= True)


