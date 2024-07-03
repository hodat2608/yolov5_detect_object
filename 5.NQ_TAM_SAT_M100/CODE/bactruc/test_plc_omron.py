from udp import UDPFinsConnection
#from init_test import FinsPLCMemoryAreas
from initialization import FinsPLCMemoryAreas
def connect_plc(host):
    global fins_instance
    try:
        fins_instance = UDPFinsConnection()
        fins_instance.connect(host)
        fins_instance.dest_node_add=1
        fins_instance.srce_node_add=25

        return True
    except:
        print("Can't connect to PLC")
        for i in range(100000000):
            pass
        #sleep(3)
        print("Reconnecting....")
        return False


connected = False
while connected == False:
    connected = connect_plc('192.168.1.50')
    print('connecting ....')
    #event, values = window.read(timeout=20)

print("connected plc")   
register_ng = (2008).to_bytes(2, byteorder='big') + b'\x00'
for i in range(1000):
    register_ng = (i).to_bytes(2, byteorder='big') + b'\x00'
    read_2000 = fins_instance.memory_area_read(FinsPLCMemoryAreas().DATA_MEMORY_WORD,register_ng)
    #fins_instance.change_to_run_mode()
    #fins_instance.change_to_program_mode()
    print(read_2000)
# read_4000 = fins_instance.memory_area_read(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x0F\xA0\x00')
# print(read_4000)
#fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,register_ng,b'\x00\x04',1)
#b'\xc0\x00\x02\x00\x19\x00\x00\x01\x00`\x01\x01\x90\x05\x002'
#b'\xc0\x00\x02\x00\x19\x00\x00\x01\x00`\x01\x01\x00\x00\x00\x05'
#b'\xc0\x00\x02\x00\x19\x00\x00\x01\x00`\x01\x01\x00\x00\x00\x05' = 5