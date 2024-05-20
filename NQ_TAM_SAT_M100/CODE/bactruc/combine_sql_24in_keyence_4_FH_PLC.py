import socket
from time import sleep


soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#soc.settimeout(5)

def socket_connect(host, port):
    try:
        soc.connect((host, port))
        return True
    except OSError:
        print("Can't connect to PLC")
        sleep(3)
        print("Reconnecting....")
        return False

def readdata(data):
    a = 'RD '
    c = '\x0D'
    d = a+ data +c
    datasend = d.encode("UTF-8")
    soc.sendall(datasend)
    data = soc.recv(1024)
    datadeco = data.decode("UTF-8")
    data1 = int(datadeco)

    return data1

#Write data
def writedata(register, data):
    a = 'WR '
    b = ' '
    c = '\x0D'
    d = a+ register + b + str(data) + c
    datasend  = d.encode("UTF-8")
    soc.sendall(datasend)
    datares = soc.recv(1024)
    #print(datares)


connected = False
while connected == False:
    connected = socket_connect('192.168.0.10',8501)
print("connected") 
print(readdata('DM2008'))
writedata('DM2010.U',2000) 
