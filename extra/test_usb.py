import serial 
ser = serial.Serial()

ser.port = "/dev/ttyUSB0" # may be called something different
ser.baudrate = 9600 # may be different
ser.open()
if ser.isOpen():
	# string = ["AM0;","TD8197;","FR50;","OF40000;"]
	string = ["AM8000;","TD8197;","FR30;","OF34904;"]
	for i in range(len(string)):
	    ser.write(string[i].encode())
	    response = ser.read(ser.inWaiting())
	    print(response)
