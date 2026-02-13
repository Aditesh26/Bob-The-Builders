import serial

# change COM4 to your port
ser = serial.Serial('COM3', 9600)

while True:
    line = ser.readline().decode().strip()
    print(line)
