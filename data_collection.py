import serial
import threading
import sqlite3
from datetime import datetime
import time
import numpy 
import math
import pyaudio
import scipy
import pandas as pd
import statistics

TIME_S = 43200     #Total time for which data is collected
val1 = []
val2 =[]


for i in range(0,int(TIME_S)):

	conn = sqlite3.connect('sensors.db')       #initializing connection to the database
	c = conn.cursor()

	CHUNK = 1024    #number of frames in the buffer
	RATE  = 44100   #number of samples collected per second
	R_SECS= 1       #Recording duration of a cycle

	pa = pyaudio.PyAudio() #pyaudio instantiation

    #Create audio stream
	stream = pa.open(
        	format=pyaudio.paInt16,    # 16-bit resolution
        	channels=1,                # channel
        	input_device_index = 5,    #device index
        	rate = RATE,               #sampling rate
        	input=True,
        	frames_per_buffer=CHUNK)   #number of samples for buffer

#Function to collect data from the sensors connected to the arduino and sound loudness value
	def data_collect() :
		
		ser = serial.Serial('/dev/ttyACM0' , 9600)    #opening serial port
		while (True):
            
			for ii in range(0,int(R_SECS)):          #loop through the audio stream

				now = datetime.now()                    #timestamp
				d = now.strftime("%d/%m/%Y %H:%M:%S")
				
  	            #sound loudness value calculation
				rawsamps = stream.read(CHUNK,False)
				samps = numpy.frombuffer(rawsamps, dtype=numpy.int16)
				samps = numpy.array(samps, dtype=float)/32768.0
				ms = math.sqrt(numpy.sum(samps ** 2.0) / len(samps))
				if ms<10e-8:
					ms = 10e-8
					db=10.0 * math.log(ms, 10.0)
				else:
					db=10.0 * math.log(ms, 10.0)
                    
            
			   #reading seral line to obtain sensor values from arduino			
				data1 = str(ser.readline())        
				data1 = data1[2: -5]
				data1=data1.replace("  ", "/")
				pieces = data1.split('/')
				new_pieces = []
				for pieces in pieces:
					if pieces == " " :
						new_pieces.append('NAN')

					else:
						new_pieces.append(pieces )

				if len(new_pieces)==8:
					my_val=str(d)+' '+str(db)+' '+new_pieces[0]+' '+new_pieces[1]+' '+new_pieces[2]+' '+new_pieces[3]+' '+new_pieces[4]+' '+new_pieces[5]+' '+new_pieces[6]+' '+new_pieces[7]
					val1.append(my_val)  #appending the read sensor values to a list
				time.sleep(1)
				

#Function to record sound frequency values	
	def sound_data():
		print("recording")
		frames = []
		t=[]
	

		for ii in range(0, int(RATE/CHUNK*R_SECS)):       # loop through the audio stream 

			now = datetime.now()                          #timestamp
			dt = now.strftime("%d/%m/%Y %H:%M:%S")
			t.append(dt)
			frames.append(stream.read(CHUNK,False))
			
		print("finished recording")
		# stop the stream, close it, and terminate the pyaudio instantiation
		stream.stop_stream()
		stream.close()
		pa.terminate()

		new_2=[]
        
        #Applying fft and filtering
		for frames in frames:
			data = numpy.frombuffer(frames, numpy.int16)
			w = numpy.abs(numpy.fft.rfft(data))
			f = numpy.fft.rfftfreq(len(data))
			f=f*RATE
			w1=w[f>100]
			new_2.append(w1)
			f1=f[f>100]		

        #Retrieving the frequencies of 10 highest peaks, min, max frequencies and mean and standard deviation
		B= numpy.mean(new_2, axis=0)
        ind_2 = f1[numpy.argpartition(B, -10)[-10:]]             
		max_f1 = f1[B.argmax()]               
		min_f1 = f1[B.argmin()]
        mean_w1 = statistics.mean(B)
        stdev_w1 = statistics.stdev(B)

		time.sleep(1)
		my_val2=str(dt)+' '+str(ind_2[0])+' '+str(ind_2[1])+' '+str(ind_2[2])+' '+str(ind_2[3])+' '+str(ind_2[4])+' '+str(ind_2[5])+' '+str(ind_2[6])+' '+str(ind_2[7])+' '+str(ind_2[8])+' '+str(ind_2[9])+' '+str(max_f1)+' '+str(min_f1)+' '+str(mean_w1)+' '+str(stdev_w1)
		val2.append(my_val2)	#Appending the frequency values to a list
	
	#threading
	t1 = threading.Thread(target = data_collect)
	t2 = threading.Thread(target = sound_data)

	t1.start()
	t2.start()

	t2.join()
	print(len(val1))
	if (len(val1))>0 :
		s=val1[-1].split()
		
	print(len(val2))
	if (len(val2))>0 :
		s1=val2[-1].split()
		
    #Writing the recorded values to the database
	if (len(s)+len(s1))==27:
		c.execute('INSERT INTO test_7  VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(s1[0]+' '+s1[1],s[2],s[3],s[4],s[5],s[6],s[7],s[8],s[9],s[10],s1[2],s1[3],s1[4],s1[5],s1[6],s1[7],s1[8],s1[9],s1[10],s1[11],s1[12],s1[13],s1[14],s1[15]))
	conn.commit()
	t1.join()


	