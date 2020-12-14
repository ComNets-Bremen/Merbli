#importing libraries
import serial
import threading
import numpy as np
import math
import pyaudio
import scipy
import pandas as pd
import statistics
from keras.models import load_model
from sklearn.metrics import accuracy_score,precision_score, recall_score, classification_report
import pickle
from pickle import load

TIME_S = 1          #Total time for which data is collected
val1 = []
val2 =[]


for i in range(0,int(TIME_S)):

	CHUNK = 1024    #number of frames in the buffer
	RATE  = 44100   #number of samples collected per second
	R_SECS= 1       #Recording duration of a cycle
	

	pa = pyaudio.PyAudio()     #pyaudio instantiation

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
		
		ser = serial.Serial('/dev/ttyACM0' , 9600)        #opening serial port
		while (True):
			for ii in range(0,int(R_SECS)):      #loop through the audio stream
				

                #sound loudness value calculation
				rawsamps = stream.read(CHUNK,False)
				samps = np.frombuffer(rawsamps, dtype=np.int16)
				samps = np.array(samps, dtype=float)/32768.0
				ms = math.sqrt(np.sum(samps ** 2.0) / len(samps))
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
					my_val=str(db)+' '+new_pieces[0]+' '+new_pieces[1]+' '+new_pieces[2]+' '+new_pieces[3]+' '+new_pieces[4]+' '+new_pieces[5]+' '+new_pieces[6]+' '+new_pieces[7]
					val1.append(my_val)        #appending the read sensor values to a list
				time.sleep(1)
				

				

#Function to read sound frequency values		
	def sound_data():
		print("recording")
		frames = []
		t=[]
	

		for ii in range(0, int(RATE/CHUNK*R_SECS)):       # loop through stream 

			frames.append(stream.read(CHUNK,False))
			
		print("finished recording")
		# stop the stream, close it, and terminate the pyaudio instantiation
		stream.stop_stream()
		stream.close()
		pa.terminate()

		new_2=[]
		
        #Applying fft and filtering
		for frames in frames:
			data = np.frombuffer(frames, np.int16)
			w = np.abs(np.fft.rfft(data))
			f = np.fft.rfftfreq(len(data))
			f=f*RATE
			w1=w[f>100]
			new_2.append(w1)
			f1=f[f>100]		


        #Retrieving the frequencies of 10 highest peaks, min, max frequencies and mean and standard deviation
		B= np.mean(new_2, axis=0)
		ind_2 = f1[np.argpartition(B, -10)[-10:]]
		max_f1 = f1[B.argmax()]              
		min_f1 = f1[B.argmin()] 
		mean_w1 = statistics.mean(B) 		 
		stdev_w1 = statistics.stdev(B)
		time.sleep(1)

		my_val2=str(ind_2[0])+' '+str(ind_2[1])+' '+str(ind_2[2])+' '+str(ind_2[3])+' '+str(ind_2[4])+' '+str(ind_2[5])+' '+str(ind_2[6])+' '+str(ind_2[7])+' '+str(ind_2[8])+' '+str(ind_2[9])+' '+str(max_f1)+' '+str(min_f1)+' '+str(mean_w1)+' '+str(stdev_w1)
		val2.append(my_val2)	     #Appending the frequency values to a list
	
    #threading
	t1 = threading.Thread(target = data_collect)
	t2 = threading.Thread(target = sound_data)

	t1.start()
	t2.start()
	
	t2.join()

	if (len(val1))>0 :
		s=val1[-1].split()
			
	if (len(val2))>0 :
		s1=val2[-1].split()
	out1 =s+s1
	out2 = np.array(out1)
	out3 = np.reshape(out2, (1, 23))	

    #arranging the input to the model as a pandas dataframe	
	df = pd.DataFrame (out3 ,columns=['s','v','h','t','l','x','y','z','m','f_1_1','f_1_2','f_1_3','f_1_4','f_1_5','f_1_6','f_1_7','f_1_8','f_1_9','f_1_10','max_f','min_f','mean_f','std_f'])	
	df["f_1_1"] = pd.to_numeric(df["f_1_1"], downcast="float")
	df["f_1_3"] = pd.to_numeric(df["f_1_3"], downcast="float")
	df["f_1_5"] = pd.to_numeric(df["f_1_5"], downcast="float")
	df["f_1_7"] = pd.to_numeric(df["f_1_7"], downcast="float")
	df["f_1_9"] = pd.to_numeric(df["f_1_9"], downcast="float")

	#bins for f_1
	conditions = [(df['f_1_1'] <=200),
			(df['f_1_1'] <= 300) & (df['f_1_1'] > 200),
              		(df['f_1_1'] <= 400) & (df['f_1_1'] > 300),
              		(df['f_1_1'] <= 500) & (df['f_1_1'] > 400),
              		(df['f_1_1'] <= 600) & (df['f_1_1'] > 500),
              		(df['f_1_1'] <= 700) & (df['f_1_1'] > 600),
              		(df['f_1_1'] <= 800) & (df['f_1_1'] > 700),
              		(df['f_1_1'] <= 3000) & (df['f_1_1'] > 800),]
	choices = ['1','2','3','4','5','6','7','8']
	df['f_1_1_bin'] = np.select(conditions, choices, default='0')


	#bins for f_3
	conditions_2 =[(df['f_1_3'] <=200),
               		(df['f_1_3'] <= 300) & (df['f_1_3'] > 200),
               		(df['f_1_3'] <= 400) & (df['f_1_3'] > 300),
               		(df['f_1_3'] <= 500) & (df['f_1_3'] > 400),
               		(df['f_1_3'] <= 600) & (df['f_1_3'] > 500),
               		(df['f_1_3'] <= 700) & (df['f_1_3'] > 600),
               		(df['f_1_3'] <= 800) & (df['f_1_3'] > 700),
               		(df['f_1_3'] <= 3000) & (df['f_1_3'] > 800),]
	choices_2 = ['1','2','3','4','5','6','7','8']
	df['f_1_3_bin'] = np.select(conditions_2, choices_2, default='0')


	#bins for f_5
	conditions_4 = [(df['f_1_5'] <=200),
                	(df['f_1_5'] <= 300) & (df['f_1_5'] > 200),
                	(df['f_1_5'] <= 400) & (df['f_1_5'] > 300),
                	(df['f_1_5'] <= 500) & (df['f_1_5'] > 400),
                	(df['f_1_5'] <= 600) & (df['f_1_5'] > 500),
                	(df['f_1_5'] <= 700) & (df['f_1_5'] > 600),
                	(df['f_1_5'] <= 800) & (df['f_1_5'] > 700),
                	(df['f_1_5'] <= 3000) & (df['f_1_5'] > 800),]
	choices_4 = ['1','2','3','4','5','6','7','8']
	df['f_1_5_bin'] = np.select(conditions_4, choices_4, default='0')


	#bins for f_7
	conditions_6 = [(df['f_1_7'] <=200),
                	(df['f_1_7'] <= 300) & (df['f_1_7'] > 200),
                	(df['f_1_7'] <= 400) & (df['f_1_7'] > 300),
                	(df['f_1_7'] <= 500) & (df['f_1_7'] > 400),
                	(df['f_1_7'] <= 600) & (df['f_1_7'] > 500),
                	(df['f_1_7'] <= 700) & (df['f_1_7'] > 600),
                	(df['f_1_7'] <= 800) & (df['f_1_7'] > 700),
                	(df['f_1_7'] <= 3000) & (df['f_1_7'] > 800),]
	choices_6 = ['1','2','3','4','5','6','7','8']
	df['f_1_7_bin'] = np.select(conditions_6, choices_6, default='0')


	#bins for f_9
	conditions_8 = [(df['f_1_9'] <=200),
                	(df['f_1_9'] <= 300) & (df['f_1_9'] > 200),
                	(df['f_1_9'] <= 400) & (df['f_1_9'] > 300),
                	(df['f_1_9'] <= 500) & (df['f_1_9'] > 400),
                	(df['f_1_9'] <= 600) & (df['f_1_9'] > 500),
                	(df['f_1_9'] <= 700) & (df['f_1_9'] > 600),
                	(df['f_1_9'] <= 800) & (df['f_1_9'] > 700),
                	(df['f_1_9'] <= 3000) & (df['f_1_9'] > 800),]
	choices_8 = ['1','2','3','4','5','6','7','8']
	df['f_1_9_bin'] = np.select(conditions_8, choices_8, default='0')
	

	# load the model - uncomment according to the used model
	print("loading model...")
    
    #For ANN
	#from keras.models import load_model
	#model = load_model('best_model_ANN.h5')
    
    #other classifiers
	model = pickle.load(open('finalized_model_new.sav', 'rb'))
	
    #Defining and reshaping the test input - X_test
	#X_test = df.iloc[0, [0,2,3,19,20,21,22,23,24,25,26,27]].values    #sound and DHT11 sensors as inputs
	X_test = df.iloc[0, [0,2,3,8,19,20,21,22,23,24,25,26,27]].values  #sound, DHT11 and motion sensors as inputs
    X_test = np.reshape(X_test, (1, -1))
    
    #loading the model scaling parametrs and applying feature scaling to test input
	scaler = load(open('scaler_new.pkl', 'rb'))
	X_test = scaler.transform(X_test)

	#prediction - individual events
	pred = model.predict(X_test) 
	#pred = pred.argmax(axis=1)     #uncomment for ANN
	if pred == 0:
		label = "blender"
	elif pred == 1:
		label = "Kettle"
	elif pred == 2:
		label = "Microwave"
	elif pred == 3 :
		label = "None"
	else:
		label = "Rice Cooker"
	print(label)
    
    #uncomment for simultanepus events
    '''#prediction - simultaneous events 
    pred = model.predict(X_test)
    if pred == 0:
		label = "B&K"
	elif pred == 1:
		label = "B&M"
	elif pred == 2:
		label = "B&R"
	elif pred == 3 :
		label = "Blender"
	elif pred == 4:
		label = "K&R"
	elif pred == 5:
		label = "kettle"
	elif pred == 6 :
		label = "M&K"
	elif pred == 7:
		label = "M&R"
	elif pred == 8:
		label = "Microwave"
	elif pred == 9 :
		label = "None"
	else:
		label = "Rice Cooker"

	print(label)'''

    
	t1.join()


