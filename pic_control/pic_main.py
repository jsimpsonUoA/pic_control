'''
This is a temperature control algorithm for
the pressure vessel. It is designed to achieve
the desired temperature inside the pressure
vessel chamber using an iterative algorithm
that predicts the final temperature at the 
current settings and then adjusts the heating
input to achieve the desired setpoint.

Jonathan Simpson, PAL Lab UoA, 2021
'''

import time
import csv  
import warnings
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)
import os
import pandas as pd
import threading
import serial
import watlow

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit

class TemperatureControl():
    def __init__(self):

        self.ramp_trol_usb_port = '/dev/ttyUSB3'
        self.omega_usb_port = '/dev/ttyUSB1'
        self.seconds_between_reads = 10.0
        self.current_filename = 'test_currents.csv' #'/mnt/office_machine/home/jsim921/Dropbox (Uni of Auckland)/experiment-data/lab_stuff/ir_thermometer/scripts/current_temps.csv'
        
        self.mode = 'heat'                          # or 'cool'
        self.desired_temperature = 90.0
        self.initial_setpoint_guess = 92.9
        self.make_predictions = True                # True to run predictive algorithm, False to just read temps.
        self.prediction_stability_threshold = 0.04
        self.final_temp_tolerance = 0.2             # Within plus or minus this of desired temp
        self.min_time_before_predict_start = 900.0  # Seconds before starting prediction algorithm
        self.min_prediction_time = 600.0            # Minimum number of seconds of data to use for prediction.
        self.max_setpoint_deviation = 10.0          # Maximum deviation of setpoint from desired temp
        self.time_contant = 4000.0                  # Time constant of system for heating
        self.log_file = ''
        self.data_filename = ''
        self.predictions_filename = ''

        self.use_prerecorded_data = False
        self.prerecorded_data_name = 'pic_temp_data_2021-05-07_11:15.csv'
        self.data_directory = 'data/{}/'.format(datetime.now().strftime("%Y-%m-%d"))

    def mainloop(self):
        '''
        Main loop
        '''

        now = datetime.now()
        self.log_file = 'pic_log_{}.txt'.format(now.strftime("%Y-%m-%d_%H:%M"))
        self.data_filename = 'pic_temp_data_{}.csv'.format(now.strftime("%Y-%m-%d_%H:%M"))
        self.predictions_filename = 'pic_predictions_{}.csv'.format(now.strftime("%Y-%m-%d_%H:%M"))
        self._create_data_directory()

        try:
            current_setpoint = None
            last_three_minutes = []
            predict_start_time = None
            time_at_setpoint_change = None
            time_at_last_prediction = time.time() 
            time_at_last_read = time.time() 
            begun_predictions = False
            
            if self.use_prerecorded_data: 
                test_data = pd.read_csv(self.prerecorded_data_name).values
                time_at_last_prediction = test_data[0][0] 
                time_at_last_read = test_data[0][0] 
                current_setpoint = self.initial_setpoint_guess

            self.write_to_log(self.log_file, 'data_directory {}'.format(self.data_directory))
            self.write_to_log(self.log_file, 'using_prerecorded_data {}'.format(self.use_prerecorded_data))
            self.write_to_log(self.log_file, 'prerecorded_data_file {}'.format(self.prerecorded_data_name))
            self.write_to_log(self.log_file, 'data_filename {}'.format(self.data_filename))
            self.write_to_log(self.log_file, 'predictions_filename {}'.format(self.predictions_filename))
            self.write_to_log(self.log_file, 'mode {}'.format(self.mode))
            self.write_to_log(self.log_file, 'desired_temperature {}'.format(self.desired_temperature))
            self.write_to_log(self.log_file, 'initial_setpoint_guess {}'.format(self.initial_setpoint_guess))
            self.write_to_log(self.log_file, 'prediction_stability_threshold {}'.format(self.prediction_stability_threshold))
            self.write_to_log(self.log_file, 'final_temp_tolerance {}'.format(self.final_temp_tolerance))
            self.write_to_log(self.log_file, 'min_time_before_predict_start {}'.format(self.min_time_before_predict_start))
            self.write_to_log(self.log_file, 'min_prediction_time {}'.format(self.min_prediction_time))
            self.write_to_log(self.log_file, 'time_constant {}'.format(self.time_contant))

            # Read the initial temperatures for two minutes then change setpoint.
            self.write_to_log(self.log_file, 'reading_initial_temps')
            print('PIC: {}: Reading starting temperatures for two minutes'.format(datetime.now().strftime("%d_%H:%M:%S")))
            loop_start = time.time()
            loop_ind = 0 
            while (not self.use_prerecorded_data and (time.time()-loop_start) < 120.0) or (self.use_prerecorded_data and test_data[loop_ind][0]-test_data[0][0]<120.0):
                try:
                    if not self.use_prerecorded_data:
                        time.sleep(self.seconds_between_reads-(time.time()-time_at_last_read))
                        ramp_temp, ramp_setpoint = self.read_ramptrol()
                        oir_temp, ok_temp = self.read_omega()
                        time_at_last_read = time.time()
                    else:
                        ramp_temp, ramp_setpoint = test_data[loop_ind][3], current_setpoint
                        oir_temp, ok_temp = test_data[loop_ind][1], test_data[loop_ind][2]
                        time_at_last_read = test_data[loop_ind][0]
                        loop_ind+=1
                    self.write_to_file(self.data_filename, self.current_filename, time_at_last_read, oir_temp, ok_temp, ramp_temp, ramp_setpoint)
                except IOError:
                    time.sleep(1)
                    print('PIC: {}: Serial read error'.format(datetime.now().strftime("%d_%H:%M:%S")))
                    self.write_to_log(self.log_file, 'serial_read_error')
                    continue
            if abs(ok_temp-self.desired_temperature) > self.final_temp_tolerance:
                if abs(self.initial_setpoint_guess-self.desired_temperature) < self.max_setpoint_deviation:
                    initial_setpoint = self.initial_setpoint_guess
                else:
                    initial_setpoint = self.desired_temperature
                print('PIC: {0}: Setting initial setpoint to {1}C'.format(datetime.now().strftime("%d_%H:%M:%S"), round(initial_setpoint,1)))
                self.change_ramptrol_setpoint(round(initial_setpoint,1))
                setpoint_change = True
            else:
                print('PIC: {0}: Already at desired temperature.'.format(datetime.now().strftime("%d_%H:%M:%S")))
                print('PIC: {0}: Continuing to read temperatures.'.format(datetime.now().strftime("%d_%H:%M:%S")))
                self.write_to_log(self.log_file, 'already_at_desired_temp') 
                self.write_to_log(self.log_file, 'continuing_to_read_temps') 
                self.make_predictions, setpoint_change = False, False

            self.write_to_log(self.log_file, 'beginning_mainloop')
            print()
            print('PIC: {}: Beginning mainloop'.format(datetime.now().strftime("%d_%H:%M:%S")))
            loop_ind = 0

            while not self.use_prerecorded_data or loop_ind<len(test_data):
                # Read current temperatures
                try:
                    if not self.use_prerecorded_data:
                        time.sleep(max(0,self.seconds_between_reads-(time.time()-time_at_last_read)))
                        ramp_temp, ramp_setpoint = self.read_ramptrol()
                        oir_temp, ok_temp = self.read_omega()
                        time_at_last_read = time.time()
                    else:
                        ramp_temp, ramp_setpoint = test_data[loop_ind][3], current_setpoint
                        oir_temp, ok_temp = test_data[loop_ind][1], test_data[loop_ind][2]
                        time_at_last_read = test_data[loop_ind][0]
                        loop_ind+=1
                    self.write_to_file(self.data_filename, self.current_filename, time_at_last_read, oir_temp, ok_temp, ramp_temp, ramp_setpoint)
                    last_three_minutes.append((time_at_last_read,ramp_temp))
                except IOError:
                    time.sleep(1)
                    print('PIC: {}: Serial read error'.format(datetime.now().strftime("%d_%H:%M:%S")))
                    self.write_to_log(self.log_file, 'serial_read_error')
                    continue

                if not self.make_predictions:
                    continue

                # Check for setpoint change
                if current_setpoint == None or setpoint_change or abs(current_setpoint-ramp_setpoint) > 0.06:
                    setpoint_change = False
                    current_setpoint = ramp_setpoint
                    last_three_minutes = []
                    predict_start_time = None
                    begun_predictions = False
                    time_at_setpoint_change = time.time()
                    if self.use_prerecorded_data:
                        time_at_setpoint_change = time_at_last_read
                    print('PIC: {0}: Setpoint change to {1}C detected'.format(datetime.now().strftime("%d_%H:%M:%S"),round(current_setpoint,1)))
                    print('PIC: {0}: Waiting 3 minutes'.format(datetime.now().strftime("%d_%H:%M:%S")))
                    self.write_to_log(self.log_file, 'setpoint_change_detected {}'.format(round(current_setpoint,2)))
                    next_log = True
                    continue

                # Check if setpoint has been constant longer than 3 minutes
                calculate_av = False
                while len(last_three_minutes) > 0:
                    if time_at_last_read - last_three_minutes[0][0] > 180.0:
                        calculate_av = True
                        del last_three_minutes[0]
                    else:
                        break
                if calculate_av:
                    three_minute_av = np.average([tup[1] for tup in last_three_minutes])
                    if next_log:
                        print('PIC: {0}: Checking controller temp stability'.format(datetime.now().strftime("%d_%H:%M:%S")))
                        self.write_to_log(self.log_file, 'starting_controller_temp_stability_checking')
                        next_log = False
                else:
                    continue

                # Check stability of setpoint over last three minutes
                if predict_start_time == None:
                    if (time_at_last_read-time_at_setpoint_change) > self.min_time_before_predict_start:
                        if abs(three_minute_av - current_setpoint) < 0.5:
                            predict_start_time = time_at_last_read
                            print('PIC: {0}: Prediction start time set. Waiting {1} minutes.'.format(datetime.now().strftime("%d_%H:%M:%S"),round(self.min_prediction_time/60.0)))
                            self.write_to_log(self.log_file, 'prediction_start_time_set {}'.format(predict_start_time))
                        else:
                            continue
                    else:
                        continue

                # Check we've reached the minimum time needed for prediction
                if (time_at_last_read - predict_start_time) < self.min_prediction_time:
                    continue
                elif not begun_predictions:
                    begun_predictions = True
                    print('PIC: {0}: Beginning final temperature predictions.'.format(datetime.now().strftime("%d_%H:%M:%S")))
                    self.write_to_log(self.log_file, 'beginning_temp_predictions')                    

                # Make a final temeprature prediction
                if time_at_last_read - time_at_last_prediction > 60.0:
                    raw_times, ir_temp_arr, k_temp_arr, ramp_temp_arr, ramp_setpoint_arr = self.read_from_file(self.data_filename)
                    times = raw_times-raw_times[0]
                    params, errors, arrival_time, clock_time = self.predict_final_temp(predict_start_time-raw_times[0], times, raw_times, k_temp_arr, ramp_temp_arr)
                    if len(params) > 0:    
                        time_at_last_prediction = time.time()
                        if self.use_prerecorded_data:
                            time_at_last_prediction = time_at_last_read 
                        self._save_predictions(self.predictions_filename,predict_start_time,time_at_last_prediction,params, errors, arrival_time, clock_time)
                    else:
                        continue
                else:
                    continue

                # Check stability of temperature predictions
                temp_stable, final_temp = self.check_prediction_stability(self.predictions_filename)
                if temp_stable:
                    temp_delta = self.desired_temperature - final_temp
                    print('PIC: {0}: Temperature prediction stability achieved.'.format(datetime.now().strftime("%d_%H:%M:%S")))
                    print('PIC: {0}: Predicted final temperature: {1}C.'.format(datetime.now().strftime("%d_%H:%M:%S"),round(final_temp,1)))
                    self.write_to_log(self.log_file, 'temperature_prediction_stability_achieved {}'.format(round(final_temp,2)))     
                else:
                    continue    

                # Change the setpoint based on prediction
                if abs(temp_delta) > self.final_temp_tolerance:
                    if abs(round(current_setpoint + temp_delta,1)-self.desired_temperature) < self.max_setpoint_deviation:
                        print('PIC: {0}: Changing setpoint to {1}C.'.format(datetime.now().strftime("%d_%H:%M:%S"),round(current_setpoint + temp_delta,1)))
                        print()
                        #print(loop_ind,time_at_last_read-test_data[0][0],oir_temp,ok_temp,ramp_temp)
                        self.change_ramptrol_setpoint(round(current_setpoint + temp_delta,1))
                        setpoint_change = True
                        if self.use_prerecorded_data:
                            current_setpoint = round(current_setpoint + temp_delta,1)
                    else:
                        print('PIC: {0}: Deviation of predicted temp from desired temp too high.'.format(datetime.now().strftime("%d_%H:%M:%S")))
                        print('PIC: {0}: Not changing setpoint. Continuing to read temperatures.'.format(datetime.now().strftime("%d_%H:%M:%S")))
                        print()
                        self.write_to_log(self.log_file, 'temp_delta_exceeded_tolerance') 
                        self.write_to_log(self.log_file, 'ending_temp_predictions') 
                        self.make_predictions = False                        
                    #raise KeyboardInterrupt
                else:
                    print('PIC: {0}: Desired temperature will be reached without setpoint change.'.format(datetime.now().strftime("%d_%H:%M:%S")))
                    print('PIC: {0}: Continuing to read temperatures.'.format(datetime.now().strftime("%d_%H:%M:%S")))
                    print()
                    self.write_to_log(self.log_file, 'desired_temp_will_be_reached') 
                    self.write_to_log(self.log_file, 'ending_temp_predictions') 
                    self.make_predictions = False

        except KeyboardInterrupt:
            self.write_to_log(self.log_file, 'closed_by_operator')
        except Exception as e:
            self.write_to_log(self.log_file, 'exception_raised {}'.format(e))
            raise
        finally:
            print('PIC: {0}: Exiting.'.format(datetime.now().strftime("%d_%H:%M:%S")))
            self.write_to_log(self.log_file, 'exiting_script') 

    def mainloop_in_thread(self):
        '''Run the mainloop in a separate daemon thread'''
        thread = threading.Thread(target=self.mainloop, daemon=True)
        thread.start()
        return thread

    def write_to_file(self, filename, current_f, read_time, ir_temp=None, k_temp=None, ramp_temp=None, ramp_setpoint=None):
        '''
        Write the reocrded temperature data to a file
        '''

        filename = self.data_directory + filename
        if os.path.isfile(filename):
            with open(filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([read_time, ir_temp, k_temp, ramp_temp, ramp_setpoint])
        else:
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['TIME','IR_TEMP','K_TEMP','RAMP_TEMP','RAMP_SETPOINT'])
                writer.writerow([read_time, ir_temp, k_temp, ramp_temp, ramp_setpoint])       

        writer = csv.writer(open(current_f,'w'))
        writer.writerows([[ir_temp],[k_temp],[ramp_temp],[ramp_setpoint]])

    def read_from_file(self, filename):
        '''Read temeprature data from csv file'''

        filename = self.data_directory + filename
        with open(filename, 'r') as f:
            lines = csv.reader(f)
            raw_times, ir_temp, k_temp, ramp_temp, ramp_setpoint = [], [], [], [], []
            for line in lines:
                try:
                    raw_times.append(float(line[0]))
                    ir_temp.append(float(line[1]))
                    k_temp.append(float(line[2]))
                    ramp_temp.append(float(line[3]))
                    ramp_setpoint.append(float(line[4]))
                except ValueError:
                    pass # For headers

        # Remove unreasonable data
        i = 0
        while i < len(raw_times)-1:
            gradient = abs((k_temp[i+1]-k_temp[i])/(raw_times[i+1]-raw_times[i]))
            if gradient > 0.03:
                del raw_times[i+1]
                del ir_temp[i+1]
                del k_temp[i+1]
                del ramp_temp[i+1]
                del ramp_setpoint[i+1]
                continue
            else:
                i += 1

        raw_times, ir_temp, k_temp, ramp_temp, ramp_setpoint = np.array(raw_times), np.array(ir_temp), np.array(k_temp), np.array(ramp_temp), np.array(ramp_setpoint)
        return raw_times, ir_temp, k_temp, ramp_temp, ramp_setpoint

    def write_to_log(self, filename, data):
        '''Write data to the log file'''

        filename = self.data_directory + filename
        if not os.path.isfile(filename):
            with open(filename, 'w') as f:
               f.write(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+' '+data+'\n')
        else:
            with open(filename, 'a') as f:
               f.write(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+' '+data+'\n')

    def initialise_ramptrol(self):
        '''
        Initialise the RampTrol controller
        '''
        if not self.use_prerecorded_data:
            return watlow.TemperatureController(self.ramp_trol_usb_port)

    def initialise_omega(self):
        '''
        Initialise the Omega thermometer
        '''
        if not self.use_prerecorded_data:
            return serial.Serial(port=self.omega_usb_port,baudrate=19200,bytesize=8,stopbits = serial.STOPBITS_ONE, parity = serial.PARITY_NONE,timeout=5)

    def read_ramptrol(self):
        '''Read RampTrol temperatures'''
        tc = self.initialise_ramptrol()
        temps = tc.get()
        tc.close()
        return temps['actual'], temps['setpoint']

    def read_omega(self):
        '''Read the Omega temps'''
        om = self.initialise_omega()
        om.flushInput()
        data = om.read(11)
        om.close()
        if len(data) > 0:
            ir_high_byte = data[1]
            ir_low_byte = data[2]
            ir_temp = (ir_low_byte | (ir_high_byte << 8)) / 10.

            k_high_byte = data[3]
            k_low_byte = data[4]
            k_temp = (k_low_byte | (k_high_byte << 8)) / 10.
        else:
            raise IOError
            return None, None

        return ir_temp, k_temp

    def change_ramptrol_setpoint(self, new_temp):
        '''Change the setpoint to new_temp'''
        if not self.use_prerecorded_data:
            if 0.0 < new_temp < 170.1:
                if self.log_file != '':
                    self.write_to_log(self.log_file, 'changing_setpoint {}'.format(new_temp)) 
                tc = self.initialise_ramptrol()
                tc.set(new_temp)
                tc.close()

    def predict_final_temp(self, start_time, times, raw_times, k_temp, ramp_temp, end_time=None):
        '''Make a prediction of the final temperature by fitting Newton's law to the data
        The first element in the returned params list is the predicted final temeprature'''

        if end_time == None:
            end_time = times[-1]
        predict_to = 60000.0 # Arbitrary seconds
        start_ind = np.argmin(np.abs(times-start_time))
        end_ind = np.argmin(np.abs(times-end_time))
        print(start_ind,end_ind)

        try:
            params, cov = curve_fit(self._newtons_law,times[start_ind:end_ind],k_temp[start_ind:end_ind],p0=(ramp_temp[start_ind],k_temp[start_ind],self.time_contant),maxfev=2000)
        except RuntimeError:
            return [], None, None, None
        predict_times = np.arange(start_time,times[-1]+predict_to,10.0)
        errors = np.sqrt(np.diag(cov))
        arrival_index = np.argmin(np.abs(self._newtons_law(predict_times,*params)-params[0]+0.05))
        arrival_time = predict_times[arrival_index]
        clock_time = datetime.utcfromtimestamp(int(arrival_time+raw_times[0]))

        return params, errors, arrival_time, clock_time

    def check_prediction_stability(self, predictions_filename):
        '''Check if the predictions of temp are stable over a certian period'''

        df = pd.read_csv(self.data_directory + predictions_filename)
        data = df.to_numpy()

        start_times = data[:,0]
        end_times = data[:,1]
        start_ind = np.argmin(np.abs(end_times-(end_times[-1]-300.0)))   #Use five minutes of predictions
        five_minute_temps = data[:,2][start_ind:]
        five_minute_temps_errs = data[:,3][start_ind:]

        for element in five_minute_temps_errs:
            if element > 10.0:
                return False, None

        if len(five_minute_temps) > 4:
            if np.std(five_minute_temps) < self.prediction_stability_threshold:   # IMPORTANT CONDITION
                return True, np.mean(five_minute_temps)
            
        return False, None

    def plot_data(self, plot_predictions=False):
        '''Plot the temperature data'''

        try:
            raw_times, ir_temp_arr, k_temp_arr, ramp_temp_arr, ramp_setpoint_arr = self.read_from_file(self.data_filename)
            times = (raw_times-raw_times[0]) / 60.0
            plt.plot(times, k_temp_arr, label='Thermocouple')
            plt.plot(times, ramp_temp_arr, label='Controller')

            if plot_predictions and os.path.exists(self.data_directory + self.predictions_filename):
                df = pd.read_csv(self.data_directory + self.predictions_filename)
                data = df.to_numpy()[-1]
                pred_start, temp_env, temp_0, tau, arrival_time = data[0], data[2], data[4], data[6], data[8]
                if 0.0 < temp_env < 160.0:
                    predict_plot_times = np.arange(pred_start-raw_times[0],arrival_time,0.1)
                    plt.plot(predict_plot_times/60.0,self._newtons_law(predict_plot_times,temp_env,temp_0,tau),zorder=0)
                    plt.plot(arrival_time/60.0,params[0]-0.05,'rx')

            plt.xlabel('Time (minutes)'), plt.ylabel('Temperature (C)')
            plt.legend()
            plt.show()
        except:
            print('PIC: Plotting error')

    def _create_data_directory(self):
        '''Create a directory to save teh data in'''

        if not os.path.exists(self.data_directory):
            os.mkdir(self.data_directory)

    def _newtons_law(self,t,temp_env,temp_0,tau):
        '''Implementation of Newton's Law of cooling/heating'''
        return temp_env+(temp_0-temp_env)*np.exp(-t/tau)

    def _save_predictions(self,filename,start_time,end_time,params,errors,arrival_time,clock_time):
        '''Save the temperature predictions'''
        
        filename = self.data_directory + filename
        clock_time = clock_time.strftime('%Y-%m-%d_%H:%M:%S')
        line = [start_time,end_time,params[0],errors[0],params[1],errors[1],params[2],errors[2],arrival_time,clock_time]
        if os.path.isfile(filename):
            with open(filename,'a') as f:
                writer = csv.writer(f)
                writer.writerow(line)
        else:
            with open(filename,'w') as f:
                writer = csv.writer(f)
                writer.writerow(['PREDICTION_START_TIME','PREDICTION_END_TIME','PREDICTED_FINAL_TEMP','PREDICTED_FINAL_TEMP_ERR'
                                                  ,'T0_FROM_FIT','T0_FROM_FIT_ERR'
                                                  ,'TAU_FROM_FIT','TAU_FROM_FIT_ERR'
                                                  ,'PREDICTED_TIME_FINAL_T_RELATIVE'
                                                  ,'PREDICTED_TIME_FINAL_T_CLOCK'])
                writer.writerow(line)


if __name__ == "__main__":
    controller = TemperatureControl()
    controller.mainloop()
