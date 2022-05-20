"""

Title:          GlobalLocalOddballs (GLO) stimulus generation code
Author:         Jacob A. Westerberg (Vanderbilt University)
Contact:        jacob.a.westerberg@vanderbilt.edu
Git Repo:       openscope-glo-stim (westerberg-science)
Written:        2022-03-24
Updated:        2022-04-12 (Jerome Lecoq)

"""
import camstim
from psychopy import visual
from camstim import SweepStim, Stimulus, Foraging
from camstim import Window, Warp
import random
import numpy as np
import os

"""
runs optotagging code for ecephys pipeline experiments
by joshs@alleninstitute.org, corbettb@alleninstitute.org, chrism@alleninstitute.org, jeromel@alleninstitute.org

(c) 2018 Allen Institute for Brain Science
"""
try:
    import Tkinter as tk
    import tkMessageBox as messagebox  # py2
except ImportError:
    import tkinter as tk
    from tkinter import messagebox
    
from toolbox.IO.nidaq import AnalogOutput
from toolbox.IO.nidaq import DigitalOutput
import datetime
import time
import pickle as pkl
import argparse
import yaml
from copy import deepcopy
from camstim.misc import get_config
from camstim.zro import agent

def run_optotagging(levels, conditions, waveforms, isis, sampleRate = 10000.):
    
    from toolbox.IO.nidaq import AnalogOutput
    from toolbox.IO.nidaq import DigitalOutput

    sweep_on = np.array([0,0,1,0,0,0,0,0], dtype=np.uint8)
    stim_on = np.array([0,0,1,1,0,0,0,0], dtype=np.uint8)
    stim_off = np.array([0,0,1,0,0,0,0,0], dtype=np.uint8)
    sweep_off = np.array([0,0,0,0,0,0,0,0], dtype=np.uint8)

    ao = AnalogOutput('Dev1', channels=[1])
    ao.cfg_sample_clock(sampleRate)
    
    do = DigitalOutput('Dev1', 2)
    
    do.start()
    ao.start()
    
    do.write(sweep_on)
    time.sleep(5)
    
    for i, level in enumerate(levels):
        
        print(level)
    
        data = waveforms[conditions[i]]
    
        do.write(stim_on)
        ao.write(data * level)
        do.write(stim_off)
        time.sleep(isis[i])
        
    do.write(sweep_off)
    do.clear()
    ao.clear()
 
def generatePulseTrain(pulseWidth, pulseInterval, numRepeats, riseTime, sampleRate = 10000.):
    
    data = np.zeros((int(sampleRate),), dtype=np.float64)    
    
    rise_and_fall = (((1 - np.cos(np.arange(sampleRate*riseTime/1000., dtype=np.float64)*2*np.pi/10))+1)-1)/2
    # half_length = rise_and_fall.size / 2
    half_length = int(rise_and_fall.size / 2)
    rise = rise_and_fall[:half_length]
    fall = rise_and_fall[half_length:]
    
    peak_samples = int(sampleRate*(pulseWidth-riseTime*2)/1000)
    peak = np.ones((peak_samples,))
    
    pulse = np.concatenate((rise, \
                           peak, \
                           fall))
    
    interval = int(pulseInterval*sampleRate/1000.)
    
    for i in range(0, numRepeats):
        data[i*interval:i*interval+pulse.size] = pulse
        
    return data

# HS 220516: CHECK THAT LEVELS (LED POWERS) HERE ARE THE SAME SCALE AS THE ALLEN BRAIN OBSERVATORY
#def optotagging(mouse_id, operation_mode='experiment', level_list = [1.15, 1.28, 1.345], output_dir = 'C:/ProgramData/camstim/output/'):
def optotagging(mouse_id, operation_mode='experiment', level_list = [1.4], output_dir = 'C:/ProgramData/camstim/output/'):

    sampleRate = 10000

    # 1 s cosine ramp:
    data_cosine = (((1 - np.cos(np.arange(sampleRate, dtype=np.float64)
                                * 2*np.pi/sampleRate)) + 1) - 1)/2  # create raised cosine waveform

    # 1 ms cosine ramp:
    rise_and_fall = (
        ((1 - np.cos(np.arange(sampleRate*0.001, dtype=np.float64)*2*np.pi/10))+1)-1)/2
    half_length = rise_and_fall.size / 2

    # pulses with cosine ramp:
    pulse_2ms = np.concatenate((rise_and_fall[:half_length], np.ones(
        (int(sampleRate*0.001),)), rise_and_fall[half_length:]))
    pulse_5ms = np.concatenate((rise_and_fall[:half_length], np.ones(
        (int(sampleRate*0.004),)), rise_and_fall[half_length:]))
    pulse_10ms = np.concatenate((rise_and_fall[:half_length], np.ones(
        (int(sampleRate*0.009),)), rise_and_fall[half_length:]))

    data_2ms_10Hz = np.zeros((sampleRate,), dtype=np.float64)

    for i in range(0, 10):
        interval = sampleRate / 10
        data_2ms_10Hz[i*interval:i*interval+pulse_2ms.size] = pulse_2ms

    data_5ms = np.zeros((sampleRate,), dtype=np.float64)
    data_5ms[:pulse_5ms.size] = pulse_5ms

    data_10ms = np.zeros((sampleRate,), dtype=np.float64)
    data_10ms[:pulse_10ms.size] = pulse_10ms

    data_10s = np.zeros((sampleRate*10,), dtype=np.float64)
    data_10s[:-2] = 1

    ##### HS 220516: THESE STIMULI ADDED FOR OPENSCOPE ILLUSION PROJECT #####
    #generatePulseTrain(pulseWidth, pulseInterval, numRepeats, riseTime, sampleRate = 10000.)
    Trise = .2 # in ms
    Tpulse = 2.4
    data_1Hz = generatePulseTrain(Tpulse, 1000, 1, Trise) # just a single pulse of 2 ms
    data_1Hz_10ms = generatePulseTrain(10, 1000, 1, Trise) # just a single pulse of 10 ms
    data_5Hz = generatePulseTrain(Tpulse, 200, 5, Trise) # 1 second of 5Hz pulse train. Each pulse is 2 ms wide
    data_10Hz = generatePulseTrain(Tpulse, 100, 10, Trise)
    data_20Hz = generatePulseTrain(Tpulse, 50, 20, Trise)
    data_30Hz = generatePulseTrain(Tpulse, 33.3, 30, Trise)
    data_40Hz = generatePulseTrain(Tpulse, 25, 40, Trise)  # 1 second of 40 Hz pulse train. Each pulse is 2 ms wide
    data_50Hz = generatePulseTrain(Tpulse, 20, 50, Trise)
    data_60Hz = generatePulseTrain(Tpulse, 16.7, 60, Trise)
    data_80Hz = generatePulseTrain(Tpulse, 12.5, 80, Trise)
    #data_100Hz = generatePulseTrain(Tpulse, 10, 100, Trise) # 1 second of 100 Hz pulse train. Each pulse is 2 ms wide
    data_square1s = generatePulseTrain(1000, 1000, 1, Trise) # 1 second square pulse: continuously on for 1s
    #########################################################
    
    # for experiment

    isi = 2.0
    isi_rand = 1.0
    numRepeats = 50

    # condition_list = [2, 3]
    # waveforms = [data_2ms_10Hz, data_5ms, data_10ms, data_cosine]
    condition_list = [0,1,2,3,4,5,6,7,8,9,10,11]
    waveforms = [data_1Hz_10ms, data_1Hz, data_5Hz, data_10Hz, data_20Hz, data_30Hz, data_40Hz, data_50Hz, data_60Hz, data_80Hz, data_square1s, data_cosine]
    
    opto_levels = np.array(level_list*numRepeats*len(condition_list)) #     BLUE
    opto_conditions = condition_list*numRepeats*len(level_list)
    opto_conditions = np.sort(opto_conditions)
    opto_isis = np.random.random(opto_levels.shape) * isi_rand + isi
    
    p = np.random.permutation(len(opto_levels))
    
    # implement shuffle?
    opto_levels = opto_levels[p]
    opto_conditions = opto_conditions[p]
    
    # for testing
    
    if operation_mode=='test_levels':
        isi = 2.0
        isi_rand = 0.0

        numRepeats = 2

        # condition_list = [0]
        # waveforms = [data_10s, data_10s]
        condition_list = [0, 1]
        waveforms = [data_1Hz, data_square1s]
        
        opto_levels = np.array(level_list*numRepeats*len(condition_list)) #     BLUE
        opto_conditions = condition_list*numRepeats*len(level_list)
        opto_conditions = np.sort(opto_conditions)
        opto_isis = np.random.random(opto_levels.shape) * isi_rand + isi

    elif operation_mode=='pretest':
        numRepeats = 1
        
        # condition_list = [0]
        # data_2s = data_10s[-sampleRate*2:]
        # waveforms = [data_2s]
        condition_list = [0, 1]
        waveforms = [data_1Hz, data_square1s]
        
        opto_levels = np.array(level_list*numRepeats*len(condition_list)) #     BLUE
        opto_conditions = condition_list*numRepeats*len(level_list)
        opto_conditions = np.sort(opto_conditions)
        opto_isis = [1]*len(opto_conditions)
    # 

    outputDirectory = output_dir
    fileDate = str(datetime.datetime.now()).replace(':', '').replace(
        '.', '').replace('-', '').replace(' ', '')[2:14]
    fileName = os.path.join(outputDirectory, fileDate + '_'+mouse_id + '.opto.pkl')

    print('saving info to: ' + fileName)
    fl = open(fileName, 'wb')
    output = {}

    output['opto_levels'] = opto_levels
    output['opto_conditions'] = opto_conditions
    output['opto_ISIs'] = opto_isis
    output['opto_waveforms'] = waveforms

    pkl.dump(output, fl)
    fl.close()
    print('saved.')

    # 
    run_optotagging(opto_levels, opto_conditions,
                    waveforms, opto_isis, float(sampleRate))
""" 
end of optotagging section
"""

if __name__ == "__main__":
    
    # This part load parameters from mtrain
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", nargs="?", type=str, default="")

    args, _ = parser.parse_known_args() # <- this ensures that we ignore other arguments that might be needed by camstim
    # print args
    with open(args.json_path, 'r') as f:
        # we use the yaml package here because the json package loads as unicode, which prevents using the keys as parameters later
        json_params = yaml.load(f)
    
    opto_disabled = json_params.get('disable_opto', True)
    if not(opto_disabled):
        
        opto_params = deepcopy(json_params.get("opto_params"))
        opto_params["mouse_id"] = json_params["mouse_id"]
        opto_params["output_dir"] = agent.OUTPUT_DIR
        #Read opto levels from stim.cfg file
        config_path = agent.CAMSTIM_CONFIG_PATH
        stim_cfg_opto_params = get_config(
            'Optogenetics',
            path=config_path,
        )
        
        # We override the level list here per Hyeyoung specifications
        opto_params["level_list"] = [1.4]
        
        optotagging(**opto_params)
        