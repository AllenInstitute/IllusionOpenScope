"""
modified from brain_observatory_stimulus.py
"""
from psychopy import visual, monitors
from camstim import Stimulus, SweepStim
from camstim import Foraging
from camstim import Window, Warp
# from zro.proxy import DeviceProxy
import time
import datetime
import numpy as np

# from .optotagging import optotagging
# from ecephys_stimulus_scripts.optotagging import optotagging
# from ecephys import optotagging


# Create display window
window = Window(fullscr=True,
                monitor="illusionMonitor",
                screen=1, warp=None)


# paths to our stimuli
ICwcfg1_path =       r"C:\Users\Hyeyoung\Documents\OpenScope\illusion_camstim\ICwcfg1.stim" #  minutes ( s)
ICwcfg0_path =       r"C:\Users\Hyeyoung\Documents\OpenScope\illusion_camstim\ICwcfg0.stim" #  minutes ( s)
ICkcfg1_path =       r"C:\Users\Hyeyoung\Documents\OpenScope\illusion_camstim\ICkcfg1.stim" #  minutes ( s)
ICkcfg0_path =       r"C:\Users\Hyeyoung\Documents\OpenScope\illusion_camstim\ICkcfg0.stim" #  minutes ( s)

RFCI_path =         r"C:\Users\Hyeyoung\Documents\OpenScope\illusion_camstim\RFCI.stim"
sizeCI_path =         r"C:\Users\Hyeyoung\Documents\OpenScope\illusion_camstim\sizeCI.stim"

# grating_path =         r"C:\Users\Hyeyoung\Documents\OpenScope\illusion_camstim\grating.stim"


# load our stimuli
ICwcfg1 = Stimulus.from_file(ICwcfg1_path, window) 
ICwcfg0 = Stimulus.from_file(ICwcfg0_path, window) 
ICkcfg1 = Stimulus.from_file(ICkcfg1_path, window) 
ICkcfg0 = Stimulus.from_file(ICkcfg0_path, window) 
RFCI = Stimulus.from_file(RFCI_path, window) 
sizeCI = Stimulus.from_file(sizeCI_path, window)

# grating = Stimulus.from_file(grating_path, window) 


# each tuple determines in seconds start and end of each block.
ICwcfg1_ds = [(0, 4800)]
part1s = ICwcfg1_ds[-1][-1] # end of part 1
ICwcfg0_ds = [(part1s, part1s+480)]
ICkcfg1_ds = [(part1s+480, part1s+960)]
ICkcfg0_ds = [(part1s+960, part1s+1440)]
part2s = ICkcfg0_ds[-1][-1] # end of part 2. 4800+1440=6240
RFCI_ds = [(part2s, part2s+210)]
sizeCI_ds = [(part2s+210, part2s+980)] # 6240+980=7220

ICwcfg1.set_display_sequence(ICwcfg1_ds)
ICwcfg0.set_display_sequence(ICwcfg0_ds)
ICkcfg1.set_display_sequence(ICkcfg1_ds)
ICkcfg0.set_display_sequence(ICkcfg0_ds)
RFCI.set_display_sequence(RFCI_ds)
sizeCI.set_display_sequence(sizeCI_ds)

# grating.set_display_sequence(grating_ds)

# kwargs
params = {
    'syncpulse': True,
    'syncpulseport': 1,
    'syncpulselines': [4, 7],  # frame, start/stop
    'trigger_delay_sec': 0.0,
    'bgcolor': (-1,-1,-1),
    'eyetracker': False,
    'eyetrackerip': "W7DT12722",
    'eyetrackerport': 1000,
    'syncsqr': True,
    'syncsqrloc': (0,0), 
    'syncsqrfreq': 60,
    'syncsqrsize': (100,100),
    'showmouse': True
}


# W7DT12722

# create SweepStim instance
ss = SweepStim(window,
               stimuli= [ICwcfg1, ICwcfg0, ICkcfg1, ICkcfg0, RFCI, sizeCI],
               pre_blank_sec=0,
               post_blank_sec=30,
               params=params,
               )

# # add in foraging so we can track wheel, potentially give rewards, etc
# f = Foraging(window=window,
#             auto_update=False,
#             params=params,
#             nidaq_tasks={'digital_input': ss.di,
#                          'digital_output': ss.do,})  #share di and do with SS
# ss.add_item(f, "foraging")

# run it
ss.run()

# optotagging.optotagging('404555',genotype='c57')
