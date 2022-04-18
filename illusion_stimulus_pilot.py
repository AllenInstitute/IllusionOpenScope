"""
Author:         Hyeyoung Shin (Berkeley University), Jerome Lecoq (Allen Institute)
"""
import camstim
from camstim import Stimulus, SweepStim
from camstim import Foraging
from camstim import Window
import argparse
import yaml
import os

if __name__ == "__main__":
    # This part load parameters from mtrain
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", nargs="?", type=str, default="")

    args, _ = parser.parse_known_args() # <- this ensures that we ignore other arguments that might be needed by camstim
    
    # print args
    with open(args.json_path, 'r') as f:
        # we use the yaml package here because the json package loads as unicode, which prevents using the keys as parameters later
        json_params = yaml.load(f)
    # end of mtrain part
    
    # mtrain should be providing : a path to a network folder or a local folder with the entire repo pulled
    shared_repository_location = json_params.get('shared_repository_location', r"C:\Users\Hyeyoung\Documents\OpenScope\illusion_camstim")
    
    # mtrain should be providing : Gamma1.Luminance50
    monitor_name = json_params.get('shared_repository_location', "illusionMonitor")
    
    # mtrain should be providing varying value depending on stage
    Nrep = int(json_params.get('Nrep', 2))

    print('Nrep: ' + str(Nrep))

    # Create display window
    window = Window(fullscr=True,
                    monitor=monitor_name,
                    screen=1, warp=None)

    # paths to our stimuli
    ICwcfg1_path = os.path.join(shared_repository_location, "ICwcfg1_1rep.stim")
    ICwcfg0_path = os.path.join(shared_repository_location, "ICwcfg0_1rep.stim")
    ICkcfg1_path = os.path.join(shared_repository_location, "ICkcfg1_1rep.stim") 
    ICkcfg0_path = os.path.join(shared_repository_location, "ICkcfg0_1rep.stim")

    RFCI_path = os.path.join(shared_repository_location, "RFCI_1rep.stim")
    sizeCI_path = os.path.join(shared_repository_location, "sizeCI_1rep.stim")

    # load our stimuli
    ICwcfg1 = Stimulus.from_file(ICwcfg1_path, window) 
    ICwcfg0 = Stimulus.from_file(ICwcfg0_path, window) 
    ICkcfg1 = Stimulus.from_file(ICkcfg1_path, window) 
    ICkcfg0 = Stimulus.from_file(ICkcfg0_path, window) 
    RFCI = Stimulus.from_file(RFCI_path, window) 
    sizeCI = Stimulus.from_file(sizeCI_path, window)

    # each tuple determines in seconds start and end of each block.
    ICwcfg1_ds = [(0, 30*Nrep)]
    ICwcfg0_ds = [(30*Nrep, 60*Nrep)]
    ICkcfg1_ds = [(60*Nrep, 90*Nrep)]
    ICkcfg0_ds = [(90*Nrep, 120*Nrep)]
    RFCI_ds = [(120*Nrep, 145*Nrep)]
    sizeCI_ds = [(145*Nrep, 220*Nrep)]

    ICwcfg1.set_display_sequence(ICwcfg1_ds)
    ICwcfg0.set_display_sequence(ICwcfg0_ds)
    ICkcfg1.set_display_sequence(ICkcfg1_ds)
    ICkcfg0.set_display_sequence(ICkcfg0_ds)
    RFCI.set_display_sequence(RFCI_ds)
    sizeCI.set_display_sequence(sizeCI_ds)

    # create SweepStim instance
    ss = SweepStim(window,
                stimuli= [ICwcfg1, ICwcfg0, ICkcfg1, ICkcfg0, RFCI, sizeCI],
                pre_blank_sec=0, #60,
                post_blank_sec=30, #60,
                params={},
                )
    
    # add in foraging so we can track wheel, potentially give rewards, etc
    f = Foraging(window       = window,
                                auto_update = False,
                                params      = {}
                                )
    
    ss.add_item(f, "foraging")

    # run it
    ss.run()