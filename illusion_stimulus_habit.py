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
    monitor_name = json_params.get('monitor_name', "illusionMonitor")
    
    # mtrain should be providing varying value depending on stage
    DURFAC = int(json_params.get('DURFAC', 1))

    print('DURFAC: ' + str(DURFAC))

    # Create display window
    window = Window(fullscr=True,
                    monitor=monitor_name,
                    screen=1, warp=None)

    # paths to our stimuli
    ICwcfg1_path = os.path.join(shared_repository_location, "ICwcfg1_habit.stim")
    ICwcfg0_path = os.path.join(shared_repository_location, "ICwcfg0_habit.stim")      
    ICkcfg1_path = os.path.join(shared_repository_location, "ICkcfg1_habit.stim")
    ICkcfg0_path = os.path.join(shared_repository_location, "ICkcfg0_habit.stim")

    RFCI_path = os.path.join(shared_repository_location, "RFCI_habit.stim")
    sizeCI_path = os.path.join(shared_repository_location, "sizeCI_habit.stim")

    # load our stimuli
    ICwcfg1 = Stimulus.from_file(ICwcfg1_path, window) 
    ICwcfg0 = Stimulus.from_file(ICwcfg0_path, window) 
    ICkcfg1 = Stimulus.from_file(ICkcfg1_path, window) 
    ICkcfg0 = Stimulus.from_file(ICkcfg0_path, window) 
    RFCI = Stimulus.from_file(RFCI_path, window) 
    sizeCI = Stimulus.from_file(sizeCI_path, window)

    # each tuple determines in seconds start and end of each block.
    ICwcfg1_ds = [(0, 1200*DURFAC)]
    part1s = ICwcfg1_ds[-1][-1] # end of part 1
    ICwcfg0_ds = [(part1s, part1s+120*DURFAC)]
    ICkcfg1_ds = [(part1s+120*DURFAC, part1s+240*DURFAC)]
    ICkcfg0_ds = [(part1s+240*DURFAC, part1s+360*DURFAC)]
    part2s = ICkcfg0_ds[-1][-1] # end of part 1
    RFCI_ds = [(part2s, part2s+60*DURFAC)]
    sizeCI_ds = [(part2s+60*DURFAC, part2s+240*DURFAC)]

    ICwcfg1.set_display_sequence(ICwcfg1_ds)
    ICwcfg0.set_display_sequence(ICwcfg0_ds)
    ICkcfg1.set_display_sequence(ICkcfg1_ds)
    ICkcfg0.set_display_sequence(ICkcfg0_ds)
    RFCI.set_display_sequence(RFCI_ds)
    sizeCI.set_display_sequence(sizeCI_ds)

    # create SweepStim instance
    ss = SweepStim(window,
                stimuli= [ICwcfg1, ICwcfg0, ICkcfg1, ICkcfg0, RFCI, sizeCI],
                pre_blank_sec=0,
                post_blank_sec=30,
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