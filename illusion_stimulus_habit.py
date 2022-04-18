"""
Author:         Hyeyoung Shin (Berkeley University), Jerome Lecoq (Allen Institute)
"""
import camstim
from camstim import Stimulus, SweepStim, Foraging, Window, NaturalScenes
import argparse
import yaml
import os
import os
import numpy as np

def create_ICwcfg1_habit(DURFAC, shared_repository_location):
    path =  os.path.join(shared_repository_location, 'visICtxiwcfg1')

    Nblank_sweeps_start=60
    Nblank_sweeps_end=60

    stimulus = NaturalScenes(image_path_list=path,
                            window=window,
                            sweep_length=0.4,
                            blank_length=0,
                            blank_sweeps=0, #1,
                            runs=1,
                            shuffle=True,)

    image_path_list = stimulus.image_path_list
    highrepstim = ["110000", "110101", "110105", "110106", "110107", "110109", "110110", 
                "110111", "111105", "111109", "111201", "111299"]
    highrepstiminds = np.zeros(len(highrepstim))
    highcnt = 0
    lowrepstiminds = np.zeros(len(image_path_list)-len(highrepstim))
    lowcnt = 0
    for ii, imfn in enumerate(image_path_list):
        imid = imfn.split('\\')[-1].split('.')[0]
        if imid in highrepstim:
            highrepstiminds[highcnt] = ii
            highcnt += 1
        else:
            lowrepstiminds[lowcnt] = ii
            lowcnt += 1

    if not highcnt==len(highrepstim) and lowcnt==len(image_path_list)-len(highrepstim):
        raise Exception('check highrepstiminds and lowrepstiminds')

    # define and randomize the order of non-blank trials
    trialorder = np.concatenate((np.tile(highrepstiminds, 100*DURFAC), np.tile(lowrepstiminds, 10*DURFAC)))
    np.random.shuffle(trialorder)

    # replace the auto-generated sweep order with a custom one
    sweep_order_main = np.zeros(2*len(trialorder))
    sweep_order_main[range(0,len(sweep_order_main),2)]=trialorder
    sweep_order_main[range(1,len(sweep_order_main),2)]=-1

    # pad blank stimuli at the beginning and end
    sweepstart = -1*np.ones(Nblank_sweeps_start, dtype=sweep_order_main.dtype)
    sweepend = -1*np.ones(Nblank_sweeps_end, dtype=sweep_order_main.dtype)
    sweep_order = np.concatenate((sweepstart, sweep_order_main, sweepend))

    # replace blank stimulus with the first stimulus in path. (make sure this is four white circles!)
    sweep_order[sweep_order==-1]=0
    stimulus.sweep_order = sweep_order.reshape(-1).tolist()
    stimulus._build_frame_list()

    print('ICcfg1 stim')
    print(stimulus.sweep_order)
    
    return stimulus

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
    ICwcfg0_path = os.path.join(shared_repository_location, "ICwcfg0_habit.stim")      
    ICkcfg1_path = os.path.join(shared_repository_location, "ICkcfg1_habit.stim")
    ICkcfg0_path = os.path.join(shared_repository_location, "ICkcfg0_habit.stim")

    RFCI_path = os.path.join(shared_repository_location, "RFCI_habit.stim")
    sizeCI_path = os.path.join(shared_repository_location, "sizeCI_habit.stim")

    # load our stimuli
    ICwcfg1 = create_ICwcfg1_habit(DURFAC, shared_repository_location) 
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