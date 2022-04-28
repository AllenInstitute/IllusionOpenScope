"""
Author:         Hyeyoung Shin (Berkeley University), Jerome Lecoq (Allen Institute)
"""
import camstim
from camstim import Stimulus, SweepStim, Foraging, Window, NaturalScenes
import argparse
import yaml
import os
import numpy as np
import glob
from psychopy import visual

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

    print('ICwcfg1 stim')
    print(stimulus.sweep_order)
    
    return stimulus

def create_ICwcfg0_habit(DURFAC, shared_repository_location):
    path =  os.path.join(shared_repository_location, 'visICtxiwcfg0')

    Nblank_sweeps_start=30,
    Nblank_sweeps_end=30,

    stimulus = NaturalScenes(image_path_list=path,
                            window=window,
                            sweep_length=0.4,
                            blank_length=0,
                            blank_sweeps=0, #1,
                            runs=1,
                            shuffle=False,)

    image_path_list = stimulus.image_path_list
    trialorder = np.tile(range(len(image_path_list)), 4*DURFAC)
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

    print('ICwcfg0 stim')
    print(stimulus.sweep_order) 
    
    return stimulus

def create_ICkcfg1_habit(DURFAC, shared_repository_location):
    path =  os.path.join(shared_repository_location, 'visICtxikcfg1')

    Nblank_sweeps_start=30,
    Nblank_sweeps_end=30,

    stimulus = NaturalScenes(image_path_list=path,
                            window=window,
                            sweep_length=0.4,
                            blank_length=0,
                            blank_sweeps=0, #1,
                            runs=1,
                            shuffle=False,)

    image_path_list = stimulus.image_path_list
    trialorder = np.tile(range(len(image_path_list)), 4*DURFAC)
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

    print('ICkcfg1 stim')
    print(stimulus.sweep_order)
        
    return stimulus

def create_ICkcfg0_habit(DURFAC, shared_repository_location):
    path =  os.path.join(shared_repository_location, 'visICtxikcfg0')

    Nblank_sweeps_start=30,
    Nblank_sweeps_end=30,

    stimulus = NaturalScenes(image_path_list=path,
                            window=window,
                            sweep_length=0.4,
                            blank_length=0,
                            blank_sweeps=0, #1,
                            runs=1,
                            shuffle=False,)

    image_path_list = stimulus.image_path_list
    trialorder = np.tile(range(len(image_path_list)), 4*DURFAC)
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

    print('ICkcfg0 stim')
    print(stimulus.sweep_order)

    return stimulus

def create_RFCI_habit(DURFAC, shared_repository_location):
    # note, in visual.GratingStim, when using a image file as a mask, the pixel depth must be 3 (e.g., RGB image)
    # uint8 numbers, "white" part of the image (255,255,255) are fully transparent
    # note, tif image mask gets rescaled to "size" parameter in Stimulus(visual.GratingStim())
    # i.e., must have the same aspect ratio
    # be careful with tif file size (e.g., 4096X4096 pixel tifs can lead to a lag ~+50%)

    tiffn0 =  os.path.join(shared_repository_location, 'visRFposmask', '01000.tif' )
    tiffn1 =  os.path.join(shared_repository_location, 'visRFposmask', '11000.tif' )

    rfpos = [(0,0), (0,-203.3786), (203.3786/2**0.5,-203.3786/2**0.5), (203.3786,0), \
                (203.3786/2**0.5,203.3786/2**0.5), (0,203.3786), (-203.3786/2**0.5,203.3786/2**0.5), \
                (-203.3786,0), (-203.3786/2**0.5,-203.3786/2**0.5)]
    masklist = [tiffn0, tiffn1]
    orivec = range(0, 180, 45) # [0]

    stimulus = Stimulus(visual.GratingStim(window,
                            # pos=(0, 0), #, (10, 0)],
                            units='pix',
                            size= (2560,2560), #(1920,1200),
                            #mask= tiffn, #alphamask, #"circle", #"None"
                            tex='sqr',
                            texRes=256,
                            #ori = 135,
                            sf=0.04/12.7112, # cycles/units. if units='pix', cycles/pix
                            contrast=1,
                            ),
                    # pos 0, TF 1, SF 2, Ori 3, contrast 4
                    # regardless of column index, when shuffle=False, 
                    # iterates first through ori, then through masks, then through positions
                    sweep_params={
                        'Pos': (rfpos,0), #(+right,+up))
                        'TF': ([2.0], 1),
                        'Mask': (masklist, 4),
                        'Ori': (orivec, 3),
                        },
                    sweep_length=0.25,
                    start_time=0.0,
                    blank_length=0.0,
                    blank_sweeps=0, # len(orivec), when 2, 1 out of 3 stimuli are blank (if shuffle=False, stim stim blank)
                    runs=1, # when sweep_order is custom designated, runs is ignored
                    shuffle=False,
                    save_sweep_table=True,
                    )
    # replace the auto-generated sweep order with a custom one
    # blank is -1, first iterates through orientation then masks then positions 
    # (iori+len(orivec)*imask+len(orivec)*len(masklist)*ipos)
    # randomize position and mask, but not orientation
    sweep_order = np.asarray(range(len(orivec)*len(masklist)*len(rfpos))).reshape(len(masklist)*len(rfpos),len(orivec))
    sweep_order = np.tile(sweep_order, (2*DURFAC,1))
    np.random.shuffle(sweep_order)
    stimulus.sweep_order = sweep_order.reshape(-1).tolist()

    # rebuild the frame list (I may make this automatic in the future)
    stimulus._build_frame_list()

    print('RFCI stim')
    print(stimulus.sweep_order)

    return stimulus

def create_sizeCI_habit(DURFAC, shared_repository_location):
    # note, in visual.GratingStim, when using a image file as a mask, the pixel depth must be 3 (e.g., RGB image)
    # uint8 numbers, "white" part of the image (255,255,255) are fully transparent
    # note, tif image mask gets rescaled to "size" parameter in Stimulus(visual.GratingStim())
    # i.e., must have the same aspect ratio
    # be careful with tif file size (e.g., 4096X4096 pixel tifs can lead to a lag ~+50%)
    tifdir =  os.path.join(shared_repository_location, 'vissizemask//' )
    masklist = glob.glob(tifdir + '*.tif')

    rfpos = [(0,0)]
    orivec = range(0, 360, 45) # [0]

    stimulus = Stimulus(visual.GratingStim(window,
                            pos=(0, 0), 
                            units='pix',
                            size= (2560,2560), #(4096,4096), #(1920,1200),
                            #mask= tiffn, #alphamask, #"circle", #"None"
                            tex='sqr',
                            texRes=256,
                            #ori = 135,
                            sf=0.04/12.7112, # cycles/units. if units='pix', cycles/pix
                            contrast=1,
                            ),
                    # pos 0, TF 1, SF 2, Ori 3, contrast 4
                    # regardless of column index, when shuffle=False, 
                    # iterates first through ori, then through masks, then through positions
                    sweep_params={
                        'TF': ([2.0], 1),
                        'Mask': (masklist, 4),
                        'Ori': (orivec, 3),
                        },
                    sweep_length=0.25,
                    start_time=0.0,
                    blank_length=0.5,
                    blank_sweeps=0, # when 2, 1 out of 3 stimuli are blank (if shuffle=False, stim stim blank)
                    runs=2*DURFAC, # when sweep_order is custom designated, runs is ignored
                    shuffle=False,
                    save_sweep_table=True,
                    )

    # # replace the auto-generated sweep order with a custom one
    ## randomize the order of non-blank trials
    sweep_order = np.asarray(stimulus.sweep_order)
    nonblanktrialinds = np.where(np.not_equal(sweep_order, -1))
    nonblanktrials = sweep_order[nonblanktrialinds[0]]
    np.random.shuffle(nonblanktrials)
    sweep_order[nonblanktrialinds[0]] = nonblanktrials
    stimulus.sweep_order = sweep_order.reshape(-1).tolist()


    # rebuild the frame list (I may make this automatic in the future)
    stimulus._build_frame_list()

    print('sizeCI stim')
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
                    screen=0, warp=None)
    
    # load our stimuli
    ICwcfg1 = create_ICwcfg1_habit(DURFAC, shared_repository_location) 
    ICwcfg0 = create_ICwcfg0_habit(DURFAC, shared_repository_location) 
    ICkcfg1 = create_ICkcfg1_habit(DURFAC, shared_repository_location) 
    ICkcfg0 = create_ICkcfg0_habit(DURFAC, shared_repository_location) 
    RFCI = create_RFCI_habit(DURFAC, shared_repository_location) 
    sizeCI = create_sizeCI_habit(DURFAC, shared_repository_location)

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
                params= {'sync_sqr_loc' : (868, 528)}
                )

    # add in foraging so we can track wheel, potentially give rewards, etc
    f = Foraging(window       = window,
                                auto_update = False,
                                params      = {'sync_sqr_loc' : (868, 528)}
                                )
    
    ss.add_item(f, "foraging")

    # run it
    ss.run()