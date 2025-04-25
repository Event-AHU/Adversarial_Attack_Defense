from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    settings.coesot_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/COESOT'
    settings.fe108_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/fe108'
    settings.network_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/event_attack/event_CEUTrack/pretrained_models'    # Where tracking networks are stored.
    settings.prj_dir = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/event_attack/event_CEUTrack'
    settings.result_plot_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/event_attack/event_CEUTrack/output/test/result_plots'
    settings.results_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/RGB_Event_Frame_Attack/event_CEUTrack/visual/ceosot/00/ours_new' #改这个即可   # Where to store tracking results
    settings.save_dir = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/RGB_Event_Frame_Attack/event_CEUTrack/output'
    settings.segmentation_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/event_attack/event_CEUTrack/output/test/segmentation_results'
    settings.visevent_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/VisEvent'

    return settings