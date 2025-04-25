class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/AttackTracking/CEUTrack/output/checkpoints'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/AttackTracking/CEUTrack/output/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/chenqiang/AttackTracking/CEUTrack/pretrained_models'
        self.coesot_dir = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/COESOT/train'
        self.coesot_val_dir = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/COESOT/train'

        self.fe108_dir = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/fe108/train'
        self.fe108_val_dir = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/fe108/train'

        self.visevent_dir = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/VisEvent/train'
        self.visevent_val_dir = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/VisEvent/train'