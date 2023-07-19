_base_ = [
    './base.py'
]
train_pipeline = dict(
    type = 'NuscenseDatesetDepth',
    dataload_config = dict(
        batch_size = 1,
        num_workers = 4,
        debug = True,
        transform_aug = True,
        scale_aug = [0.95, 1.05],
        flip_aug = True,
        rotate_aug = True,
        dropout_aug = True,
        shuffle = True,
        ida_aug_conf = dict(
            img_scale = [448, 800],
            # rot_lim = [-5.4, 5.4],
            rot_lim = [0, 0],
            rand_flip = True,
            img_mean = [123.675, 116.28, 103.53],
            img_std = [58.395, 57.12, 57.375]
        ),
        bda_aug_conf = dict(
            # rot_lim = [-22.5, 22.5],
            rot_lim = [0, 0],
            scale_lim = [0.95, 1.05],
            flip_dx_ratio = 0.5,
            flip_dy_ratio = 0.5
        )
    )
)

val_pipeline = dict(
    type = 'NuscenseDatesetDepth',
    dataload_config = dict(
        batch_size = 1,
        num_workers = 4,
        debug = True,
        transform_aug = False,
        scale_aug = False,
        flip_aug = False,
        rotate_aug = False,
        dropout_aug = False,
        shuffle = False,
        ida_aug_conf = dict(
            img_scale = [448, 800],
            # rot_lim = [-5.4, 5.4],
            rot_lim = [0, 0],
            rand_flip = False,
            img_mean = [123.675, 116.28, 103.53],
            img_std = [58.395, 57.12, 57.375]
        ),
        bda_aug_conf = dict(
            rot_lim = [0, 0],
            scale_lim = [1.0, 1.0],
            flip_dx_ratio = 0,
            flip_dy_ratio = 0
        )
    )
)

model = dict(
    img = dict(
        depth = dict(
            camera_depth_range = [2.0, 58.0, 0.5],
            pc_range = [-51.2, -51.2, -5, 51.2, 51.2, 3],
            downsample = 16,
            has_depth_net = True,
            depth_net_conf = dict(
                in_channels = 512,
                mid_channels = 512
            )
        )
    )
)
