dataset_params = dict(
    resize = [400, 200],
    color_jitter = [0.4, 0.4, 0.4],
    image_normalizer = dict(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    ),
    trans_std = [0.1, 0.1, 0.1],
    max_volume_space = [50, 50, 3],
    min_volume_space = [-50, -50, -4],
    label_mapping = 'config/label_mapping/nuscenes.yaml',
    max_dropout_ratio = 0
)

train_pipeline = dict(
    type = 'NuscenseDateset',
    imageset = 'train',
    # data_path = '/share-global/xinli.xu/dataset/nuscenes',
    data_path = '/share/qi.chao/open_sor_data/nuscenes_mini',
    dataset_params = dataset_params,
    dataload_config = dict(
        debug = True,
        flip2d = 0.5,
        transform_aug = True,
        scale_aug = [0.95, 1.05],
        flip_aug = True,
        rotate_aug = True,
        dropout_aug = True,
        shuffle = True,
        batch_size = 1,
        num_workers = 4
    )
)

val_pipeline = dict(
    type = 'NuscenseDateset',
    imageset = 'val',
    # data_path = '/share-global/xinli.xu/dataset/nuscenes',
    data_path = '/share/qi.chao/open_sor_data/nuscenes_mini',
    dataset_params = dataset_params,
    dataload_config = dict(
        debug = True,
        flip2d = False,
        transform_aug = False,
        scale_aug = False,
        flip_aug = False,
        rotate_aug = False,
        dropout_aug = False,
        shuffle = False,
        batch_size = 1,
        num_workers = 4
    )
)

model = dict(
    type = '',
    img = dict (
        backbone = dict(
            type = ''
        ),
        neck = dict(
            type=''
        ),
    )
)