import torch

REGISTERED_COLATE_CLASSES = {}

def register_collate_fn(cls, name=None):
    global REGISTERED_COLATE_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_COLATE_CLASSES, f"exist class: {REGISTERED_COLATE_CLASSES}"
    REGISTERED_COLATE_CLASSES[name] = cls
    return cls

def get_collate_class(name):
    global REGISTERED_COLATE_CLASSES
    assert name in REGISTERED_COLATE_CLASSES, f"available class: {REGISTERED_COLATE_CLASSES}"
    return REGISTERED_COLATE_CLASSES[name]

@register_collate_fn
def collate_fn_default(data):
    point_num = [d['point_num'] for d in data]
    batch_size = len(point_num)
    ref_labels = data[0]['ref_label']
    origin_len = data[0]['origin_len']
    ref_indices = [torch.from_numpy(d['ref_index']) for d in data]
    point2img_index = [d['point2img_index'] for d in data]
    # path = [d['root'] for d in data]

    img = [d['img'] for d in data]
    img_indices = [d['img_indices'] for d in data]
    img_label = [d['img_label'] for d in data]

    b_idx = []
    for i in range(batch_size):
        b_idx.append(torch.ones(point_num[i]) * i)
    points = [d['point_feat'] for d in data]
    labels = [d['point_label'] for d in data]

    ret_data = {
        'points': torch.cat(points).float(),
        'batch_idx': torch.cat(b_idx).long(),
        'batch_size': batch_size,
        'labels': torch.cat(labels).long().squeeze(1),
        'raw_labels': torch.from_numpy(ref_labels).long(),
        'origin_len': origin_len,
        'indices': torch.cat(ref_indices).long(),
        'point2img_index': point2img_index,
        'img': torch.stack(img, 0).permute(0, 3, 1, 2),
        'img_indices': img_indices,
        'img_label': torch.cat(img_label, 0).squeeze(1).long(),
    }
    if 'lidar_depths' in data[0].keys():
        ida_mats = [d['ida_mats'] for d in data]
        sensor2ego_mats = [d['sensor2ego_mats'] for d in data]
        bda_mat = [d['bda_mat'] for d in data]
        intrin_mats = [d['intrin_mats'] for d in data]
        lidar_depths = [d['lidar_depths'] for d in data]
        ret_data.update({
            'img': torch.stack(img, 0),
            'lidar_depths': torch.stack(lidar_depths),
            'meta': {
                'ida_mats': torch.stack(ida_mats),
                'sensor2ego_mats': torch.stack(sensor2ego_mats),
                'bda_mat': torch.stack(bda_mat),
                'intrin_mats': torch.stack(intrin_mats)
            }
        })

    return ret_data
