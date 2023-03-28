from nuscenes_ import NuScenes, NuScenesExplorer

nusc = NuScenes(version='v1.0-mini', dataroot='/share/qi.chao/open_sor_data/nuscenes', verbose=True)
# nusc.list_lidarseg_categories(sort_by='count')
# print('--------------')
# nusc.list_panoptic_instances(sort_by='count')
my_sample = nusc.sample[5]
nusc.get_sample_lidarseg_stats(my_sample['token'], sort_by='count', gt_from='panoptic')

sample_data_token = my_sample['data']['LIDAR_TOP']
nusc.render_sample_data(sample_data_token,
                        with_anns=False,
                        out_path='/home/qi.chao/c7/mycode/nuscenes-devkit/work_dir/jpg/lidarseg_lidartop',
                        # filter_lidarseg_labels=[30],
                        # show_lidarseg=True,
                        show_panoptic=True,
                        show_lidarseg_legend=True)

nusc.render_pointcloud_in_image(my_sample['token'],
                                out_path='/home/qi.chao/c7/mycode/nuscenes-devkit/work_dir/jpg/lidartop2camfrontleft',
                                pointsensor_channel='LIDAR_TOP',
                                camera_channel='CAM_FRONT_LEFT',
                                render_intensity=False,
                                # show_lidarseg=True,
                                filter_lidarseg_labels=[30],
                                show_lidarseg_legend=True,
                                show_panoptic=True,
                                )

nusc.render_sample(my_sample['token'],
                   out_path='/home/qi.chao/c7/mycode/nuscenes-devkit/work_dir/jpg/rendersample',
                   show_lidarseg=True,
                   filter_lidarseg_labels=[30])

my_scene = nusc.scene[0]

# nusc.render_scene_channel_lidarseg(my_scene['token'],
#                                    'CAM_FRONT',
#                                    out_folder='/home/qi.chao/c7/mycode/nuscenes-devkit/work_dir/jpg/render_scene_channel_lidarseg',
#                                    render_mode='video',
#                                    filter_lidarseg_labels=[30],
#                                 #    verbose=True,
#                                 #    dpi=100,
#                                    imsize=(1280, 720))

# nusc.render_scene_lidarseg(my_scene['token'],
#                            filter_lidarseg_labels=[30],
#                            out_path='/home/qi.chao/c7/mycode/nuscenes-devkit/work_dir/jpg/render_scene_lidarseg.avi'
#                         #    verbose=True,
#                         #    dpi=100,
#                            )