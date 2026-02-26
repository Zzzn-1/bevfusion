import os
import pickle
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

def points_cam2img(points_3d, proj_mat):
    """极简的坐标转换函数，仅用于生成info"""
    points_num = len(points_3d)
    points_4d = np.concatenate([points_3d, np.ones((points_num, 1))], axis=-1)
    points_2d = points_4d @ proj_mat.T
    points_2d[:, 2] = np.clip(points_2d[:, 2], a_min=1e-5, a_max=1e5)
    points_2d[:, 0] /= points_2d[:, 2]
    points_2d[:, 1] /= points_2d[:, 2]
    return points_2d[:, :2]

def create_nuscenes_mini_info(root_path, out_path):
    # 初始化nuScenes mini版
    nusc = NuScenes(version='v1.0-mini', dataroot=root_path, verbose=True)
    
    # 仅处理mini版场景（不分train/val，统一为mini）
    mini_scenes = create_splits_scenes('mini')
    info = []
    
    for scene_idx, scene in enumerate(nusc.scene):
        if scene['name'] not in mini_scenes:
            continue
        
        # 遍历场景中的样本
        sample_token = scene['first_sample_token']
        while sample_token:
            sample = nusc.get('sample', sample_token)
            sample_info = {
                'token': sample['token'],
                'scene_token': scene['token'],
                'timestamp': sample['timestamp'],
                'sensor2ego': {},
                'ego2global': {},
                'cam_intrinsic': {},
                'images': {},
                'lidar_points': None,
            }
            
            # 处理LiDAR（仅保留TOP LiDAR）
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = nusc.get('sample_data', lidar_token)
            sample_info['lidar_points'] = {
                'token': lidar_token,
                'filename': os.path.join(root_path, lidar_data['filename']),
            }
            
            # 处理相机（6个相机）
            cam_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                         'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
            for cam_name in cam_names:
                cam_token = sample['data'][cam_name]
                cam_data = nusc.get('sample_data', cam_token)
                calib_data = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
                
                # 保存坐标转换矩阵和相机内参
                sample_info['sensor2ego'][cam_name] = calib_data['rotation'] + calib_data['translation']
                sample_info['ego2global'][cam_name] = ego_pose['rotation'] + ego_pose['translation']
                sample_info['cam_intrinsic'][cam_name] = calib_data['camera_intrinsic']
                sample_info['images'][cam_name] = {
                    'token': cam_token,
                    'filename': os.path.join(root_path, cam_data['filename']),
                }
            
            info.append(sample_info)
            sample_token = sample['next']
    
    # 保存mini版info文件
    with open(os.path.join(out_path, 'nuscenes_infos_mini.pkl'), 'wb') as f:
        pickle.dump(info, f)
    print(f"Success! mini版info文件已保存到: {os.path.join(out_path, 'nuscenes_infos_mini.pkl')}")

if __name__ == '__main__':
    # 数据集根路径（容器内的/data，即mini版数据）
    ROOT_PATH = './data'
    # 输出路径（和根路径一致）
    OUT_PATH = './data'
    # 安装nuscenes-dev-kit（若未安装）
    try:
        from nuscenes import NuScenes
    except ImportError:
        os.system('pip install nuscenes-dev-kit -i https://pypi.tuna.tsinghua.edu.cn/simple')
        from nuscenes import NuScenes
    
    create_nuscenes_mini_info(ROOT_PATH, OUT_PATH)

