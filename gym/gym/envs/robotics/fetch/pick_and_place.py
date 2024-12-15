import os
from gym import utils
from gym.envs.robotics import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')
MODEL_XML_PATH_LARGE = os.path.join('fetch', 'pick_and_place_large_obj.xml')


class FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', action_type='pos'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        if action_type == 'pos':
            model_path = MODEL_XML_PATH
        elif action_type == 'posrot':
            model_path = MODEL_XML_PATH_LARGE
        fetch_env.FetchEnv.__init__(
            self, model_path, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, action_type=action_type)
        utils.EzPickle.__init__(self)
