import cv2
import numpy as np

from badgr.envs.env import EnvSpec
from badgr.utils.np_utils import imrectify


class JackalEnvSpec(EnvSpec):

    def __init__(self):
        super(JackalEnvSpec, self).__init__(
            names_shapes_limits_dtypes=(
                ('images/rgb_left_depth', (96, 128, 1), (0, 65535), np.uint16),
                ('images/rgb_left_rgbd', (96, 128, 4), (0, 65535), np.uint16),
                ('images/rgb_left', (96, 128, 3), (0, 255), np.uint8),
                #('images/rgb_left', (96, 128, 4), (0, 65535), np.uint16),
                ('collision/close', (1,), (0, 1), np.bool),
                ('jackal/position', (3,), (-0.5, 0.5), np.float32),
                ('jackal/yaw', (1,), (-np.pi, np.pi), np.float32),
                ('bumpy', (1,), (0, 1), np.bool),
                # NOTE: eval limit has to change correspondingly
                ('commands/angular_velocity', (1,), (-1.5, 1.5), np.float32),
                ('commands/linear_velocity', (1,), (0.0, 1.5), np.float32)
            )
        )

        fx, fy, cx, cy = 384.944396973, 384.575073242, 309.668579102, 243.29864502
        self._dim = (640, 480)
        self._K = np.array([[fx, 0., cx],
                      [0., fy, cy],
                      [0., 0., 1.]])
        self._D = np.array([[-0.0548635944724, 0.0604563839734, -0.00111321196891, -4.80580529256e-05, -0.0191334541887]]).T
        self._balance = 0.5

    @property
    def observation_names(self):
        return (
            'images/rgb_left',
            'images/rgb_left_depth',
            'images/rgb_left_rgbd',
            'images/rgb_right',
            'images/thermal',
            'collision/close',
            'collision/flipped',
            'collision/stuck',
            'collision/any',
            'gps/is_fixed',
            'gps/latlong',
            'imu/angular_velocity',
            'imu/compass_bearing',
            'imu/linear_acceleration',
            'jackal/angular_velocity',
            'jackal/linear_velocity',
            'jackal/imu/angular_velocity',
            'jackal/imu/linear_acceleration',
            'jackal/position',
            'jackal/yaw',
            'android/illuminance',
        )

    @property
    def output_observation_names(self):
        return (name for name in self.observation_names if 'rgb' not in name)

    @property
    def action_names(self):
        return (
            'commands/angular_velocity',
            'commands/linear_velocity'
        )

    def process_image(self, name, image):
        if len(image.shape) == 4:
            return np.array([self.process_image(name, im_i) for im_i in image])

        if name in ('images/rgb_left', 'images/rgb_left_depth', 'images/rgb_right'):
            image = imrectify(image, self._K, self._D, balance=self._balance)
        
        if (name == 'images/rgb_left_depth'):
            return super(JackalEnvSpec, self).process_depth_image(name, image)

        return super(JackalEnvSpec, self).process_image(name, image)

    @property
    def image_intrinsics(self):
        # return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        #     self._K, self._D, self._dim, np.eye(3), balance=self._balance)
        return self._K

    @property
    def image_distortion(self):
        return self._D


class JackalPositionCollisionEnvSpec(JackalEnvSpec):

    def __init__(self, left_image_only=False):
        self._left_image_only = left_image_only
        super(JackalPositionCollisionEnvSpec, self).__init__()

    @property
    def observation_names(self):
        names = [
            'images/rgb_left_depth',
            'images/rgb_left',
            'images/rgb_left_rgbd',

            'jackal/position',
            'jackal/yaw',

            'collision/close',  # NOTE: only required for training
        ]
        if not self._left_image_only:
            names.append('images/rgb_right')
        return tuple(names)


class JackalBumpyEnvSpec(JackalEnvSpec):

    def __init__(self, left_image_only=False):
        self._left_image_only = left_image_only
        super(JackalBumpyEnvSpec, self).__init__()

    @property
    def observation_names(self):
        names = [
            'images/rgb_left',

            'bumpy',
        ]
        if not self._left_image_only:
            names.append('images/rgb_right')
        return tuple(names)
