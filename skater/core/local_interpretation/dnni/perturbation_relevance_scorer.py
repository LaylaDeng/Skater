from skater.util.image_ops import view_windows
from skater.core.local_interpretation.dnni.initializer import Initializer
from skater.util.logger import build_logger
from skater.util.logger import _INFO
from skater.util.exceptions import TensorflowUnavailableError
try:
    import tensorflow as tf
except ImportError:
    raise (TensorflowUnavailableError("TensorFlow binaries are not installed"))

import numpy as np


class BasePerturbationMethod(Initializer):
    """
    Base class for perturbation-based relevance/attribution computation

    """

    __name__ = "BasePerturbationMethod"
    logger = build_logger(_INFO, __name__)

    def __init__(self, output_tensor, input_tensor, samples, current_session):
        super(BasePerturbationMethod, self).__init__(output_tensor, input_tensor, samples, current_session)


class Occlusion(BasePerturbationMethod):
    """
    Reference
    - https://github.com/marcoancona/DeepExplain/blob/master/deepexplain/tensorflow/methods.py
    """
    __name__ = "Occlusion"
    logger = build_logger(_INFO, __name__)

    def __init__(self, output_tensor, input_tensor, samples, current_session, window_shape=None, step=1):
        super(Occlusion, self).__init__(output_tensor, input_tensor, samples, current_session)

        self.input_shape = samples.shape[1:]
        if window_shape is not None:
            assert len(self.window_shape) == len(self.input_shape), \
                'window_shape and input must match{}'.format(len(self.input_shape))
            self.window_shape = tuple(self.window_shape)
        else:
            self.window_shape = (1,) * len(self.input_shape)

        self.replace_value = 0.0
        # the input samples are expected to be of the shape,
        # (1, 150, 150, 3) <batch_size, image_width, image_height, no_of_channels>
        self.batch_size = self.samples.shape[0]
        self.total_dim = np.prod(self.input_shape)  # e.g. 268203 = 299*299*3
        Occlusion.logger.info('Input shape: {}; window_shape {}; step {}'.format((self.input_shape,
                                                                                  self.window_shape, self.step)))


    def run(self):
        self.__session_run()
        # Create a rolling window view of the input matrix
        # sample input is of the following shape (1, 150, 150, 3) <batch_size, image_width, image_height, no_of_channels>
        # self.samples[0] returns the actual input shape (150, 150, 3)
        input_patches = view_windows(self.samples[0], self.window_shape, self.step).reshape((-1,) + self.window_shape)
        heatmap = np.zeros_like(self.samples, dtype=np.float32).reshape((-1), self.total_dim)

        # Compute original output
        eval0 = self._session_run()

        # Perturb through the feature space by replacing and masking
        for item, index in enumerate(input_patches):
            # create a mask
            mask = np.ones(self.input_shape).flatten()
            mask[index.flatten()] = self.replace_value
            masked_input = mask.reshape((1,) + self.input_shape) * self.samples
            delta = eval0 - self._run_input(masked_input)
            delta_aggregated = np.sum(delta.reshape((self.batch_size, -1)), -1, keepdims=True)
            heatmap[:, index.flatten()] += delta_aggregated
            return heatmap
