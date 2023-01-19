import numpy as np

from tf_agents.metrics import py_metrics
from tf_agents.typing import types
from tf_agents.utils import nest_utils
from tf_agents.utils import numpy_storage

class AverageDistanceToPickup(py_metrics.StreamingMetric):
    def __init__(
        self,
        env,
        name='AverageDistanceToPickup',
        buffer_size=10,
        batch_size=None
    ):
        self._env = env
        super(AverageDistanceToPickup, self).__init__(
            name, buffer_size=buffer_size, batch_size=batch_size,
        )
    
    def _reset(self):
        pass
    
    def _batched_call(self, trajectory):
        lasts = trajectory.is_last()

        if np.any(lasts):
            is_last = np.where(lasts)
            pickup_distance = np.asarray(self._env.min_dist_to_pickup, np.float32)

            if pickup_distance.shape is ():
                pickup_distance = nest_utils.batch_nested_array(pickup_distance)
            
            self.add_to_buffer(pickup_distance[is_last])

class AverageDistanceToGoal(py_metrics.StreamingMetric):
    def __init__(
        self,
        env,
        name='AverageDistanceToGoal',
        buffer_size=10,
        batch_size=None
    ):
        self._env = env
        super(AverageDistanceToGoal, self).__init__(
            name, buffer_size=buffer_size, batch_size=batch_size,
        )
    
    def _reset(self):
        pass
    
    def _batched_call(self, trajectory):
        lasts = trajectory.is_last()

        if np.any(lasts):
            is_last = np.where(lasts)
            goal_distance = np.asarray(self._env.min_dist_to_goal, np.float32)

            if goal_distance.shape is ():
                goal_distance = nest_utils.batch_nested_array(goal_distance)
            
            self.add_to_buffer(goal_distance[is_last])

class AverageObjectDistanceToGoal(py_metrics.StreamingMetric):
    def __init__(
        self,
        env,
        name='AverageObjectDistanceToGoal',
        buffer_size=10,
        batch_size=None
    ):
        self._env = env
        super(AverageObjectDistanceToGoal, self).__init__(
            name, buffer_size=buffer_size, batch_size=batch_size,
        )
    
    def _reset(self):
        pass
    
    def _batched_call(self, trajectory):
        lasts = trajectory.is_last()

        if np.any(lasts):
            is_last = np.where(lasts)
            object_distance = np.asarray(self._env.obj_dist_to_goal, np.float32)

            if object_distance.shape is ():
                object_distance = nest_utils.batch_nested_array(object_distance)
            
            self.add_to_buffer(object_distance[is_last])

class AverageSuccessMetric(py_metrics.StreamingMetric):
    def __init__(
        self,
        env,
        name='AverageSuccessMetric',
        buffer_size=10,
        batch_size=None
    ):
        self._env = env
        super(AverageSuccessMetric, self).__init__(
            name, buffer_size=buffer_size, batch_size=batch_size,
        )
    
    def _reset(self):
        pass
    
    def _batched_call(self, trajectory):
        lasts = trajectory.is_last()

        if np.any(lasts):
            is_last = np.where(lasts)
            
            if self._env.succeeded:
                succeed = 1.0
            else:
                succeed = 0.0
            
            succeed = np.asarray(succeed, np.float32)

            if succeed.shape is ():
                succeed = nest_utils.batch_nested_array(succeed)
            
            self.add_to_buffer(succeed[is_last])
