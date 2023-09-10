import abc
import collections

import typing
from typing import Any, Callable, Mapping, Optional, Sequence, Set, Text, Union

from absl import logging
from dm_control import composer, mjcf
from dm_control.composer.observation import observable as base_observable
from dm_control.locomotion.mocap import loader

from dm_control.locomotion.tasks.reference_pose import datasets
from dm_control.locomotion.tasks.reference_pose import types
from dm_control.locomotion.tasks.reference_pose import utils
from dm_control.locomotion.tasks.reference_pose import rewards
from dm_control.locomotion.tasks.reference_pose import tracking

from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils import transformations as tr

from dm_env import specs

import numpy as np
import tree

if typing.TYPE_CHECKING:
  from dm_control.locomotion.walkers import legacy_base
  from dm_control import mjcf

mjlib = mjbindings.mjlib
DEFAULT_PHYSICS_TIMESTEP = 0.005
_MAX_END_STEP = 10000

class MocapClipTrackingContinuation(tracking.MultiClipMocapTracking):
    
    def __init__(self,
        walker: Callable[..., 'legacy_base.Walker'],
        arena: composer.Arena,
        ref_path: Text,
        ref_steps: Sequence[int],
        dataset: Union[Text, Sequence[Any]],
        termination_error_threshold: float = 0.3,
        prop_termination_error_threshold: float = 0.1,
        min_steps: int = 10,
        reward_type: Text = 'comic',
        physics_timestep: float = DEFAULT_PHYSICS_TIMESTEP,
        always_init_at_clip_start: bool = False,
        proto_modifier: Optional[Any] = None,
        prop_factory: Optional[Any] = None,
        disable_props: bool = True,
        ghost_offset: Optional[Sequence[Union[int, float]]] = None,
        body_error_multiplier: Union[int, float] = 1.0,
        actuator_force_coeff: float = 0.015,
        enabled_reference_observables: Optional[Sequence[Text]] = None,
        max_speed=4.5,
        reward_margin=0.75,
        direction_exponent=1.,
        target_speed=4):
        
        super().__init__(
            walker=walker,
            arena=arena,
            ref_path=ref_path,
            ref_steps=ref_steps,
            termination_error_threshold=termination_error_threshold,
            prop_termination_error_threshold=prop_termination_error_threshold,
            min_steps=min_steps,
            dataset=dataset,
            reward_type=reward_type,
            physics_timestep=physics_timestep,
            always_init_at_clip_start=always_init_at_clip_start,
            proto_modifier=proto_modifier,
            prop_factory=prop_factory,
            disable_props=disable_props,
            ghost_offset=ghost_offset,
            body_error_multiplier=body_error_multiplier,
            actuator_force_coeff=actuator_force_coeff,
            enabled_reference_observables=enabled_reference_observables,
            )
        
        self._walker.observables.add_observable(
            'time_in_clip',
            base_observable.Generic(self.get_normalized_time_in_clip))
        self.direction_of_speed = np.array([])
        self._max_speed = max_speed
        self._reward_margin = reward_margin
        self._direction_exponent = direction_exponent
        self._target_speed = target_speed

        self._move_speed = 0.
        self._move_angle = 0.
        self._move_speed_counter = 0.
        
    def get_reward(self, physics: mjcf.Physics) -> float:
       
        # HACK THAT ONLY SHOULD WORK FOR RUNNING DATASET
        # print("time step: ", self._time_step)
        
        reward, unused_debug_outputs, reward_channels = self._reward_fn(
            termination_error=self._termination_error,
            termination_error_threshold=self._termination_error_threshold,
            reference_features=self._current_reference_features,
            walker_features=self._walker_features,
            reference_observations=self._reference_observations)
        
        # self.collected_ref_features.append(self._current_reference_features)

        if 'actuator_force' in self._reward_keys:
            reward_channels['actuator_force'] = -self._actuator_force_coeff*np.mean(
                np.square(self._walker.actuator_force(physics)))
        
        # print("time step: ", self._time_step)
        # print(self._walker.observables)

        if self._time_step < 48:
            
            self._should_truncate = self._termination_error > self._termination_error_threshold

            if self._props:
                prop_termination = self._prop_termination_error > self._prop_termination_error_threshold
                self._should_truncate = self._should_truncate or prop_termination
        
            return reward
        
        else:     
            xvel = self._walker.observables.torso_xvel(physics)
            yvel = self._walker.observables.torso_yvel(physics)
            speed = np.linalg.norm([xvel, yvel])  
            
            speed_error = (self._target_speed - speed)
            speed_reward = np.exp(-(speed_error / self._reward_margin)**2)
            if np.isclose(xvel, 0.) and np.isclose(yvel, 0.):
                angle_reward = 1.
            else:
                direction = np.array([xvel, yvel])
                direction /= np.linalg.norm(direction)
                direction_tgt = np.array([0, -1])
                dot = direction_tgt.dot(direction)
                angle_reward = ((dot + 1) / 2)**self._direction_exponent
                
            # print("time_step: ", self._time_step)
            # print("speed: ", speed)
            # print("direction: ", direction)
            # print("angle_reward: ", angle_reward)
                
            self._should_truncate = speed_error**2 > self._termination_error_threshold
            return speed_reward + angle_reward
