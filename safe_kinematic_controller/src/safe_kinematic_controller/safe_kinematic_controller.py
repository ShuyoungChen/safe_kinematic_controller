# Copyright (c) 2017, Rensselaer Polytechnic Institute, Wason Technology LLC
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Rensselaer Polytechnic Institute, or Wason 
#       Technology LLC, nor the names of its contributors may be used to 
#       endorse or promote products derived from this software without 
#       specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import QuadProg as qp
import numpy as np
import general_robotics_toolbox as rox
from collections import namedtuple
from scipy.interpolate import PchipInterpolator, interp1d
import threading
import traceback
import copy
import abc
from enum import Enum
import time

class ControllerMode(Enum):

	INVALID_EXTERNAL_SETPOINT = -17
	FT_THRESHOLD_VIOLATION = -16
	SENSOR_INVALID_STATE=-15
	SENSOR_COMMUNICATION_ERROR = -14
	SENSOR_FAULT = -13
	COLLISION_IMMINENT = -12
	TRAJECTORY_ABORTED = -11
	TRAJECTORY_TRACKING_ERROR = -10
	INVALID_TRAJECTORY = -9
	SETPOINT_TRACKING_ERROR = -8
	ROBOT_SINGULARITY = -7
	ROBOT_INVALID_STATE = -6
	ROBOT_COMMUNICATION_ERROR = -5
	ROBOT_FAULT = -4
	INVALID_ARGUMENT = -3
	INVALID_OPERATION = -2
	INTERNAL_ERROR = -1
	HALT = 0
	JOINT_TELEOP = 1
	CARTESIAN_TELEOP = 2
	CYLINDRICAL_TELEOP = 3
	SPHERICAL_TELEOP = 4
	SHARED_TRAJECTORY = 5
	AUTO_TRAJECTORY = 6
	EXTERNAL_SETPOINT_0 = 7
	EXTERNAL_SETPOINT_1 = 8
	EXTERNAL_SETPOINT_2 = 9
	EXTERNAL_SETPOINT_3 = 10
	EXTERNAL_SETPOINT_4 = 11
	EXTERNAL_SETPOINT_5 = 12
	EXTERNAL_SETPOINT_6 = 13
	EXTERNAL_SETPOINT_7 = 14
	CUSTOM_MODE_0 = 0xFFFF
	SUCCESS = 0x7FFFFFFF

class ControllerState(object):
	def __init__(self, controller_time=0, ts=0, mode=ControllerMode.HALT, error_msg=None, joint_position=None, joint_setpoint=None, \
			      joint_command=None, ft_wrench=None, ft_wrench_status=None, ft_wrench_bias = None, ft_wrench_threshold=None,\
			      active_trajectory=None, joystick_command=None, speed_scalar=1, external_joint_setpoints=[None]*8):
		self.controller_time = controller_time
		self.ts = ts
		self.mode = mode
		self.error_msg = error_msg
		self.joint_position = joint_position
		self.joint_command = joint_command
		self.joint_setpoint = joint_setpoint
		self.ft_wrench = ft_wrench
		if ft_wrench_status is None:
			self.ft_wrench_status = ControllerMode.SENSOR_FAULT
		else:
			self.ft_wrench_status = ft_wrench_status
		if ft_wrench_bias is None:
			ft_wrench_bias = np.zeros((6,))
		self.ft_wrench_bias = ft_wrench_bias
		self.ft_wrench_threshold = ft_wrench_threshold
		self.active_trajectory = active_trajectory
		self.joystick_command = joystick_command
		self.speed_scalar = speed_scalar
		self.external_joint_setpoints = external_joint_setpoints

class Controller(object):

	def __init__(self, robot, ts, clock=None, max_joint_vel = None, joint_trajectory_manager = None):
	
		assert robot is not None
		assert len(robot.joint_type) == len(robot.joint_names)
		assert joint_trajectory_manager is not None           
	 
		self._robot = robot        
		self._lock = threading.Lock()
		if clock is None:
			self._clock = SystemClock()
		else:
			self._clock = clock
	
		if max_joint_vel is not None:
			self._max_joint_vel = max_joint_vel
		else:
			self._max_joint_vel = robot.joint_vel_limit
	
		self._state = ControllerState(controller_time = self._clock.now(), ts=ts)
		self._joint_trajectory_manager = joint_trajectory_manager
	
		####################################################
		self._quadprog = qp.QuadProg(self._robot)
		####################################################

	def read_robot_joint_position(self):
		return ControllerMode.ROBOT_FAULT, None 

	def read_ft_wrench(self):
		return ControllerMode.SENSOR_FAULT, None

	def write_robot_command(self, setpoint):
		return ControllerMode.ROBOT_FAULT

	def write_robot_no_command(self):
		return ControllerMode.SUCCESS

	def read_external_setpoint(self, index):
		return ControllerMode.INVALID_EXTERNAL_SETPOINT, None

	def read_external_setpoints(self):
		o = [None]*8
		for i in xrange(8):
			res, p = self.read_external_setpoint(i)
			if res.value > 0:
			    #TODO: verify external setpoint is valid
			    o[i] = p
		return o

	def read_joystick_command(self):
		return JoystickCommand(np.zeros((6,)), np.zeros((6,)), 0, False)

	def step(self):
		with self._lock:
			self._step()

	def _step(self):
	
		robot_res, joint_position = self.read_robot_joint_position()
		if robot_res.value > 0:
			assert np.shape(joint_position) == (len(self._robot.joint_type),)
			self._state.joint_position = joint_position            
			ft_sensor_res, ft_wrench = self.read_ft_wrench()
			self._state.ft_wrench_status = ft_sensor_res
			self._state.ft_wrench = ft_wrench           
		else:
			self._state.mode = robot_res
	
		self._state.external_joint_setpoints = self.read_external_setpoints()                
	
		self._state.joystick_command = self.read_joystick_command()
	
		self._compute_joint_command()
	
		if self._state.joint_command is not None:
			robot_write_res = self.write_robot_command(self._state.joint_command)
			if robot_write_res.value < 0:
			    self._state.mode=robot_write_res
		else:
			self.write_robot_no_command()
	
		self._publish_state(self._state)        
			

	def compute_joint_command(self, controller_state = None):
		with self._lock:
			return self._compute_joint_command(controller_state)

	def _compute_joint_command(self, controller_state = None):
		if controller_state is None:
			controller_state = self._state
	
		controller_state.controller_time = self._clock.now
		step_ts = controller_state.ts * controller_state.speed_scalar
			            
		if controller_state.mode.value < 0:
			if controller_state.mode == ControllerMode.ROBOT_INVALID_STATE \
			    or controller_state.mode == ControllerMode.ROBOT_COMMUNICATION_ERROR \
			    or controller_state.mode == ControllerMode.ROBOT_FAULT:
			                                
			    controller_state.joint_setpoint = None
			    controller_state.joint_command = None
			                               
			controller_state.ft_wrench = None
			while True:
			    self._update_active_trajectory(controller_state)
			    if controller_state.active_trajectory is not None:
			        controller_state.active_trajectory.abort(controller_state.mode)
			        controller_state.active_trajectory = None
			    else:
			        break
			    
			return controller_state.mode
		else:
			
			self._update_active_trajectory(controller_state)
	
		if controller_state.joint_setpoint is None:
			controller_state.joint_setpoint = controller_state.joint_position
	
		if controller_state.ft_wrench_threshold is not None \
		  and np.shape(controller_state.ft_wrench_threshold) == (6,) \
		  and controller_state.ft_wrench is None:
			controller_state.mode = ControllerMode.SENSOR_FAULT
			return ControllerMode.SENSOR_FAULT
	
		if controller_state.joystick_command is not None \
		  and controller_state.joystick_command.halt_command:
			controller_state.mode = ControllerMode.HALT
	
		if controller_state.mode.value > 0 and not self._check_ft_threshold(controller_state.ft_wrench, controller_state.ft_wrench_bias):
			if controller_state.active_trajectory is not None:
			    controller_state.active_trajectory.abort(ControllerMode.FT_THRESHOLD_VIOLATION, \
			                                             "Force/Torque Threshold Violated")
			    controller_state.active_trajectory = None
			controller_state.mode = ControllerMode.FT_THRESHOLD_VIOLATION
			return ControllerMode.FT_THRESHOLD_VIOLATION
		try:
			if controller_state.mode == ControllerMode.HALT:
			    pass
			elif controller_state.mode.value < ControllerMode.HALT.value:
			    if controller_state.active_trajectory is not None:
			        controller_state.active_trajectory.abort(controller_state.mode)
			        controller_state.active_trajectory = None
			elif controller_state.mode == ControllerMode.JOINT_TELEOP:
			    if controller_state.joystick_command is not None \
			      and controller_state.joystick_command.joint_velocity_command is not None:
			        #controller_state.joint_setpoint += \
			          #controller_state.joystick_command.joint_velocity_command.dot(step_ts)
			        ####################################################################
			        J = rox.robotjacobian(self._robot, controller_state.joint_position)
			        cmd_vel = np.dot(J, controller_state.joystick_command.joint_velocity_command)
			        controller_state.joint_setpoint += self._quadprog.compute_quadprog(controller_state.joint_position).dot(step_ts*5)
			        ####################################################################
			        #self._clip_joint_angles(controller_state)
			elif controller_state.mode == ControllerMode.CARTESIAN_TELEOP:
			    if controller_state.joystick_command is not None \
			      and controller_state.joystick_command.spatial_velocity_command is not None:
			        #self._compute_joint_vel_from_spatial_vel(controller_state, step_ts, controller_state.joystick_command.spatial_velocity_command)  
			        ####################################################
			        ########## change step_ts*5 #################
			        controller_state.joint_setpoint += self._quadprog.compute_quadprog(controller_state.joint_position).dot(step_ts*2)
			        #self._clip_joint_angles(controller_state)
			        ####################################################                 
			elif controller_state.mode == ControllerMode.CYLINDRICAL_TELEOP:
			    if controller_state.joystick_command is not None \
			      and controller_state.joystick_command.spatial_velocity_command is not None:
			        if (not all(self._robot.H[:,0] == (0,0,1)) or self._robot.joint_type[0] != 0):
			            controller_state.mode = ControllerMode.INVALID_OPERATION
			        else:
			            cmd_vel = controller_state.joystick_command.spatial_velocity_command
			            transform_0T = rox.fwdkin(self._robot, controller_state.joint_position)
			            d = np.linalg.norm((transform_0T.p[0], transform_0T.p[1]))
			            d = np.max((d,0.05))
			            theta = np.arctan2(transform_0T.p[1], transform_0T.p[0])
			            s_0 = transform_0T.p[1] / d
			            c_0 = transform_0T.p[0] / d
			            v_x = -d*s_0*cmd_vel[3] + c_0*cmd_vel[4]
			            v_y =  d*c_0*cmd_vel[3] + s_0*cmd_vel[4]
			            v_z = cmd_vel[5]
			            w_x = cmd_vel[0]
			            w_y = cmd_vel[1]
			            w_z = cmd_vel[2] + cmd_vel[3]
			            cmd_vel2 = np.array([w_x, w_y, w_z, v_x, v_y, v_z])               
			            #self._compute_joint_vel_from_spatial_vel(controller_state, step_ts, cmd_vel2)
			            ########## change step_ts*5 #################
			            controller_state.joint_setpoint += self._quadprog.compute_quadprog(controller_state.joint_position).dot(step_ts*5)
			        	####################################################  
			    
			elif controller_state.mode == ControllerMode.SPHERICAL_TELEOP:
			    if controller_state.joystick_command is not None \
			      and controller_state.joystick_command.spatial_velocity_command is not None:
			        if (not all(self._robot.H[:,0] == (0,0,1)) or self._robot.joint_type[0] != 0):
			            controller_state.mode = ControllerMode.INVALID_OPERATION
			        else:
			            #TODO: more clever solution that doesn't require trigonometry?
			            cmd_vel = controller_state.joystick_command.spatial_velocity_command
			            transform_0T = rox.fwdkin(self._robot, controller_state.joint_position)
			            d = np.linalg.norm(transform_0T.p)
			            d = np.max((d,0.05))
			            theta_phi_res = rox.subproblem2(np.dot([1,0,0],d), transform_0T.p, [0,0,1], [0,1,0])
			            if (len(theta_phi_res) == 0):
			                controller_state.mode = ControllerMode.ROBOT_SINGULARITY
			            else:
			                theta_dot = cmd_vel[3]
			                phi_dot = cmd_vel[4]
			                d_dot = cmd_vel[5]
			                if (len(theta_phi_res) == 1):
			                    theta, phi = theta_phi_res[0]
			                elif (np.abs(theta_phi_res[0][1]) < np.deg2rad(90)):                                
			                    theta, phi = theta_phi_res[0]
			                else:
			                    theta, phi = theta_phi_res[1]
			                                            
			                s_theta = np.sin(theta)
			                c_theta = np.cos(theta)
			                s_phi = np.sin(phi)
			                c_phi = np.cos(phi)
			                v_x = -d*s_phi*c_theta*phi_dot -d*s_theta*c_theta*theta_dot + c_phi*c_theta*d_dot
			                v_y = -d*s_phi*s_theta*phi_dot + d*c_phi*c_theta*theta_dot + s_theta*c_phi*d_dot
			                v_z = -d*c_phi*phi_dot - s_phi*d_dot                            
			                w_x = cmd_vel[0] - phi_dot*s_theta
			                w_y = cmd_vel[1] + phi_dot*c_theta
			                w_z = cmd_vel[2] + theta_dot
			                cmd_vel2 = np.array([w_x, w_y, w_z, v_x, v_y, v_z])               
			                #self._compute_joint_vel_from_spatial_vel(controller_state, step_ts, cmd_vel2)
			                ########## change step_ts*5 ###################
			                controller_state.joint_setpoint += self._quadprog.compute_quadprog(controller_state.joint_position).dot(step_ts*5)
			                ###############################################
			elif controller_state.mode == ControllerMode.SHARED_TRAJECTORY:
			    if controller_state.joystick_command is not None \
			      and controller_state.joystick_command.trajectory_velocity_command is not None:
			        active_trajectory = controller_state.active_trajectory
			        if active_trajectory is not None and active_trajectory.trajectory_valid:
			            res, setpoint = active_trajectory.increment_trajectory_time(
			                step_ts * controller_state.joystick_command.trajectory_velocity_command, controller_state)
			            if res == ControllerMode.SUCCESS:
			                controller_state.joint_setpoint = setpoint
			elif controller_state.mode == ControllerMode.AUTO_TRAJECTORY:
			    active_trajectory = controller_state.active_trajectory
			    if active_trajectory is not None and active_trajectory.trajectory_valid:
			        res, setpoint = active_trajectory.increment_trajectory_time(step_ts, controller_state)
			        if res == ControllerMode.SUCCESS:
			            controller_state.joint_setpoint = setpoint
			elif controller_state.mode.value >= ControllerMode.EXTERNAL_SETPOINT_0.value \
			    and controller_state.mode.value <= ControllerMode.EXTERNAL_SETPOINT_7.value:
			    i = controller_state.mode.value - ControllerMode.EXTERNAL_SETPOINT_0.value
			    setpoint = controller_state.external_joint_setpoints[i]
			    if setpoint is not None:
			        controller_state.joint_setpoint = np.copy(setpoint)
			        self._clip_joint_angles(controller_state)                    
			else:
			    self._compute_setpoint_custom_mode(controller_state)
		except:
			traceback.print_exc()
			controller_state.mode = ControllerMode.INTERNAL_ERROR
			controller_state.joint_command = controller_state.joint_position
	
		#TODO: add in joint command filter
		controller_state.joint_command = controller_state.joint_setpoint
	
		return controller_state.mode 
	##########################################

	def _update_active_trajectory(self, controller_state):
		if controller_state.active_trajectory is not None \
			and not controller_state.active_trajectory.trajectory_valid:
			controller_state.active_trajectory = None
	
		if controller_state.active_trajectory is None:
			next_trajectory = self._joint_trajectory_manager.next_trajectory(controller_state)
			if next_trajectory is None:
			    return False, None
			
			controller_state.active_trajectory = next_trajectory
   
	def _compute_setpoint_custom_mode(self, controller_state):
		if controller_state.active_trajectory is not None:
			controller_state.active_trajectory.abort(controller_state.mode)
			controller_state.active_trajectory = None
	
		controller_state.mode = ControllerMode.INVALID_ARGUMENT
		return controller_state.mode    

	def _compute_joint_vel_from_spatial_vel(self, controller_state, step_ts, vel):
		J = rox.robotjacobian(self._robot, controller_state.joint_position)
		joints_vel = np.linalg.pinv(J).dot(vel)
		joints_vel = np.clip(joints_vel, -self._max_joint_vel, self._max_joint_vel)
		controller_state.joint_setpoint += joints_vel.dot(step_ts)
		self._clip_joint_angles(controller_state)        
		return ControllerMode.SUCCESS
	

	def set_mode(self, mode):
		with self._lock:
			return self._set_mode(mode)
	
	def _set_mode(self, mode):        
		if self._state.mode.value < 0 and mode.value != 0:
			return ControllerMode.INVALID_ARGUMENT        
		if mode.value < 0:
			return ControllerMode.INVALID_ARGUMENT
		self._state.mode = mode
		return ControllerMode.SUCCESS

	def set_speed_scalar(self, speed_scalar):
		with self._lock:
			return self._set_speed_scalar(speed_scalar)
	
	def _set_speed_scalar(self, speed_scalar):
		if speed_scalar < 0 or speed_scalar > 5:
			return ControllerMode.INVALID_ARGUMENT
		self._state.speed_scalar = speed_scalar
		return ControllerMode.SUCCESS

	def set_ft_wrench_threshold(self, ft_threshold):
		with self._lock:
			return self._set_ft_wrench_threshold(ft_threshold)

	def _set_ft_wrench_threshold(self, ft_threshold):
		ft = ft_threshold
		if np.shape(ft) == (0,):
			pass
		elif np.shape(ft) != (6,):
			return ControllerMode.INVALID_ARGUMENT
		else:
			if np.any(ft < 0):
			    return ControllerMode.INVALID_ARGUMENT
		self._state.ft_wrench_threshold = ft
		return ControllerMode.SUCCESS 

	def set_ft_wrench_bias(self, ft_bias):
		with self._lock:
			return self._set_ft_wrench_bias(ft_bias)

	def _set_ft_wrench_bias(self, ft_bias):
		ft = ft_bias
		if np.shape(ft) == (0,):
			ft=np.zeros((6,))
		elif np.shape(ft) != (6,):
			return ControllerMode.INVALID_ARGUMENT        
		self._state.ft_wrench_bias = ft
		return ControllerMode.SUCCESS 

	def _check_ft_threshold(self, ft, ft_bias = None):
	
		if self._state.ft_wrench_threshold is None:
			return True
	
		if (ft_bias is None):
			ft_bias = np.zeros((6,))
		if np.shape(self._state.ft_wrench_threshold) != (6,):
			return True
		if ft is None:
			return False # Stop if invalid wrench        
		if np.all(self._state.ft_wrench_threshold < 1e-6):
			return True
		ft = ft - ft_bias
		if np.any(np.logical_and((self._state.ft_wrench_threshold > 1e-6), (np.abs(ft) > self._state.ft_wrench_threshold))):
			return False        
		return True

	def get_state(self):
		with self._lock:
			return copy.deepcopy(self._state)

	def _clip_joint_angles(self, controller_state):
		if controller_state is None:
			controller_state = self._state
		if self._robot.joint_lower_limit is not None:
			controller_state.joint_setpoint = np.clip(controller_state.joint_setpoint, \
			    self._robot.joint_lower_limit, self._robot.joint_upper_limit, \
			    controller_state.joint_setpoint)
			controller_state.joint_command = np.clip(controller_state.joint_command, \
			    self._robot.joint_lower_limit, self._robot.joint_upper_limit, \
			    controller_state.joint_command)
	  
	def _publish_state(self, state):
		pass

JointTrajectoryWaypoint = namedtuple("JointTrajectoryPoint", ["time_from_start", "positions", "velocities", "accelerations", "effort"])
JointTrajectoryTolerance = namedtuple("JointTrajectoryTolerance", ["positions", "velocities", "accelerations"])
JointTrajectoryWaypointPlan = namedtuple("JointTrajectoryWaypointPlan", ["waypoints", "start_tolerance", "path_tolerance", "goal_tolerance", "goal_time_tolerance"])

class JointTrajectory(object):
	def __init__(self, plan, robot):
		self._lock = threading.RLock()
		self._listeners = []
		self._plan = plan
		self._robot = robot
		self._joint_interpolators = self._init_joint_interpolators(plan)
		self._trajectory_t = 0
		self._trajectory_max_t = max(w.time_from_start for w in plan.waypoints)
		self._trajectory_valid = self._joint_interpolators is not None

	def _init_joint_interpolators(self, plan):
		#Create a default interpolator that uses pchip interpolator
		assert isinstance(plan, JointTrajectoryWaypointPlan)
		assert len(plan.waypoints) >= 2
			    
		interps=[]
		t = np.array([p.time_from_start for p in plan.waypoints])           
	
		if len(plan.waypoints) == 2:
			#Use linear interpolator for 2 points
			for j in xrange(len(self._robot.joint_type)):
			    t1=np.hstack((-10, t, t[-1] + 10))
			    x = np.array([p.positions[j] for p in plan.waypoints])
			    x1 = np.hstack((x[0], x, x[-1]))
			    pchip = interp1d(t1,x1)
			    interps.append(pchip) 
		else:
			#Use pchip interpolators for three or more points
			for j in xrange(len(self._robot.joint_type)):
			    t1=np.hstack((-10, t, t[-1] + 10))
			    x = np.array([p.positions[j] for p in plan.waypoints])
			    x1 = np.hstack((x[0], x, x[-1]))
			    pchip = PchipInterpolator(t1,x1)
			    interps.append(pchip)
			    
		return interps           

	def add_listener(self, listener):
		with self._lock:
			self._listeners.append(listener)
	
	def remove_listener(self, listener):
		with self._lock:
			self._listeners.remove(listener)
			    
	def abort(self, error_code = -1, error_string=""):
		with self._lock:
			listeners=copy.copy(self._listeners)
			self._trajectory_valid=False
	
		for l in listeners:
			try:
			    l.aborted(self, self._trajectory_t, error_code, error_string)
			except:
			    traceback.print_exc()

	@property
	def trajectory_valid(self):
		with self._lock:
			return self._trajectory_valid

	@property
	def trajectory_time(self):
		with self._lock:
			return self._trajectory_t
	
	@property
	def trajectory_max_time(self):
		with self._lock:
			return self._trajectory_max_t
	
	def _set_trajectory_time(self, t, controller_state):
			    
		if not self._trajectory_valid:
			return ControllerMode.INVALID_TRAJECTORY, None
			
		if t > self._trajectory_max_t or t < 0.0:
			return ControllerMode.INVALID_OPERATION, None
			            
		self._trajectory_t = t        
	
		trajectory_angles = self._get_trajectory_joint_angles(t)
	
		if (t == 0.0):
			if (np.all(np.abs(controller_state.joint_position - trajectory_angles) > self._plan.start_tolerance.positions)):
			    self.abort(ControllerMode.INVALID_TRAJECTORY, "Trajectory tracking error")
			    return ControllerMode.INVALID_TRAJECTORY, None
	
		#if (np.all(np.abs(controller_state.joint_position - trajectory_angles) > self._plan.path_tolerance.positions)):
		#    self.abort(ControllerMode.TRAJECTORY_TRACKING_ERROR, "Trajectory tracking error")
		#    return ControllerMode.TRAJECTORY_TRACKING_ERROR, None
	
	
		if t >= self._trajectory_max_t:
			if (np.all(np.abs(controller_state.joint_position - trajectory_angles) < self._plan.goal_tolerance.positions)):
			    for l in self._listeners:
			        try:
			            l.completed(self._trajectory_t, trajectory_angles)
			        except:
			            traceback.print_exc()                    
			    self._trajectory_valid=False
		else:
			for l in self._listeners:
			    try:
			        l.setpoint_updated(self, self._trajectory_t, trajectory_angles, controller_state)
			    except:
			        traceback.print_exc()
			    
		return ControllerMode.SUCCESS, trajectory_angles

	def _get_trajectory_joint_angles(self, t):        
		return np.array([interp(t) for interp in self._joint_interpolators])

	def set_trajectory_time(self,t, controller_state):
		with self._lock:            
			return self._set_trajectory_time(t, controller_state)
			
	def increment_trajectory_time(self, dt, controller_state):
		with self._lock:
			if not self._trajectory_valid:
			    return False, None
			          
			t2 = self._trajectory_t + dt
			if t2 < 0: t2 = 0
			if t2 > self._trajectory_max_t: t2 = self._trajectory_max_t
			
			return self._set_trajectory_time(t2, controller_state)

class JointTrajectoryListener:
	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def aborted(self, trajectory, trajectory_time, error_code, error_string):
		pass

	@abc.abstractmethod
	def completed(self, trajectory, trajectory_time):
		pass

	@abc.abstractmethod
	def setpoint_updated(self, trajectory, trajectory_time, joint_command, controller_state):
		pass

class JointTrajectoryManager(object):
	def __init__(self, robot, trajectory_type = JointTrajectory):
		assert robot is not None
		self._trajectory_type=trajectory_type
		self._trajectories=[]
		self._lock = threading.Lock()
		self._robot = robot

	def trajectory_received(self, trajectory):
		with self._lock:        
			self._trajectories.append(trajectory)

	def next_trajectory(self, controller_state):
		with self._lock:
			if len(self._trajectories) > 0:
			    return self._trajectories.pop(0)
			else:
			    return None
			
	def abort_all(self):
		with self._lock:
			trajectories = self._trajectories
			self._trajectories = []
	
		for t in trajectories:
			t.abort()

class Clock:
	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def now(self):
		"""timestamps in seconds as float"""
		pass

class SystemClock(Clock):

	def now(self):
		return time.time()

JoystickCommand = namedtuple("JoystickCommand", ["joint_velocity_command", "spatial_velocity_command", \
			                                     "trajectory_velocity_command", "halt_command"])
class JoystickHardwareInterface:
	__metaclass__ = abc.ABCMeta
	@abc.abstractmethod
	def read_joystick(self):
		return None

##########################
#robot=robot_host=rospy.get_param('~abb_irc5_uri')
#controller = Controller(robot, 0.004)
#controller.compute_joint_command()    
 ######################
