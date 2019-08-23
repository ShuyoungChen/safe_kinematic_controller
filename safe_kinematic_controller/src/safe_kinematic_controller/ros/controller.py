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

from __future__ import absolute_import

from ..safe_kinematic_controller import Controller, ControllerMode, JoystickCommand, \
    JointTrajectoryManager, JointTrajectory, JointTrajectoryListener, JointTrajectoryWaypointPlan, \
    JointTrajectoryWaypoint, JointTrajectoryTolerance

from sensor_msgs.msg import Joy, JointState
from geometry_msgs.msg import WrenchStamped
import rospy
import numpy as np
import threading
from std_msgs.msg import Float64
from general_robotics_toolbox import urdf as rox_urdf
from urdf_parser_py.urdf import URDF
from .. import msg as safe_kinematic_controller_msg
from .. import srv as safe_kinematic_controller_srv  
from geometry_msgs.msg import Wrench, Vector3
from actionlib import action_server
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryFeedback, FollowJointTrajectoryResult
from weakref import WeakValueDictionary

class ROSXboxGamepadInterface(object):
    
    def __init__(self, n_joints):
        self._lock=threading.Lock()        
                
        self._cmd_vel=None
        self._cmd_vel_time=rospy.Time.from_sec(0)        
        self._joint_vel=None
        self._joint_vel_time=rospy.Time.from_sec(0)
        self._trajectory_vel=None
        self._trajectory_vel_time=rospy.Time.from_sec(0)
        
        assert n_joints == 5 or n_joints == 6 or n_joints == 7
        
        if n_joints == 5:
            self._joint_vel_gain=np.array([-0.5, 1, 1, 1, 1])*np.deg2rad(5)
        elif n_joints == 6:
            self._joint_vel_gain=np.array([-0.5, 1, 1, 1, 1, -1])*np.deg2rad(5)
        else:
            self._joint_vel_gain=np.array([-0.5, 1, 1, 1, 1, 1, -1])*np.deg2rad(5)
        #self._cmd_vel_gain=np.array([-0.15, 0.15, 0.05, 0.087, 0.175, 0.087])
        self._cmd_vel_gain=np.array([-0.15, 0.15, 0.05, 0.047, 0.087, 0.047])
        self._cmd_halt=False
               
        self.n_joints = n_joints
        
        self._joy_subscription=rospy.Subscriber("joy", Joy, self._joy_callback)
    
    def _joy_callback(self, data):
        with self._lock:            
            if any(data.buttons[0:4]):
                #Trajectory shared control
                if data.buttons[0] != 0:
                    self._trajectory_vel = 0.25 * data.axes[1]                
                elif data.buttons[1] != 0:
                    self._trajectory_vel = data.axes[1]                
                elif data.buttons[2] != 0:
                    self._trajectory_vel = 0.25                
                elif data.buttons[3] != 0:
                    self._trajectory_vel = 1
                self._trajectory_vel_time=rospy.Time.now()
                self._joint_vel=None
                self._cmd_vel=None                                
                               
            elif data.buttons[4] != 0:
                #Joint level teleop control
                if self.n_joints == 5:
                    self._joint_vel = np.array([data.axes[0], data.axes[1], data.axes[4], \
                                          data.axes[6], data.axes[7]]) \
                                          *self._joint_vel_gain
                elif self.n_joints == 6:           
                    self._joint_vel = np.array([data.axes[0], data.axes[1], data.axes[4], \
                                          data.axes[6], data.axes[7], data.axes[3]]) \
                                          *self._joint_vel_gain
                else:
                    #TODO: Control joint 3 with gamepad
                    self._joint_vel = np.array([data.axes[0], data.axes[1], 0, data.axes[4], \
                                          data.axes[6], data.axes[7], data.axes[3]]) \
                                          *self._joint_vel_gain
                
                self._joint_vel_time=rospy.Time.now()
                self._trajectory_vel=None
                self._cmd_vel=None
            elif data.buttons[5] != 0:
                #Cartesian teleop control
                self._cmd_vel = np.array([data.axes[7], data.axes[6], data.axes[3], \
                                          data.axes[0], data.axes[1], data.axes[4]]) \
                                          *self._cmd_vel_gain
                self._cmd_vel_time=rospy.Time.now()
                self._trajectory_vel=None
                self._joint_vel=None
            else:
                self._joint_vel = None
                self._cmd_vel=None
                self._trajectory_vel=None
            
            self._cmd_halt = data.axes[2] < 0 or data.axes[5] < 0
    
    def read_joystick_command(self):
        with self._lock:
            #Clear stale command data
            now=rospy.Time.now()
            dt=rospy.Duration(0.5) #Half second timeout
            if (self._joint_vel is not None) and (now - self._joint_vel_time > dt):
                self._joint_vel=np.zeros((6,))
            if (self._cmd_vel is not None) and (now - self._cmd_vel_time > dt):
                self._cmd_vel=np.zeros((6,))
            if (self._trajectory_vel is not None) and (now-self._trajectory_vel_time > dt):
                self._trajectory_vel=0        
            
            return JoystickCommand(self._joint_vel, self._cmd_vel, self._trajectory_vel, self._cmd_halt)
            

class ROSController(Controller):
    
    def __init__(self, robot, ts, clock = None, joint_trajectory_manager = None, joystick_interface=None, \
                 robot_interface = None, ft_sensor_interface = None):
        
        if joint_trajectory_manager is None:
            joint_trajectory_manager = ROSJointTrajectoryManager(robot)
        
        super(ROSController,self).__init__(robot, ts, clock, joint_trajectory_manager = joint_trajectory_manager)
        if joystick_interface is not None:
            self._joystick_interface = joystick_interface
        else:
            self._joystick_interface = ROSXboxGamepadInterface(len(robot.joint_type))
        
        self._robot_interface = robot_interface
        self._ft_sensor_interface = ft_sensor_interface
        
        self._joint_state_subscriber = None
        self._joint_command_publishers = []
        self._ft_wrench_subscriber = None
        self._external_setpoint_subscribers = []
        self._publish_state_publisher = None
        
        self._joint_states_position = np.full((len(self._robot.joint_type),), np.nan)
        self._ft_wrench = None
        self._external_setpoint = [None]*8
        
        
        self._init_read_robot_state()
        self._init_write_robot_command()
        self._init_read_ft_sensor()
        self._init_read_external_setpoints()
        self._init_publish_state()
        
        self._set_controller_mode_service=rospy.Service("set_controller_mode", safe_kinematic_controller_srv.SetControllerMode, self._set_controller_mode_cb)
        self._get_controller_state_service=rospy.Service("get_controller_state", safe_kinematic_controller_srv.GetControllerState, self._get_controller_state_cb)
                            
    def _init_read_robot_state(self):
        if self._robot_interface is not None:
            self._robot_interface.init_read_robot_state()
        else:
            self._joint_state_subscriber=rospy.Subscriber("joint_states", JointState, self._joint_states_cb)
    
    def _joint_states_cb(self, joint_states):
        
        with self._lock:
            for i in xrange(len(self._robot.joint_names)):
                j = joint_states.name.index(self._robot.joint_names[i])            
                self._joint_states_position[i] = joint_states.position[j] 
    
    def _init_write_robot_command(self):
        if self._robot_interface is not None:
            self._robot_interface.init_write_robot_command()
        else:
            for j in self._robot.joint_names:
                pub = rospy.Publisher(j + "_position_controller/command", Float64, queue_size=10)
                self._joint_command_publishers.append(pub)
    
    def _ft_wrench_cb(self, ft_wrench):
        #TODO: Add reading timeout?
        with self._lock:
            t = ft_wrench.wrench.torque
            f = ft_wrench.wrench.force
            self._ft_wrench = np.array([t.x, t.y, t.z, f.x, f.y, f.z])
    
    def _init_read_ft_sensor(self):
        if self._ft_sensor_interface is not None:
            self._ft_sensor_interface.init_read_ft_sensor()
        else:
            self._ft_wrench_subscriber = rospy.Subscriber("ft_wrench", WrenchStamped, self._ft_wrench_cb)
    
    def _external_setpoint_cb(self, i, joint_setpoint):
        if joint_setpoint.name == self._robot.joint_names and len(joint_setpoint.position) == len(self._robot.joint_names):
            with self._lock:
                self._external_setpoint[i] = joint_setpoint.position
    
    def _init_read_external_setpoints(self):
        self._external_setpoint_subscribers=[]
        for i in xrange(8):
            sub = rospy.Subscriber("external_setpoint_%d" % i, safe_kinematic_controller_msg.JointSetpoint, lambda msg,i=i: self._external_setpoint_cb(i, msg))
        
    def _init_publish_state(self):
        self._publish_state_publisher = rospy.Publisher("controller_state", safe_kinematic_controller_msg.ControllerState, queue_size=10)
        
    def read_robot_joint_position(self):
        if self._robot_interface is not None:
            return self._robot_interface.read_robot_joint_position()
        p = np.copy(self._joint_states_position)
        if any(np.isnan(p)):
            return ControllerMode.ROBOT_COMMUNICATION_ERROR, None
        return ControllerMode.SUCCESS, p        
    
    def write_robot_command(self, setpoint):
        if self._robot_interface is not None:
            return self._robot_interface.write_robot_command(setpoint)
        else:
            for i in xrange(len(self._joint_command_publishers)):
                self._joint_command_publishers[i].publish(Float64(setpoint[i]))
            return ControllerMode.SUCCESS
    
    def write_robot_no_command(self):
        if self._robot_interface is not None:
            return self._robot_interface.write_robot_no_command()
        else:
            return ControllerMode.SUCCESS
        
    
    def read_joystick_command(self):
        return self._joystick_interface.read_joystick_command()
    
    def read_ft_wrench(self):
        if self._ft_sensor_interface is not None:
            return self._ft_sensor_interface.read_ft_wrench()
        if self._ft_wrench is None:
            return ControllerMode.SENSOR_COMMUNICATION_ERROR, None
        return ControllerMode.SUCCESS, self._ft_wrench
    
    def read_external_setpoint(self, index):
        setpoint = self._external_setpoint[index]
        if setpoint is None:
            return ControllerMode.INVALID_EXTERNAL_SETPOINT, None
        return ControllerMode.SUCCESS, setpoint
    
    def _publish_state(self, state):
        
        if self._publish_state_publisher is not None:
            s = self._fill_controller_state_msg(state)
            self._publish_state_publisher.publish(s)
    
    def _fill_controller_state_msg(self, state):
        
        def vector2wrench(v):
            if v is None:
                return Wrench(Vector3(0,0,0), Vector3(0,0,0))
            return Wrench(Vector3(v[3], v[4], v[5]), Vector3(v[0], v[1], v[2]))
                
        s = safe_kinematic_controller_msg.ControllerState()
        s.header.stamp = rospy.Time.now()
        s.mode.mode = state.mode.value
        s.joint_name = self._robot.joint_names
        s.joint_position = list(state.joint_position) if state.joint_position is not None else []
        s.joint_setpoint_position = list(state.joint_setpoint) if state.joint_setpoint is not None else []
        s.joint_command_position = list(state.joint_command) if state.joint_command is not None else []
        s.ft_wrench = vector2wrench(state.ft_wrench)
        s.ft_wrench_valid = state.ft_wrench_status.value > 0
        s.ft_wrench_bias = vector2wrench(state.ft_wrench_bias)
        if state.active_trajectory is not None:
            s.trajectory_valid = True
            s.trajectory_time = state.active_trajectory.trajectory_time
            s.trajectory_max_time = state.active_trajectory.trajectory_max_time
        return s
    
    def _get_controller_state_cb(self, req):
        with self._lock:
            res = safe_kinematic_controller_srv.GetControllerStateResponse()
            res.state = self._fill_controller_state_msg(self._state)
            res.error_code.mode=ControllerMode.SUCCESS.value
            return res
      
    def _set_controller_mode_cb(self, req):
        with self._lock:
            res = safe_kinematic_controller_srv.SetControllerModeResponse()
            if req.speed_scalar < 0 or req.speed_scalar > 5:
                res.error_code.mode = ControllerMode.INVALID_ARGUMENT.value
                return res
            
            ft_threshold = []
            ft_bias = []
            if len(req.ft_stop_threshold) != 0:
                ft_threshold = np.array(req.ft_stop_threshold)
                if (len(ft_threshold) != 6 or np.any(ft_threshold < 0)):
                    res.error_code.mode = ControllerMode.INVALID_ARGUMENT.value
                    return res
            
            if len(req.ft_bias) != 0:
                ft_bias = np.array(req.ft_bias)
                if (len(req.ft_bias)) != 6:
                    res.error_code.mode = ControllerMode.INVALID_ARGUMENT.value
                    return res
            
            ret = self._set_mode(ControllerMode(req.mode.mode))
            if (ret.value < 0):
                res.error_code.mode=ret.value
                return res
            
            ret = self._set_speed_scalar(req.speed_scalar)
            if (ret.value < 0):
                res.error_code.mode=ret.value
                return res
            ret = self._set_ft_wrench_threshold(ft_threshold)
            if (ret.value < 0):
                res.error_code.mode=ret.value
                return res            
            ret = self._set_ft_wrench_bias(ft_bias)
            if (ret.value < 0):
                res.error_code.mode=ret.value
                return res
            
            res.error_code.mode=ControllerMode.SUCCESS.value
                        
            return res
        
    
class ROSJointTrajectoryListener():
    def __init__(self, gh):
        self._lock = threading.Lock()
        self.gh = gh
        
    def aborted(self, trajectory, trajectory_time, error_code, error_string):
        with self._lock:
            if self.gh is None:
                return
            self.gh.set_aborted(text=error_string)
            self.gh = None
    
    def completed(self, trajectory, trajectory_time):
        with self._lock:
            if self.gh is None:
                return
            self.gh.set_succeeded()
            self.gh=None
        
    def setpoint_updated(self, trajectory, trajectory_time, joint_command, controller_state):
        pass
        
        

class ROSJointTrajectoryManager(JointTrajectoryManager):
    
    def __init__(self, robot, trajectory_type=JointTrajectory, default_tolerance = 0.001):
        super(ROSJointTrajectoryManager,self).__init__(robot, trajectory_type)

        assert trajectory_type is not None

        self._action = action_server.ActionServer("joint_trajectory_action", FollowJointTrajectoryAction, self.goal_cb, self.cancel_cb, auto_start=False)        
        self._action.start()

        #self._lock = self._action.lock
        self._default_tolerance = default_tolerance
    
        self._gh_trajectory = WeakValueDictionary()
        self._trajectory_type = trajectory_type
        
    def goal_cb(self, gh):
        
        g = gh.get_goal()
        
        if (g.trajectory.joint_names != self._robot.joint_names):
            gh.set_rejected()
            rospy.logerr("Invalid joint names provided by action client")
            return
                        
        n_joints = len(self._robot.joint_names)
        
        last_t = 0
        waypoints = []
        for p in g.trajectory.points:
            
            time_from_start = p.time_from_start.to_sec()
            
            if (time_from_start < last_t):
                gh.set_rejected()
                rospy.logerr("Invalid duration_from_start in trajectory")
                return
            
            last_t = time_from_start
            
            positions = p.positions
            if np.shape(positions) != (n_joints,):
                gh.set_rejected()
                rospy.logerr("Invalid joint position in trajectory")
                return
            
            velocities = p.velocities
            if np.size(velocities) != 0 and np.shape(velocities) != (n_joints,):
                gh.set_rejected()
                rospy.logerr("Invalid joint velocity in trajectory")
                return
                
            accelerations = p.accelerations
            if np.size(accelerations) != 0 and np.shape(accelerations) != (n_joints,):
                gh.set_rejected()
                rospy.logerr("Invalid joint acceleration in trajectory")
                return
            
            effort = p.effort
            if np.size(effort) != 0 and np.shape(effort) != (n_joints,):
                gh.set_rejected()
                rospy.logerr("Invalid joint effort in trajectory")
                return
            
            
            
            w = JointTrajectoryWaypoint(time_from_start, positions, velocities, accelerations, effort)
            
            waypoints.append(w)
        
        def fill_default_tolerance():
            return JointTrajectoryTolerance(
                np.array([self._default_tolerance] * n_joints),
                np.array([self._default_tolerance] * n_joints),
                np.array([self._default_tolerance] * n_joints)
                )
        
        def convert_tolerance(ros_tolerance):
            if (len(ros_tolerance) == 0):
                return fill_default_tolerance()            
            if (len(ros_tolerance) != n_joints):
                rospy.logerr("Invalid joint tolerance in trajectory")
                return
            
            positions = np.zeros((n_joints,))
            velocities = np.zeros((n_joints,))
            accelerations = np.zeros((n_joints,))
            
            for i in n_joints:
                if ros_tolerance[i].name != self._robot.joint_names[i]:
                    rospy.logerr("Invalid joint tolerance in trajectory")
                    return
                positions[i] = ros_tolerance[i].position
                velocities[i] = ros_tolerance[i].velocity 
                accelerations[i] = ros_tolerance[i].acceleration
            
            return JointTrajectoryTolerance(positions, velocities, accelerations)
                
                
        start_tolerance = fill_default_tolerance()
        path_tolerance = convert_tolerance(g.path_tolerance)
        goal_tolerance = convert_tolerance(g.goal_tolerance)
        goal_time_tolerance = g.goal_time_tolerance.to_sec()
        
        waypoint_trajectory =  JointTrajectoryWaypointPlan(waypoints, start_tolerance, path_tolerance,
                                                       goal_tolerance, goal_time_tolerance)
        
        gh.set_accepted()
        
        trajectory=self._trajectory_type(waypoint_trajectory, self._robot)
        trajectory_listener = ROSJointTrajectoryListener(gh)
        trajectory.add_listener(trajectory_listener)
        self._gh_trajectory[gh] = trajectory
        self.trajectory_received(trajectory)
                       
        rospy.loginfo("Trajectory received")
        
    def cancel_cb(self, gh):
        
        self.abort_all()
        
        try:
            trajectory = self._gh_trajectory[gh]
            trajectory.abort()
        except KeyError: pass

def main(robot_interface=None, ft_sensor_interface=None):
    
    if not rospy.core.is_initialized():
        rospy.init_node("safe_kinematic_controller")
    
    print rospy.get_param_names()
    
    robot_root_link = rospy.get_param("~robot_root_link", None)
    robot_tip_link = rospy.get_param("~robot_tip_link", None)
    
    rate_hz = rospy.get_param("~publish_frequency", 250.0)
    
    robot = rox_urdf.robot_from_parameter_server(root_link = robot_root_link, tip_link = robot_tip_link)
        
    controller = ROSController(robot, 1.0/rate_hz, robot_interface=robot_interface, ft_sensor_interface=ft_sensor_interface)
    
    rate = rospy.Rate(rate_hz)
    
    for _ in xrange(100):
        if rospy.is_shutdown():
            break        
        if controller.read_robot_joint_position()[0] == ControllerMode.SUCCESS:
            break        
        rate.sleep()
    
    while not rospy.is_shutdown():
        controller.step()
        rate.sleep()
        
