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

import rospy
from safe_kinematic_controller.ros.controller import main as ros_main
from safe_kinematic_controller import ControllerMode
from rpi_abb_irc5 import EGM
from rpi_ati_net_ft import NET_FT
from sensor_msgs.msg import JointState 
from rpi_ati_net_ft import NET_FT

class EGMRobotInterface(object):
    def __init__(self, joint_names, egm_local_port = 6510, egm_recv_timeout = 0.0015 ):
        self._egm_recv_timeout = egm_recv_timeout
        self._egm = EGM(egm_local_port)        
        self._joint_names = joint_names
        self._joint_states_pub = rospy.Publisher("joint_states", JointState, queue_size = 10)
        self._joint_angles = None
        self._joint_angles_time = rospy.Time(0)
        
        self._egm_joint_state_pub = rospy.Publisher("abb_irc5_egm/joint_states", JointState, queue_size=10)
        self._egm_joint_command_pub = rospy.Publisher("abb_irc5_egm/joint_commands", JointState, queue_size=10)
    
    
    def init_read_robot_state(self):
        pass
    
    def init_write_robot_command(self):
        pass
        
    def read_robot_joint_position(self):
        res, egm_state = self._egm.receive_from_robot(timeout=self._egm_recv_timeout)
        if not res:            
            #Handle if there is stutter in the received data
            if ((rospy.Time.now() - self._joint_angles_time).to_sec() < 0.2):
                return ControllerMode.SUCCESS, self._joint_angles
            #print "Robot communication error: " + str((rospy.Time.now() - self._joint_angles_time).to_sec())
            return ControllerMode.ROBOT_COMMUNICATION_ERROR, None
        if not egm_state.rapid_running or not egm_state.motors_on:
            return ControllerMode.ROBOT_INVALID_STATE, None
        self._joint_angles=egm_state.joint_angles
        self._joint_angles_time = rospy.Time.now()
        self.publish_joint_states(egm_state.joint_angles)
        self._egm_joint_state_pub.publish(self._fill_joint_state_msg(egm_state.joint_angles))
        return ControllerMode.SUCCESS, egm_state.joint_angles
    
    def write_robot_command(self, setpoint):
        self._egm.send_to_robot(setpoint)        
        self._egm_joint_command_pub.publish(self._fill_joint_state_msg(setpoint))
        return ControllerMode.SUCCESS
    
    def write_robot_no_command(self):        
        self._egm.send_to_robot(None)
        return ControllerMode.SUCCESS
    
    def publish_joint_states(self, position):
        joint_states = JointState()
        joint_states.header.stamp = rospy.Time.now()
        joint_states.name = self._joint_names
        joint_states.position = position
        self._joint_states_pub.publish(joint_states)

    def _fill_joint_state_msg(self, joint_angles):
        js = JointState()        
        js.header.stamp = rospy.Time.now()
        js.name = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']      
        js.position = joint_angles  
        js.velocity = [0,0,0,0,0,0]
        js.effort = [0,0,0,0,0,0]
        return js

class FTSensorInterface(object):
   def __init__(self, netft_host):
       self._net_ft_host = netft_host

   def init_read_ft_sensor(self):
       self._net_ft = NET_FT(self._net_ft_host)
       self._net_ft.set_tare_from_ft()
       self._net_ft.start_streaming()

   def read_ft_wrench(self):
       ft_res, ft1, ft_status_code= self._net_ft.try_read_ft_streaming(0)
       if ft_res and ft_status_code == 0:
           return ControllerMode.SUCCESS, ft1
       else:
           return ControllerMode.SENSOR_FAULT, None
   

def main():
    rospy.init_node("safe_kinematic_controller_abb_irc5_egm")
    
    egm_local_port = rospy.get_param("~abb_egm_port", 6510)
    egm_recv_timeout = rospy.get_param("~abb_egm_recv_timeout", 0.0015)
    joint_names = rospy.get_param('controller_joint_names')
    robot_interface=EGMRobotInterface(joint_names, egm_local_port=egm_local_port, egm_recv_timeout = egm_recv_timeout)
    
    netft_host=rospy.get_param("~netft_host", None)    
    netft = None
    if netft_host is not None and len(netft_host) != 0:
        netft = FTSensorInterface(netft_host)        

    ros_main(robot_interface=robot_interface, ft_sensor_interface= netft)
    
