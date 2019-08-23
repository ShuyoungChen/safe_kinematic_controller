import general_robotics_toolbox as rox
import numpy as np
import threading
import quadprog
import rospy
from numpy.linalg import inv
from scipy.linalg import logm, norm, sqrtm
from pyquaternion import Quaternion
from tesseract_msgs.msg import ContactResultVector

##### 1
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import TwistStamped
from tf.transformations import *
from pyquaternion import Quaternion
import tf
import timeit

class QuadProg(object):
	def __init__(self, robot):
		self._robot = robot  
		self._robot_link = ['gripper']
		self._env_link = ['leeward_tip_panel']
		self._lock = threading.Lock()
		rospy.Subscriber("/tesseract_monitoring_contacts/contact_results", ContactResultVector, self.collision_callback)
		
		#### inequality constraints for quadprog ####
		self._h = np.zeros((15, 1))
		self._sigma = np.zeros((13, 1))
		self._dhdq = np.vstack((np.hstack((np.eye(6), np.zeros((6, 1)), np.zeros((6, 1)))), np.hstack((-np.eye(6), np.zeros((6, 1)), np.zeros((6, 1)))), np.zeros((1, 8))))

		###### parameters for solving QP #####
		self._c = 0.3 #0.5
		self._eta = 0.02
		self._E = 0.0005 #0.0005
		self._Ke = 1

		# optimization params
		self._er = 0.05
		self._ep = 0.05
		self._epsilon = 0.05
		
		# number of joints
		self._n = len(self._robot.joint_type)
		
		# if the closest point pairs have been found
		self.pt_found = False
		
		## whether qdot is solved by qp or by pesudo-inverse
		self.safe_mode = False
		
		## for fixed orientation equality constraint ##
		self._eq_orien = False
		
		## desired orientation of eef ##
		self._R_des = np.eye(3)
		
		self._dist = 100
		
		# spring constant of gripper
		self._K = 100000/2
		
		# contact force modeled by stiff spring
		self._F = 0
  		
  		# parameter for force control
  		self._kp = 0.000001
  		self._ki = 0.000000
  		self._kd = 0.000005
  		
  		self._fd = -1000  # only in Z in tool frame
  		
  		# force error integral
  		self._force_inte = 0
  		
  		# force error derivative
  		self._force_der = 0
  		self._F_new = 0
  		self._F_old = 0
  		self._tic = 0
  		self._toc = timeit.default_timer()
  		
  		# tf transform listener
  		self._listener = tf.TransformListener()
        
        
	#def ft_callback(self, data):
		#wrench = data.wrench
		#self._fz = wrench.force.z
		
	def collision_callback(self, data):
		contacts = data.contacts
		min_dist = 100
		
		for i in range(len(contacts)):
			link_name_1 = contacts[i].link_names[0]
			link_name_2 = contacts[i].link_names[1]
			if link_name_1 in self._robot_link and link_name_2 in self._robot_link:
				continue
			elif link_name_1 in self._env_link and link_name_2 in self._env_link:
				continue
			else:
				if link_name_1 in self._robot_link and link_name_2 in self._env_link:
					dist = contacts[i].distance
					if dist < min_dist:
						min_dist = dist			
				elif link_name_1 in self._env_link and link_name_2 in self._robot_link:
					dist = contacts[i].distance
					if dist < min_dist:
						min_dist = dist
					
   		if min_dist <= 0:
			self._F = self._K*min_dist
			self._dist = min_dist
		else:
			self._F = 0
			self._dist = min_dist
						
	def compute_quadprog(self, joint_position):
		self._tic = timeit.default_timer()
		
		with self._lock:
			print '-------force-------'
			print self._F
			print '-------distance between eef and panel-------'
			print self._dist
			
			self._F_new = self._F
			
			force_err = self._F-self._fd
			self._force_inte += force_err
			self._force_der = (self._F_new-self._F_old)/(self._tic-self._toc) # derivative part
			
			# P control to approach the target
			vz = -self._kp*force_err# - self._ki*self._force_inte-self._kd*self._force_der; # velocity in tool0 frame
			
			# change velocity in world frame
			try:
				# rot is a quaternion (x, y, z, w)
				(trans,rot) = self._listener.lookupTransform('/base_link', '/gripper', rospy.Time(0))
			except (LookupException, ConnectivityException, ExtrapolationException):
				pass
			
			# convert quaternion into (w, x, y, z)
			# convert from quaternion to rotation matrix (4 by 4)
			R_rot = quaternion_matrix([rot[3], rot[0], rot[1], rot[2]])
			
			R_rot = R_rot[0:3, 0:3]
			
			# if there is no contact, only move toward the target
			if self._F == 0:
				self._kp = 0.000004
				# 3 by 1 linear velocity in tool0 frame
				v_l = np.array([0, 0, vz])
			# if there is contact, also move in x direction in sensor frame
			else:
				# increase Kp gain when in contact with the target
				self._kp = 0.000007
				# PID control
				vz = -self._kp*force_err - self._ki*self._force_inte-self._kd*self._force_der;
				# 3 by 1 linear velocity in tool0 frame
				v_l = np.array([-0.003, 0, vz])
			
			# transform into base frame
			v_l = np.dot(R_rot, v_l)

			# make a complete 6 by 1 velocity command with zero angular velocity
			spatial_velocity_command = np.array([0,0,0,v_l[0],v_l[1],v_l[2]])
			
			
			####### for Rviz visualization ######
			##### 2
			br = tf.TransformBroadcaster()
			pub_v = rospy.Publisher('resolved_velocity', TwistStamped, queue_size=10)
			msg_v = TwistStamped()
		
			transform_0T = rox.fwdkin(self._robot, joint_position)
			#p = transform_0T.p
			#print p
		
			#q_orig1 = quaternion_from_euler(spatial_velocity_command[0], spatial_velocity_command[1], spatial_velocity_command[2])
		
			#q = quaternion_from_euler(-euler[0], -euler[1], -euler[2])
			# first element is w
			q_orig2 = Quaternion(matrix=transform_0T.R) 
			#print q_orig2
			# convert q_orig2 into the form that w is the last element
			# inverse the quaternion by inverting w
			q_orig3 = np.array([q_orig2[1], q_orig2[2], q_orig2[3], -q_orig2[0]])
		
			#q_orig = np.array([0, 1, 0, 0])
		
			#q_orig5 = quaternion_multiply(q_orig3, q_orig4)
		
			# create a frame for velocity
			br.sendTransform((0.0, 0.0, -0.2),q_orig3,rospy.Time.now(),"v","gripper")
			#br.sendTransform((0.0, 0.2, 0.0),(0,0,0,1),rospy.Time.now(),"v_a","v_d")
		
			msg_v.header.seq = 1
			msg_v.header.stamp = rospy.Time.now()
			msg_v.header.frame_id = "v"
		
			msg_v.twist.linear.x = v_l[0]# + p[0]
			msg_v.twist.linear.y = v_l[1]# + p[1]
			msg_v.twist.linear.z = v_l[2]# + p[2]
	  
			# the last one should be w
			#q_orig = quaternion_multiply(q_orig1, q_orig5)
			msg_v.twist.angular.x = 0 #q_orig[0]
			msg_v.twist.angular.y = 0 #q_orig[1]
			msg_v.twist.angular.z = 0 #q_orig[2]
		
			pub_v.publish(msg_v)
			##############################################
			##############################################

			# compute joint velocity by QP
			if self.safe_mode == True:
				joints_vel = self.compute_joint_vel_cmd_qp(joint_position, spatial_velocity_command)
				joints_vel = np.clip(joints_vel, -self._robot.joint_vel_limit, self._robot.joint_vel_limit)
			# compute joint velocity by jacobian inverse
			else:	
				joints_vel = self.compute_joint_vel_cmd_jacobian_inverse(joint_position, spatial_velocity_command)
				joints_vel = np.clip(joints_vel, -self._robot.joint_vel_limit, self._robot.joint_vel_limit)
				
			self._F_old = self._F_new
			self._toc = self._tic
			return joints_vel 
		
	def compute_joint_vel_cmd_jacobian_inverse(self, joint_position, spatial_velocity_command):
		J = rox.robotjacobian(self._robot, joint_position)
		joints_vel = np.linalg.pinv(J).dot(spatial_velocity_command)
		
		return joints_vel
				
	def compute_joint_vel_cmd_qp(self, joint_position, spatial_velocity_command):		
		pp, RR = self.fwdkin_alljoints(joint_position)

		## desired eef orientation ##
		if self._eq_orien == True:
			self._R_des = RR[:,:,-1]
			self._eq_orien = False
		
		## if the self-collision has higher priority than collision with environment
		if self._self_min_dist <= self._min_dist:
			Closest_Pt = self._self_Closest_Pt_1
			Closest_Pt_env = self._self_Closest_Pt_2
		else:
			Closest_Pt = self._Closest_Pt
			Closest_Pt_env = self._Closest_Pt_env
		
		#Closest_Pt = self._Closest_Pt
		#Closest_Pt_env = self._Closest_Pt_env
	
		# where is the closest joint to the closest point
		J2C_Joint = self.Joint2Collision(Closest_Pt, pp)
			
		# jacobian of end-effector
		J_eef = rox.robotjacobian(self._robot, joint_position)

		v_tmp = Closest_Pt - pp[:, [-1]]
	
		v_tmp2 = (pp[:, [-1]] - pp[:, [-3]]) 
		p_norm2 = norm(v_tmp2)
		v_tmp2 = v_tmp2/p_norm2
		
		# desired rotational velocity
		vr = spatial_velocity_command[0:3]
		vr = vr.reshape(3, 1)
		
		# desired linear velocity
		vp = spatial_velocity_command[3:None]
		vp = vp.reshape(3, 1)
		
		J = self.getJacobian3(joint_position, Closest_Pt)    
		##### change 6 #####
		#J, _ = self.getJacobian2(joint_position, Closest_Pt, 6)
		
		### change J to J_eef ###
		Q = self.getqp_H(J_eef, vr, vp)         
	
		# make sure Q is symmetric
		Q = 0.5*(Q + Q.T)
		
		f = self.getqp_f()
		f = f.reshape((8, ))

		### change the velocity scale ###
		LB = np.vstack((-self._robot.joint_vel_limit.reshape(6, 1), 0, 0))
		UB = np.vstack((self._robot.joint_vel_limit.reshape(6, 1), 1, 1))
	
		# inequality constrains A and b
		self._h[0:6] = joint_position.reshape(6, 1) - self._robot.joint_lower_limit.reshape(6, 1)
		self._h[6:12] = self._robot.joint_upper_limit.reshape(6, 1) - joint_position.reshape(6, 1)
	
		dx = Closest_Pt_env[0] - Closest_Pt[0]
		dy = Closest_Pt_env[1] - Closest_Pt[1]
		dz = Closest_Pt_env[2] - Closest_Pt[2]
	
		dist = np.sqrt(dx**2 + dy**2 + dz**2)
		dist = norm(Closest_Pt-Closest_Pt_env)
	
		# derivative of dist w.r.t time
		der = np.array([dx*(dx**2 + dy**2 + dz**2)**(-0.5), dy*(dx**2 + dy**2 + dz**2)**(-0.5), dz*(dx**2 + dy**2 + dz**2)**(-0.5)])
		
		#print dist
		dmin = 0.03#0.115
		self._h[12] = dist - dmin
		#self._h[12] = 0.5*(dist*dist - dmin*dmin)
		
		## change here ##
		#self._dhdq[12, 0:6] = np.dot(-der.T, J[3:6,:])
		self._dhdq[12, 0:6] = np.dot(-dist*der.T, J[3:6,:])
		#self._dhdq[12, 0:6] = np.dot(-Closest_Pt_env.T+Closest_Pt.T, J[3:6,:])
			
		self._sigma[0:12] = self.inequality_bound(self._h[0:12])
		self._sigma[12] = self.inequality_bound(self._h[12])  
		#print self._h[12]
		#print self._sigma[12]
		
		#A = self._dhdq
		#b = self._sigma
	
		A = np.vstack((self._dhdq, np.eye(8), -np.eye(8)))
		b = np.vstack((self._sigma, LB, -UB))
		b = b.reshape((29, ))
		
		# equality constraints for maintaining end-effector orientation (pure translation)
		#A_eq = np.hstack((J_eef[0:3,:], np.zeros((3, 2))))            
		#w_skew = logm(np.dot(RR[:,:,-1], self._R_des.T))
		#w = np.array([w_skew[2, 1], w_skew[0, 2], w_skew[1, 0]])
		######### change -0.05 ##########
		#b_eq = -0.001*self._Ke*w
	
		# stack equality constrains on top of the inequality constraints
		#A = np.vstack((A_eq, A))
		#b = np.concatenate((b_eq.reshape((3, 1)), b.reshape((29, 1))), axis=0)
		#b = b.reshape((32, ))

		# the last argument specify the number of equality constraints
		
		#sc = norm(Q,'fro')

		#dq_sln = quadprog.solve_qp(Q/sc, -f/sc, A.T, b, A_eq.shape[0])[0]
	
		#A = np.delete(A, [0, 1, 2], axis=0)
		#b = np.delete(b, [0, 1, 2])
		
		# solve the quadprog problem
		## scale the matrix to avoid numerical errors of solver
		sc = norm(Q,'fro')
		dq_sln = quadprog.solve_qp(Q/sc, -f/sc, A.T, b)[0]
		#dq_sln = quadprog.solve_qp(Q, -f, A.T, b)[0]
			
		if len(dq_sln) < self._n:
			qdot = np.zeros((self._n,1))
			V_scaled = 0
			print 'No Solution'
		else:
			qdot = dq_sln[0: self._n]
			V_scaled = dq_sln[-1]*vp
			#vr_scaled = dq_sln[-2]*vr.reshape(3,1)
	
		V_linear = np.dot(J_eef[3:6,:], qdot)
		V_rot = np.dot(J_eef[0:3,:], qdot)
		
		qdot = qdot.reshape((6, ))
		
		#print 'desired angular velocity'
		print vr
		#print 'actual angular velocity'
		print V_rot
		#print 'desired linear velocity'
		print vp
		#print 'actual linear velocity'
		print V_linear
	
		#print 'solved joint velocity'
		#print qdot
		
		####### for Rviz visualization ######
		##### 2
		br = tf.TransformBroadcaster()
		pub_d = rospy.Publisher('desired_velocity', TwistStamped, queue_size=10)
		pub_a = rospy.Publisher('actual_velocity', TwistStamped, queue_size=10)
		msg_d = TwistStamped()
		msg_a = TwistStamped()
		
		transform_0T = rox.fwdkin(self._robot, joint_position)
		#p = transform_0T.p
		#print p
		
		#q_orig1 = quaternion_from_euler(spatial_velocity_command[0], spatial_velocity_command[1], spatial_velocity_command[2])
		
		#q = quaternion_from_euler(-euler[0], -euler[1], -euler[2])
		# first element is w
		q_orig2 = Quaternion(matrix=transform_0T.R) 
		#print q_orig2
		# convert q_orig2 into the form that w is the last element
		# inverse the quaternion by inverting w
		q_orig3 = np.array([q_orig2[1], q_orig2[2], q_orig2[3], -q_orig2[0]])
		
		#q_orig = np.array([0, 1, 0, 0])
		
		#q_orig5 = quaternion_multiply(q_orig3, q_orig4)
		
		# create a frame for velocity
		br.sendTransform((0.0, 0.0, -0.2),q_orig3,rospy.Time.now(),"v_d","vacuum_gripper_tool")
		br.sendTransform((0.0, 0.2, 0.0),(0,0,0,1),rospy.Time.now(),"v_a","v_d")
		
		msg_d.header.seq = 1
		msg_d.header.stamp = rospy.Time.now()
		msg_d.header.frame_id = "v_d"
		
		msg_d.twist.linear.x = vp[0]# + p[0]
		msg_d.twist.linear.y = vp[1]# + p[1]
		msg_d.twist.linear.z = vp[2]# + p[2]
  
		# the last one should be w
		#q_orig = quaternion_multiply(q_orig1, q_orig5)
		msg_d.twist.angular.x = vr[0] #q_orig[0]
		msg_d.twist.angular.y = vr[1] #q_orig[1]
		msg_d.twist.angular.z = vr[2] #q_orig[2]
		
		msg_a.header.seq = 1
		msg_a.header.stamp = rospy.Time.now()
		msg_a.header.frame_id = "v_a"
		
		msg_a.twist.linear.x = V_linear[0]# + p[0]
		msg_a.twist.linear.y = V_linear[1]# + p[1]
		msg_a.twist.linear.z = V_linear[2]# + p[2]
  
		# the last one should be w
		#q_orig = quaternion_multiply(q_orig1, q_orig5)
		msg_a.twist.angular.x = V_rot[0] #q_orig[0]
		msg_a.twist.angular.y = V_rot[1] #q_orig[1]
		msg_a.twist.angular.z = V_rot[2] #q_orig[2]
		#msg.pose.orientation.w = q_orig[3]
			            
		pub_d.publish(msg_d)
		pub_a.publish(msg_a)
		###################
			           
		return qdot
			
	# for inequality constraint        
	def inequality_bound(self, h):
		sigma = np.zeros((h.shape))
		h2 = h - self._eta
		sigma[np.array(h2 >= self._epsilon)] = -np.tan(self._c*np.pi/2)
		sigma[np.array(h2 >= 0) & np.array(h2 < self._epsilon)] = -np.tan(self._c*np.pi/2/self._epsilon*h2[np.array(h2 >= 0) & np.array(h2 < self._epsilon)])
		sigma[np.array(h >= 0) & np.array(h2 < 0)] = -self._E*h2[np.array(h >= 0) & np.array(h2 < 0)]/self._eta
		sigma[np.array(h < 0)] = self._E

		return sigma
			  
	# get the matrix f for solving QP       
	def getqp_f(self):
		f = -2*np.array([0, 0, 0, 0, 0, 0, self._er, self._ep]).reshape(8, 1)

		return f
	 
	# get the matrix H for solving QP        
	def getqp_H(self, J, vr, vp):
		H1 = np.dot(np.hstack((J,np.zeros((6,2)))).T,np.hstack((J,np.zeros((6,2)))))

		tmp = np.vstack((np.hstack((np.hstack((np.zeros((3, self._n)),vr)),np.zeros((3,1)))),np.hstack((np.hstack((np.zeros((3,self._n)),np.zeros((3,1)))),vp)))) 
		H2 = np.dot(tmp.T,tmp)

		H3 = -2*np.dot(np.hstack((J,np.zeros((6,2)))).T, tmp)
		H3 = (H3+H3.T)/2;

		tmp2 = np.vstack((np.array([0,0,0,0,0,0,np.sqrt(self._er),0]),np.array([0,0,0,0,0,0,0,np.sqrt(self._ep)])))
		H4 = np.dot(tmp2.T, tmp2)

		H = 2*(H1+H2+H3+H4)

		return H
			
	# return jacobian of the closest point on robot        
	def getJacobian2(self, joint_position, Closest_Pt, J2C_Joint):
		P_0_i,R_0_i = self.fwdkin_alljoints(joint_position)
		P_0_T = P_0_i[:, self._n]

		J = np.zeros((6, self._n))
		
		for i in range(self._n):
			if self._robot.joint_type[i] == 0:
			    J[:, i] = np.hstack((np.dot(R_0_i[:,:,i], self._robot.H[:,i]), np.dot(self.hat(np.dot(R_0_i[:,:,i], self._robot.H[:,i])), (P_0_T - P_0_i[:,i]))))

		J[:, J2C_Joint:7] = 0
		link_c = P_0_i[:, J2C_Joint] - P_0_i[:, J2C_Joint-1]
		link_c = link_c/norm(link_c)

		P_0_tmp = P_0_i[:, J2C_Joint-1]+ abs(np.dot(Closest_Pt - P_0_i[:, J2C_Joint-1], link_c))*link_c

		return J, P_0_tmp

	# return jacobian of the closest point on panel  
	def getJacobian3(self, joint_position, Closest_Pt):
		P_0_i, R_0_i = self.fwdkin_alljoints(joint_position)

		P_0_T = Closest_Pt
		P_0_T = P_0_T.reshape((3, ))

		J = np.zeros((6, self._n))

		for i in range(self._n):
			if self._robot.joint_type[i] == 0:
			    J[:,i] = np.hstack((np.dot(R_0_i[:,:,i], self._robot.H[:,i]), np.dot(self.hat(np.dot(R_0_i[:,:,i], self._robot.H[:,i])), (P_0_T - P_0_i[:,i]))))

		return J

	# determine the closest joint to Closest_pt  
	def Joint2Collision(self, Closest_Pt, pp):
		link_dist = []

		for i in range(self._n-1):
			link = pp[:,i+1]-pp[:,i]
			link = link/norm(link)
			pp2c = Closest_Pt - pp[:,i]
	
			link_dist.append(norm(pp2c - abs(np.dot(pp2c, link))*link))

		J2C_Joint = link_dist.index(min(link_dist)) + 1
	
		if(J2C_Joint==1):
			J2C_Joint=2
	
		return J2C_Joint

	# find closest rotation matrix 
	# A=A*inv(sqrt(A'*A))   
	def Closest_Rotation(self, R):
		R_n = np.dot(R, inv(sqrtm(np.dot(R.T, R))))
	
		return R_n

	# ROT Rotate along an axis h by q in radius
	def rot(self, h, q):
		h=h/norm(h)
		R = np.eye(3) + np.sin(q)*self.hat(h) + (1 - np.cos(q))*np.dot(self.hat(h), self.hat(h))
	
		return R
		
	def hat(self, h):
		h_hat = np.array([[0, -h[2], h[1]], [h[2], 0, -h[0]], [-h[1], h[0], 0]])
	
		return h_hat

	def fwdkin_alljoints(self, joint_position):
		R=np.eye(3)
		p=np.zeros((3,1))
		
		if self._robot.R_tool is not None and self._robot.p_tool is not None:
			RR = np.zeros((3,3,self._n+1))
			pp = np.zeros((3,self._n+1))
		else:
			RR = np.zeros((3,3,self._n))
			pp = np.zeros((3,self._n))
			
    	    #p = p + R.dot(robot.p_tool)
    	    #R = R.dot(robot.R_tool) 
              
		#RR = np.zeros((3,3,self._n+1))
		#pp = np.zeros((3,self._n+1))

		for i in range(self._n):
			h_i = self._robot.H[:,i]
	   
			if self._robot.joint_type[i] == 0:
			#rev
			    pi = self._robot.P[:,i].reshape(3, 1)
			    p = p+np.dot(R,pi)
			    Ri = self.rot(h_i,joint_position[i])
			    R = np.dot(R,Ri)
			    R = self.Closest_Rotation(R)
			elif self._robot.joint_type[i] == 1: 
			#pris
			    pi = (self._robot.P[:,i]+joint_position[i]*h_i).reshape(3, 1)
			    p = p+np.dot(R,pi)
			else: 
			# default pris
				pi = (self._robot.P[:,i]+joint_position[i]*h_i).reshape(3, 1)
				p = p+np.dot(R,pi)
	
			pp[:,[i]] = p
			RR[:,:,i] = R

		if self._robot.R_tool is not None and self._robot.p_tool is not None:
			p = p + R.dot(self._robot.p_tool.reshape(3,1))
			R = R.dot(self._robot.R_tool)
			pp[:,[self._n]] = p
			RR[:,:,self._n] = R
	
	
		# end effector T
		#p=p+np.dot(R, self._robot.P[:,self._n].reshape(3, 1))
		#pp[:,[self._n]] = p
		#RR[:,:,self._n] = R

		return pp, RR 
	
