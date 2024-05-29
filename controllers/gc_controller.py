import numpy as np
from scipy.spatial.transform import Rotation as R
from DATT.quadsim.control import Controller
from DATT.quadsim.models import RBModel
from DATT.controllers.cntrl_config import GCConfig
from DATT.quadsim.rigid_body import State_struct
from DATT.configuration.configuration import AllConfig


# new import
import sympy as sym


class GCController(Controller):
  def __init__(self, config : AllConfig, cntrl_config : GCConfig):
    super().__init__()
    self.gc_config = cntrl_config
    self.config = config

    self.pos_err_int = np.zeros(3)
    self.v_prev = np.zeros(3)
    self.prev_t = None
    self.start_pos = np.zeros(3)
    self.mass = config.drone_config.sampler.sample_param(config.drone_config.assumed_mass)
    self.I = np.eye(3) * config.drone_config.sampler.sample_param(config.drone_config.assumed_I)

    
    self.gravity_vector = np.array([0, 0, 9.81]).reshape((3,1))
    # self.gravity_vector = np.array([0,0,0]).reshape((3,1))
    self.psi = sym.Symbol('psi')
    self.s1, self.s2, self.s3 = sym.symbols("s1:4")
    self.H1_sym = sym.Matrix([[sym.cos(self.psi), -sym.sin(self.psi), 0], [sym.sin(self.psi), sym.cos(self.psi), 0], [0, 0, 1]])
    self.H2_sym = sym.Matrix([[1-((self.s1**2)/(1 + self.s3)), -1*(self.s1*self.s2)/(1 + self.s3), self.s1], [-1*(self.s1*self.s2)/(1 + self.s3), 1-((self.s2**2)/(1 + self.s3)), self.s2], [-self.s1, -self.s2, self.s3]])
    self.H2_inv_sym = self.H2_sym.inv()
    
    # Lambda function
    self.H1 = sym.lambdify([self.psi], self.H1_sym, "numpy")
    self.H2 = sym.lambdify([(self.s1, self.s2, self.s3)], self.H2_sym, "numpy")
    self.H2_inv = sym.lambdify([(self.s1, self.s2, self.s3)], self.H2_inv_sym, "numpy")
    
    self.ctbr = self.gc_config.CTBR



  def hat_map(self, vec):
    # vec = vec.reshape((3,))
    return np.array([[0, -1*vec[2], vec[1]], [vec[2], 0, -1*vec[0]], [-1*vec[1], vec[0], 0]])
    
  def vee_map(self, arr):
    return np.array([arr[2,1], arr[0,2], arr[1,0]])
    
  def response(self, **response_inputs ):
    t = response_inputs.get('t')
    state : State_struct = response_inputs.get('state')
    # ref_dict : dict = response_inputs.get('ref')

    ref_state = self.ref_func.get_state_struct(t)
    
    # print('offset : ', self.ref_func.offset_pos)

    if self.prev_t != None:
      dt = t - self.prev_t
    else:
      dt = self.config.sim_config.dt()

    # gc
    pos = state.pos - self.start_pos
    vel = state.vel
    rot = state.rot
    p_err = pos - ref_state.pos
    v_err = vel - ref_state.vel

   # Updating error for integral term.
    self.pos_err_int += p_err * dt


    # need to check signs of the addition/subtraction here. Control gains are positive
    if self.ctbr:
        x_ddot_des = -self.gc_config.kp_ctbr * p_err + -self.gc_config.kd_ctbr * v_err + ref_state.acc # no I term here
    else:
        x_ddot_des = -self.gc_config.kp_srt * p_err - self.gc_config.kd_srt * v_err + ref_state.acc
    x_ddot_des = x_ddot_des.reshape((3,1))
    
    

    r_real = state.rot.as_matrix()
    r_reference = ref_state.rot.as_matrix()
    s_des = (x_ddot_des + self.gravity_vector) / np.linalg.norm(x_ddot_des + self.gravity_vector)
    s_ref = (ref_state.acc.reshape((3,1)) + self.gravity_vector) / np.linalg.norm(ref_state.acc.reshape((3,1)) + self.gravity_vector)

    s_des = s_des.squeeze()
    s_ref = s_ref.squeeze()

    h1_psi_ref = self.H2_inv(s_ref) @ r_reference
    psi_ref = np.arctan2(h1_psi_ref[1,0], h1_psi_ref[0,0])
    omega_ref = np.linalg.inv(r_reference) @ ref_state.ang # convert to body frame
    r_des = self.H2(s_des) @ self.H1(psi_ref)

    omega_real = state.ang
    omega_real_hat = self.hat_map(omega_real)
    omega_des = np.linalg.inv(r_reference) @ ref_state.ang

    e_R = -0.5 * self.vee_map(r_des.T @ r_real - r_real.T @ r_des)
    e_omega = omega_real - r_real.T @ r_real @ omega_des

    if self.ctbr:
        f_des = (x_ddot_des + self.gravity_vector) # don't multiply mass (assume unity mass)
        thrust_command = (r_real.T @ f_des)[-1]

        # print("Reference part of omega: ", r_real.T @ r_des @ omega_ref)
        # print("Error part of omega: ", self.gc_config.kp_rot * e_R)
        omega_command  = self.gc_config.kp_ctbr * e_R + r_real.T @ r_des @ omega_ref

        return thrust_command, omega_command

    # Else, SRT inputs
    ang_acc_ref = np.zeros((3,1))
    omega_dot_des = self.gc_config.kr_srt * e_R - self.gc_config.kw_srt * e_omega + - (omega_real_hat @ r_real.T @ r_des @ omega_des).squeeze() + \
                    (r_real.T @ r_des @ ang_acc_ref).squeeze()
    
    omega_dot_des = r_real @ omega_dot_des
    
    torque = self.I @ omega_dot_des  + np.cross(state.ang, self.I @ state.ang)
    force = (x_ddot_des + self.gravity_vector) * self.mass
    body_z_force = (r_real.T @ force)[-1]
      
    return body_z_force, torque


  