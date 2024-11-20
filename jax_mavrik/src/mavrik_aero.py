# mavrik_aero.py
import functools as ft


from jax_mavrik.mavrik_types import StateVariables, ControlInputs, AeroState, Forces, Moments
from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.src.actuator import ActuatorInutState, ActuatorInput, ActuatorOutput, actuate
from jax_mavrik.src.utils.mat_tools import euler_to_dcm
from jax_mavrik.src.utils.jax_types import FloatScalar

import jax.numpy as jnp
from jax import jit
from jax import vmap

from typing import Tuple, List 

@jit
def linear_interpolate(v0, v1, weight):
    return v0 * (1 - weight) + v1 * weight

@jit
def get_index_and_weight(value, breakpoints):
    """
    Finds the index and weight for interpolation along a single dimension.
    """
    idx = jnp.clip(jnp.searchsorted(breakpoints, value) - 1, 0, len(breakpoints) - 2)
    weight = (value - breakpoints[idx]) / (breakpoints[idx + 1] - breakpoints[idx])
    return idx, weight

@jit
def interpolate_nd(inputs: jnp.ndarray, breakpoints: List[jnp.ndarray], values: jnp.ndarray) -> float:
    """
    Perform n-dimensional interpolation using vectorized JAX operations.

    Args:
        inputs (jnp.ndarray): The input coordinates at which to interpolate.
        breakpoints (list of jnp.ndarray): Each array contains the breakpoints for one dimension.
        values (jnp.ndarray): The values at each grid point with shape matching the breakpoints.

    Returns:
        jnp.ndarray: Interpolated value.
    """
    ndim = len(breakpoints)
    indices = []
    weights = []

    # Loop over each dimension instead of using vmap
    for i in range(ndim):
        idx, weight = get_index_and_weight(inputs[i], breakpoints[i])
        indices.append(idx)
        weights.append(weight)

    indices = jnp.array(indices)
    weights = jnp.array(weights)

    # Generate corner indices for interpolation
    corner_indices = jnp.stack(jnp.meshgrid(*[jnp.array([0, 1]) for _ in range(ndim)], indexing="ij"), axis=-1).reshape(-1, ndim)

    # Function to compute interpolated values for each corner
    def compute_corner_value(corner):
        corner_idx = indices + corner
        corner_value = values[tuple(corner_idx)]
        corner_weight = jnp.prod(jnp.where(corner, weights, 1 - weights))
        return corner_value * corner_weight

    # Vectorize computation across all corners
    interpolated_values = vmap(compute_corner_value)(corner_indices)

    # Sum contributions from all corners
    return jnp.sum(interpolated_values)

class JaxNDInterpolator:
    def __init__(self, breakpoints: List[jnp.ndarray], values: jnp.ndarray):
        self.breakpoints = breakpoints
        self.values = values
 
    def __call__(self, inputs: jnp.ndarray) -> float:
        # Use partial to create a JAX-compatible function with fixed breakpoints and values
        interpolator = ft.partial(interpolate_nd, breakpoints=self.breakpoints, values=self.values)
        return interpolator(inputs)

class MavrikAero:
    def __init__(self, mavrik_setup: MavrikSetup):
        self.get_lookup_table(mavrik_setup)
 
    def __call__(self, state: StateVariables, control: ControlInputs) -> Tuple[Forces, Moments]:
        # Calculate forces and moments using Mavrik Aero model
        # Transform body frame velocities (u, v, w) to inertial frame velocities (Vx, Vy, Vz)
        R = euler_to_dcm(state.roll, state.pitch, state.yaw)
        

        # Body frame velocities
        body_velocities = jnp.array([state.VXe, state.VYe, state.VZe])
        #print(body_velocities)
        # Inertial frame velocities
        inertial_velocities = R @ body_velocities
        u, v, w = inertial_velocities
        #print(f"{inertial_velocities=} vs. {state.u=}, {state.v=}, {state.w=}")
        #print(f"beta_from_inertial_velocity={jnp.arctan2(v, jnp.sqrt(u**2 + w**2))} vs. beta_from_state_vb={jnp.arctan2(state.v, jnp.sqrt(state.u**2 + state.w**2))}")

        actuator_input_state = ActuatorInutState(
            U = jnp.sqrt(u**2 + v**2 + w**2),
            alpha = jnp.arctan2(w, u),
            beta = jnp.arctan2(v, jnp.sqrt(u**2 + w**2)),
            p = state.p,
            q = state.q,
            r = state.r
        )
         
        actuator_inputs = ActuatorInput(
            wing_tilt=control.wing_tilt, tail_tilt=control.tail_tilt, aileron=control.aileron,
            elevator=control.elevator, flap=control.flap, rudder=control.rudder,
            RPM_tailLeft=control.RPM_tailLeft, RPM_tailRight=control.RPM_tailRight,
            RPM_leftOut1=control.RPM_leftOut1, RPM_left2=control.RPM_left2,
            RPM_left3=control.RPM_left3, RPM_left4=control.RPM_left4,
            RPM_left5=control.RPM_left5, RPM_left6In=control.RPM_left6In,
            RPM_right7In=control.RPM_right7In, RPM_right8=control.RPM_right8,
            RPM_right9=control.RPM_right9, RPM_right10=control.RPM_right10,
            RPM_right11=control.RPM_right11, RPM_right12Out=control.RPM_right12Out
        )

        actuator_outputs: ActuatorOutput = actuate(actuator_input_state, actuator_inputs)
        #print(f"{actuator_outputs=}")
        wing_transform = jnp.array([[jnp.cos(actuator_outputs.wing_tilt), 0, jnp.sin(actuator_outputs.wing_tilt)], [0, 1, 0], [-jnp.sin(actuator_outputs.wing_tilt), 0., jnp.cos(actuator_outputs.wing_tilt)]]);
        tail_transform = jnp.array([[jnp.cos(actuator_outputs.tail_tilt), 0, jnp.sin(actuator_outputs.tail_tilt)], [0, 1, 0], [-jnp.sin(actuator_outputs.tail_tilt), 0., jnp.cos(actuator_outputs.tail_tilt)]])

    
        F0, M0 = self.Ct(actuator_outputs, wing_transform, tail_transform)
        '''
        for key, value in F0._asdict().items():
            if jnp.isnan(value).any():
                raise ValueError(f"NaN detected in actuator outputs {key=}: {value}")
            
        for key, value in M0._asdict().items():
            if jnp.isnan(value).any():
                raise ValueError(f"NaN detected in actuator outputs {key=}: {value}")
        '''
        F1 = self.Cx(actuator_outputs, wing_transform, tail_transform)
        '''
        for key, value in F1._asdict().items():
            if jnp.isnan(value).any():
                raise ValueError(f"NaN detected in actuator outputs {key=}: {value}")
        ''' 
        F2 = self.Cy(actuator_outputs, wing_transform, tail_transform)
        '''
        for key, value in F2._asdict().items():
            if jnp.isnan(value).any():
                raise ValueError(f"NaN detected in actuator outputs {key=}: {value}")
        '''
        F3 = self.Cz(actuator_outputs, wing_transform, tail_transform)
        '''
        for key, value in F3._asdict().items():
            if jnp.isnan(value).any():
                raise ValueError(f"NaN detected in actuator outputs {key=}: {value}")
        '''
        M1 = self.L(actuator_outputs, wing_transform, tail_transform)
        '''
        for key, value in M1._asdict().items():
            if jnp.isnan(value).any():
                raise ValueError(f"NaN detected in actuator outputs {key=}: {value}")
        '''
        M2 = self.M(actuator_outputs, wing_transform, tail_transform)
        '''
        for key, value in M2._asdict().items():
            if jnp.isnan(value).any():
                raise ValueError(f"NaN detected in actuator outputs {key=}: {value}")
        '''
        M3 = self.N(actuator_outputs, wing_transform, tail_transform)
        '''
        for key, value in M3._asdict().items():
            if jnp.isnan(value).any():
                raise ValueError(f"NaN detected in actuator outputs {key=}: {value}")
        '''
        M5 = self.Kq(actuator_outputs, wing_transform, tail_transform)
        '''
        for key, value in M5._asdict().items():
            if jnp.isnan(value).any():
                raise ValueError(f"NaN detected in actuator outputs {key=}: {value}")
        '''
        Fx = F0.Fx + F1.Fx + F2.Fx + F3.Fx
        Fy = F0.Fy + F1.Fy + F2.Fy + F3.Fy
        Fz = F0.Fz + F1.Fz + F2.Fz + F3.Fz

        forces = Forces(Fx, Fy, Fz)
        #moments_by_forces = jnp.cross(jnp.array([state.X, state.Y, state.Z]), jnp.array([forces.Fx, forces.Fy, forces.Fz]))
        
        moments = Moments(M0.L + M1.L + M2.L + M3.L + M5.L, # + moments_by_forces[0], 
                          M0.M + M1.M + M2.M + M3.M + M5.M, # + moments_by_forces[1], 
                          M0.N + M1.N + M2.N + M3.N + M5.N, # + moments_by_forces[2]
                          )

        return forces, moments, actuator_outputs
    


    def get_lookup_table(self, mavrik_setup: MavrikSetup):
        self.get_Cx_table(mavrik_setup)
        self.get_Cy_table(mavrik_setup)
        self.get_Cz_table(mavrik_setup)
        self.get_Cl_table(mavrik_setup)
        self.get_Cm_table(mavrik_setup)
        self.get_Cn_table(mavrik_setup)
        self.get_Kq_table(mavrik_setup)
        self.get_Ct_table(mavrik_setup)
        
     
    def get_Cx_table(self, mavrik_setup: MavrikSetup):
        CX_aileron_wing_breakpoints = [getattr(mavrik_setup, f'CX_aileron_wing_{i}') for i in range(1, 1 + 7)]
        CX_aileron_wing_value = mavrik_setup.CX_aileron_wing_val
        self.CX_aileron_wing_lookup_table = JaxNDInterpolator(CX_aileron_wing_breakpoints, CX_aileron_wing_value)

        CX_elevator_tail_breakpoints = [getattr(mavrik_setup, f'CX_elevator_tail_{i}') for i in range(1, 1 + 7)]
        CX_elevator_tail_value = mavrik_setup.CX_elevator_tail_val
        self.CX_elevator_tail_lookup_table = JaxNDInterpolator(CX_elevator_tail_breakpoints, CX_elevator_tail_value)

        CX_flap_wing_breakpoints = [getattr(mavrik_setup, f'CX_flap_wing_{i}') for i in range(1, 1 + 7)]
        CX_flap_wing_value = mavrik_setup.CX_flap_wing_val
        self.CX_flap_wing_lookup_table = JaxNDInterpolator(CX_flap_wing_breakpoints, CX_flap_wing_value)

        CX_rudder_tail_breakpoints = [getattr(mavrik_setup, f'CX_rudder_tail_{i}') for i in range(1, 1 + 7)]
        CX_rudder_tail_value = mavrik_setup.CX_rudder_tail_val
        self.CX_rudder_tail_lookup_table = JaxNDInterpolator(CX_rudder_tail_breakpoints, CX_rudder_tail_value)
        
        CX_tail_breakpoints = [getattr(mavrik_setup, f'CX_tail_{i}') for i in range(1, 1 + 6)]
        CX_tail_value = mavrik_setup.CX_tail_val
        self.CX_tail_lookup_table = JaxNDInterpolator(CX_tail_breakpoints, CX_tail_value)

        CX_tail_damp_p_breakpoints = [getattr(mavrik_setup, f'CX_tail_damp_p_{i}') for i in range(1, 1 + 6)]
        CX_tail_damp_p_value = mavrik_setup.CX_tail_damp_p_val
        self.CX_tail_damp_p_lookup_table = JaxNDInterpolator(CX_tail_damp_p_breakpoints, CX_tail_damp_p_value)

        CX_tail_damp_q_breakpoints = [getattr(mavrik_setup, f'CX_tail_damp_q_{i}') for i in range(1, 1 + 6)]
        CX_tail_damp_q_value = mavrik_setup.CX_tail_damp_q_val
        self.CX_tail_damp_q_lookup_table = JaxNDInterpolator(CX_tail_damp_q_breakpoints, CX_tail_damp_q_value)

        CX_tail_damp_r_breakpoints = [getattr(mavrik_setup, f'CX_tail_damp_r_{i}') for i in range(1, 1 + 6)]
        CX_tail_damp_r_value = mavrik_setup.CX_tail_damp_r_val
        self.CX_tail_damp_r_lookup_table = JaxNDInterpolator(CX_tail_damp_r_breakpoints, CX_tail_damp_r_value)

        CX_wing_breakpoints = [getattr(mavrik_setup, f'CX_wing_{i}') for i in range(1, 1 + 6)]
        CX_wing_value = mavrik_setup.CX_wing_val
        self.CX_wing_lookup_table = JaxNDInterpolator(CX_wing_breakpoints, CX_wing_value)
        
        CX_wing_damp_p_breakpoints = [getattr(mavrik_setup, f'CX_wing_damp_p_{i}') for i in range(1, 1 + 6)]
        CX_wing_damp_p_value = mavrik_setup.CX_wing_damp_p_val
        self.CX_wing_damp_p_lookup_table = JaxNDInterpolator(CX_wing_damp_p_breakpoints, CX_wing_damp_p_value)
        
        CX_wing_damp_q_breakpoints = [getattr(mavrik_setup, f'CX_wing_damp_q_{i}') for i in range(1, 1 + 6)]
        CX_wing_damp_q_value = mavrik_setup.CX_wing_damp_q_val
        self.CX_wing_damp_q_lookup_table = JaxNDInterpolator(CX_wing_damp_q_breakpoints, CX_wing_damp_q_value)
        
        CX_wing_damp_r_breakpoints = [getattr(mavrik_setup, f'CX_wing_damp_r_{i}') for i in range(1, 1 + 6)]
        CX_wing_damp_r_value = mavrik_setup.CX_wing_damp_r_val
        self.CX_wing_damp_r_lookup_table = JaxNDInterpolator(CX_wing_damp_r_breakpoints, CX_wing_damp_r_value)
        
        CX_hover_fuse_breakpoints = [getattr(mavrik_setup, f'CX_hover_fuse_{i}') for i in range(1, 1 + 3)]
        CX_hover_fuse_value = mavrik_setup.CX_hover_fuse_val
        self.CX_hover_fuse_lookup_table = JaxNDInterpolator(CX_hover_fuse_breakpoints, CX_hover_fuse_value)
        

    def Cx(self, u: ActuatorOutput, wing_transform: FloatScalar, tail_transform: FloatScalar) -> Forces:
        CX_Scale = 0.5744 * u.Q
        CX_Scale_r = 0.5744 * 2.8270 * 1.225 * 0.25 * u.U * u.r
        CX_Scale_p = 0.5744 * 2.8270 * 1.225 * 0.25 * u.U * u.p
        CX_Scale_q = 0.5744 * 0.2032 * 1.225 * 0.25 * u.U * u.q

        CX_aileron_wing = self.CX_aileron_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron
        ]))
        CX_aileron_wing_padded = jnp.array([CX_aileron_wing, 0.0, 0.0])
        CX_aileron_wing_padded_transformed = jnp.dot(wing_transform, CX_aileron_wing_padded * CX_Scale)
         
        CX_elevator_tail = self.CX_elevator_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator
        ]))
        CX_elevator_tail_padded = jnp.array([CX_elevator_tail, 0.0, 0.0])
        CX_elevator_tail_padded_transformed = jnp.dot(tail_transform, CX_elevator_tail_padded * CX_Scale)

        CX_flap_wing = self.CX_flap_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap
        ]))
        CX_flap_wing_padded = jnp.array([CX_flap_wing, 0.0, 0.0])
        CX_flap_wing_padded_transformed = jnp.dot(wing_transform, CX_flap_wing_padded * CX_Scale)

        CX_rudder_tail = self.CX_rudder_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder
        ]))
        CX_rudder_tail_padded = jnp.array([CX_rudder_tail, 0.0, 0.0])
        CX_rudder_tail_padded_transformed = jnp.dot(tail_transform, CX_rudder_tail_padded * CX_Scale)

        # Tail
        CX_tail = self.CX_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        CX_tail_padded = jnp.array([CX_tail, 0.0, 0.0])
        CX_tail_padded_transformed = jnp.dot(tail_transform, CX_tail_padded * CX_Scale)

        # Tail Damp p
        CX_tail_damp_p = self.CX_tail_damp_p_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ])) 
        CX_tail_damp_p_padded = jnp.array([CX_tail_damp_p, 0.0, 0.0])
        CX_tail_damp_p_padded_transformed = jnp.dot(tail_transform, CX_tail_damp_p_padded * CX_Scale_p)

        # Tail Damp q
        CX_tail_damp_q = self.CX_tail_damp_q_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        CX_tail_damp_q_padded = jnp.array([CX_tail_damp_q, 0.0, 0.0])
        CX_tail_damp_q_padded_transformed = jnp.dot(tail_transform, CX_tail_damp_q_padded * CX_Scale_q)

        # Tail Damp r
        CX_tail_damp_r = self.CX_tail_damp_r_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        CX_tail_damp_r_padded = jnp.array([CX_tail_damp_r, 0.0, 0.0])
        CX_tail_damp_r_padded_transformed = jnp.dot(tail_transform, CX_tail_damp_r_padded * CX_Scale_r)

        # Wing
        CX_wing = self.CX_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        CX_wing_padded = jnp.array([CX_wing, 0.0, 0.0])
        CX_wing_padded_transformed = jnp.dot(wing_transform, CX_wing_padded * CX_Scale)

        # Wing Damp p
        CX_wing_damp_p = self.CX_wing_damp_p_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        CX_wing_damp_p_padded = jnp.array([CX_wing_damp_p, 0.0, 0.0])
        CX_wing_damp_p_padded_transformed = jnp.dot(wing_transform, CX_wing_damp_p_padded * CX_Scale_p)

        # Wing Damp q
        CX_wing_damp_q = self.CX_wing_damp_q_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        CX_wing_damp_q_padded = jnp.array([CX_wing_damp_q, 0.0, 0.0])
        CX_wing_damp_q_padded_transformed = jnp.dot(wing_transform, CX_wing_damp_q_padded * CX_Scale_q)

        # Wing Damp r
        CX_wing_damp_r = self.CX_wing_damp_r_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        CX_wing_damp_r_padded = jnp.array([CX_wing_damp_r, 0.0, 0.0])
        CX_wing_damp_r_padded_transformed = jnp.dot(wing_transform, CX_wing_damp_r_padded * CX_Scale_r)

        # Hover Fuse
        CX_hover_fuse = self.CX_hover_fuse_lookup_table(jnp.array([
            u.U, u.alpha, u.beta
        ]))
        CX_hover_fuse_padded = jnp.array([CX_hover_fuse * CX_Scale, 0.0, 0.0])

        return Forces(
            CX_aileron_wing_padded_transformed[0] + CX_elevator_tail_padded_transformed[0] + CX_flap_wing_padded_transformed[0] + CX_rudder_tail_padded_transformed[0] +
            CX_tail_padded_transformed[0] + CX_tail_damp_p_padded_transformed[0] + CX_tail_damp_q_padded_transformed[0] + CX_tail_damp_r_padded_transformed[0] +
            CX_wing_padded_transformed[0] + CX_wing_damp_p_padded_transformed[0] + CX_wing_damp_q_padded_transformed[0] + CX_wing_damp_r_padded_transformed[0] +
            CX_hover_fuse_padded[0],
            CX_aileron_wing_padded_transformed[1] + CX_elevator_tail_padded_transformed[1] + CX_flap_wing_padded_transformed[1] + CX_rudder_tail_padded_transformed[1] +
            CX_tail_padded_transformed[1] + CX_tail_damp_p_padded_transformed[1] + CX_tail_damp_q_padded_transformed[1] + CX_tail_damp_r_padded_transformed[1] +
            CX_wing_padded_transformed[1] + CX_wing_damp_p_padded_transformed[1] + CX_wing_damp_q_padded_transformed[1] + CX_wing_damp_r_padded_transformed[1] +
            CX_hover_fuse_padded[1],
            CX_aileron_wing_padded_transformed[2] + CX_elevator_tail_padded_transformed[2] + CX_flap_wing_padded_transformed[2] + CX_rudder_tail_padded_transformed[2] +
            CX_tail_padded_transformed[2] + CX_tail_damp_p_padded_transformed[2] + CX_tail_damp_q_padded_transformed[2] + CX_tail_damp_r_padded_transformed[2] +
            CX_wing_padded_transformed[2] + CX_wing_damp_p_padded_transformed[2] + CX_wing_damp_q_padded_transformed[2] + CX_wing_damp_r_padded_transformed[2] +
            CX_hover_fuse_padded[2]
        )
    
    def get_Cy_table(self, mavrik_setup: MavrikSetup):
        CY_aileron_wing_breakpoints = [getattr(mavrik_setup, f'CY_aileron_wing_{i}') for i in range(1, 1 + 7)]
        CY_aileron_wing_value = mavrik_setup.CY_aileron_wing_val
        self.CY_aileron_wing_lookup_table = JaxNDInterpolator(CY_aileron_wing_breakpoints, CY_aileron_wing_value)

        CY_elevator_tail_breakpoints = [getattr(mavrik_setup, f'CY_elevator_tail_{i}') for i in range(1, 1 + 7)]
        CY_elevator_tail_value = mavrik_setup.CY_elevator_tail_val
        self.CY_elevator_tail_lookup_table = JaxNDInterpolator(CY_elevator_tail_breakpoints, CY_elevator_tail_value)

        CY_flap_wing_breakpoints = [getattr(mavrik_setup, f'CY_flap_wing_{i}') for i in range(1, 1 + 7)]
        CY_flap_wing_value = mavrik_setup.CY_flap_wing_val
        self.CY_flap_wing_lookup_table = JaxNDInterpolator(CY_flap_wing_breakpoints, CY_flap_wing_value)

        CY_rudder_tail_breakpoints = [getattr(mavrik_setup, f'CY_rudder_tail_{i}') for i in range(1, 1 + 7)]
        CY_rudder_tail_value = mavrik_setup.CY_rudder_tail_val
        self.CY_rudder_tail_lookup_table = JaxNDInterpolator(CY_rudder_tail_breakpoints, CY_rudder_tail_value)

        CY_tail_breakpoints = [getattr(mavrik_setup, f'CY_tail_{i}') for i in range(1, 1 + 6)]
        CY_tail_value = mavrik_setup.CY_tail_val
        self.CY_tail_lookup_table = JaxNDInterpolator(CY_tail_breakpoints, CY_tail_value)

        CY_tail_damp_p_breakpoints = [getattr(mavrik_setup, f'CY_tail_damp_p_{i}') for i in range(1, 1 + 6)]
        CY_tail_damp_p_value = mavrik_setup.CY_tail_damp_p_val
        self.CY_tail_damp_p_lookup_table = JaxNDInterpolator(CY_tail_damp_p_breakpoints, CY_tail_damp_p_value)

        CY_tail_damp_q_breakpoints = [getattr(mavrik_setup, f'CY_tail_damp_q_{i}') for i in range(1, 1 + 6)]
        CY_tail_damp_q_value = mavrik_setup.CY_tail_damp_q_val
        self.CY_tail_damp_q_lookup_table = JaxNDInterpolator(CY_tail_damp_q_breakpoints, CY_tail_damp_q_value)

        CY_tail_damp_r_breakpoints = [getattr(mavrik_setup, f'CY_tail_damp_r_{i}') for i in range(1, 1 + 6)]
        CY_tail_damp_r_value = mavrik_setup.CY_tail_damp_r_val
        self.CY_tail_damp_r_lookup_table = JaxNDInterpolator(CY_tail_damp_r_breakpoints, CY_tail_damp_r_value)

        CY_wing_breakpoints = [getattr(mavrik_setup, f'CY_wing_{i}') for i in range(1, 1 + 6)]
        CY_wing_value = mavrik_setup.CY_wing_val
        self.CY_wing_lookup_table = JaxNDInterpolator(CY_wing_breakpoints, CY_wing_value)

        CY_wing_damp_p_breakpoints = [getattr(mavrik_setup, f'CY_wing_damp_p_{i}') for i in range(1, 1 + 6)]
        CY_wing_damp_p_value = mavrik_setup.CY_wing_damp_p_val
        self.CY_wing_damp_p_lookup_table = JaxNDInterpolator(CY_wing_damp_p_breakpoints, CY_wing_damp_p_value)

        CY_wing_damp_q_breakpoints = [getattr(mavrik_setup, f'CY_wing_damp_q_{i}') for i in range(1, 1 + 6)]
        CY_wing_damp_q_value = mavrik_setup.CY_wing_damp_q_val
        self.CY_wing_damp_q_lookup_table = JaxNDInterpolator(CY_wing_damp_q_breakpoints, CY_wing_damp_q_value)

        CY_wing_damp_r_breakpoints = [getattr(mavrik_setup, f'CY_wing_damp_r_{i}') for i in range(1, 1 + 6)]
        CY_wing_damp_r_value = mavrik_setup.CY_wing_damp_r_val
        self.CY_wing_damp_r_lookup_table = JaxNDInterpolator(CY_wing_damp_r_breakpoints, CY_wing_damp_r_value)

        CY_hover_fuse_breakpoints = [getattr(mavrik_setup, f'CY_hover_fuse_{i}') for i in range(1, 1 + 3)]
        CY_hover_fuse_value = mavrik_setup.CY_hover_fuse_val
        self.CY_hover_fuse_lookup_table = JaxNDInterpolator(CY_hover_fuse_breakpoints, CY_hover_fuse_value)


    def Cy(self, u: ActuatorOutput, wing_transform: FloatScalar, tail_transform: FloatScalar) -> Forces:
        CY_Scale = 0.5744 * u.Q
        CY_Scale_r = 0.5744 * 2.8270 * 1.225 * 0.25 * u.U * u.r
        CY_Scale_p = 0.5744 * 2.8270 * 1.225 * 0.25 * u.U * u.p
        CY_Scale_q = 0.5744 * 0.2032 * 1.225 * 0.25 * u.U * u.q

        wing_transform = jnp.array([[jnp.cos(u.wing_tilt), 0, jnp.sin( u.wing_tilt)], [0, 1, 0], [-jnp.sin(u.wing_tilt), 0., jnp.cos(u.wing_tilt)]]);
        tail_transform = jnp.array([[jnp.cos(u.tail_tilt), 0, jnp.sin(u.tail_tilt)], [0, 1, 0], [-jnp.sin(u.tail_tilt), 0., jnp.cos(u.tail_tilt)]])
     
        CY_aileron_wing = self.CY_aileron_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron
        ]))
        CY_aileron_wing_padded = jnp.array([0.0, CY_aileron_wing, 0.0])
        CY_aileron_wing_padded_transformed = jnp.dot(wing_transform, CY_aileron_wing_padded * CY_Scale)    

        CY_elevator_tail = self.CY_elevator_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator
        ]))
        CY_elevator_tail_padded = jnp.array([0.0, CY_elevator_tail, 0.0])
        CY_elevator_tail_padded_transformed = jnp.dot(tail_transform, CY_elevator_tail_padded * CY_Scale)

        CY_flap_wing = self.CY_flap_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap
        ]))
        CY_flap_wing_padded = jnp.array([0.0, CY_flap_wing, 0.0])
        CY_flap_wing_padded_transformed = jnp.dot(wing_transform, CY_flap_wing_padded * CY_Scale)

        CY_rudder_tail = self.CY_rudder_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder
        ]))
        CY_rudder_tail_padded = jnp.array([0.0, CY_rudder_tail, 0.0])
        CY_rudder_tail_padded_transformed = jnp.dot(tail_transform, CY_rudder_tail_padded * CY_Scale)

        # Tail
        CY_tail = self.CY_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        CY_tail_padded = jnp.array([0.0, CY_tail, 0.0])
        CY_tail_padded_transformed = jnp.dot(tail_transform, CY_tail_padded * CY_Scale)

        # Tail Damp p
        CY_tail_damp_p = self.CY_tail_damp_p_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ])) 
        CY_tail_damp_p_padded = jnp.array([0.0, CY_tail_damp_p, 0.0])
        CY_tail_damp_p_padded_transformed = jnp.dot(tail_transform, CY_tail_damp_p_padded * CY_Scale_p)

        # Tail Damp q
        CY_tail_damp_q = self.CY_tail_damp_q_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        CY_tail_damp_q_padded = jnp.array([0.0, CY_tail_damp_q, 0.0])
        CY_tail_damp_q_padded_transformed = jnp.dot(tail_transform, CY_tail_damp_q_padded * CY_Scale_q)

        # Tail Damp r
        CY_tail_damp_r = self.CY_tail_damp_r_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        CY_tail_damp_r_padded = jnp.array([0.0, CY_tail_damp_r, 0.0])
        CY_tail_damp_r_padded_transformed = jnp.dot(tail_transform, CY_tail_damp_r_padded * CY_Scale_r)

        # Wing
        CY_wing = self.CY_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        CY_wing_padded = jnp.array([0.0, CY_wing, 0.0])
        CY_wing_padded_transformed = jnp.dot(wing_transform, CY_wing_padded * CY_Scale)

        # Wing Damp p
        CY_wing_damp_p = self.CY_wing_damp_p_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        CY_wing_damp_p_padded = jnp.array([0.0, CY_wing_damp_p, 0.0])
        CY_wing_damp_p_padded_transformed = jnp.dot(wing_transform, CY_wing_damp_p_padded * CY_Scale_p)

        # Wing Damp q
        CY_wing_damp_q = self.CY_wing_damp_q_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        CY_wing_damp_q_padded = jnp.array([0.0, CY_wing_damp_q, 0.0])
        CY_wing_damp_q_padded_transformed = jnp.dot(wing_transform, CY_wing_damp_q_padded * CY_Scale_q)

        # Wing Damp r
        CY_wing_damp_r = self.CY_wing_damp_r_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        CY_wing_damp_r_padded = jnp.array([0.0, CY_wing_damp_r, 0.0])
        CY_wing_damp_r_padded_transformed = jnp.dot(wing_transform, CY_wing_damp_r_padded * CY_Scale_r)

        # Hover Fuse
        CY_hover_fuse = self.CY_hover_fuse_lookup_table(jnp.array([
            u.U, u.alpha, u.beta
        ]))
        CY_hover_fuse_padded = jnp.array([0.0, CY_hover_fuse * CY_Scale, 0.0])

        return Forces(
            CY_aileron_wing_padded_transformed[0] + CY_elevator_tail_padded_transformed[0] + CY_flap_wing_padded_transformed[0] + CY_rudder_tail_padded_transformed[0] +
            CY_tail_padded_transformed[0] + CY_tail_damp_p_padded_transformed[0] + CY_tail_damp_q_padded_transformed[0] + CY_tail_damp_r_padded_transformed[0] +
            CY_wing_padded_transformed[0] + CY_wing_damp_p_padded_transformed[0] + CY_wing_damp_q_padded_transformed[0] + CY_wing_damp_r_padded_transformed[0] +
            CY_hover_fuse_padded[0],
            CY_aileron_wing_padded_transformed[1] + CY_elevator_tail_padded_transformed[1] + CY_flap_wing_padded_transformed[1] + CY_rudder_tail_padded_transformed[1] +
            CY_tail_padded_transformed[1] + CY_tail_damp_p_padded_transformed[1] + CY_tail_damp_q_padded_transformed[1] + CY_tail_damp_r_padded_transformed[1] +
            CY_wing_padded_transformed[1] + CY_wing_damp_p_padded_transformed[1] + CY_wing_damp_q_padded_transformed[1] + CY_wing_damp_r_padded_transformed[1] +
            CY_hover_fuse_padded[1],
            CY_aileron_wing_padded_transformed[2] + CY_elevator_tail_padded_transformed[2] + CY_flap_wing_padded_transformed[2] + CY_rudder_tail_padded_transformed[2] +
            CY_tail_padded_transformed[2] + CY_tail_damp_p_padded_transformed[2] + CY_tail_damp_q_padded_transformed[2] + CY_tail_damp_r_padded_transformed[2] +
            CY_wing_padded_transformed[2] + CY_wing_damp_p_padded_transformed[2] + CY_wing_damp_q_padded_transformed[2] + CY_wing_damp_r_padded_transformed[2] +
            CY_hover_fuse_padded[2]
        )
    



    def get_Cz_table(self, mavrik_setup: MavrikSetup):
        CZ_aileron_wing_breakpoints = [getattr(mavrik_setup, f'CZ_aileron_wing_{i}') for i in range(1, 1 + 7)]
        CZ_aileron_wing_value = mavrik_setup.CZ_aileron_wing_val
        self.CZ_aileron_wing_lookup_table = JaxNDInterpolator(CZ_aileron_wing_breakpoints, CZ_aileron_wing_value)

        CZ_elevator_tail_breakpoints = [getattr(mavrik_setup, f'CZ_elevator_tail_{i}') for i in range(1, 1 + 7)]
        CZ_elevator_tail_value = mavrik_setup.CZ_elevator_tail_val
        self.CZ_elevator_tail_lookup_table = JaxNDInterpolator(CZ_elevator_tail_breakpoints, CZ_elevator_tail_value)

        CZ_flap_wing_breakpoints = [getattr(mavrik_setup, f'CZ_flap_wing_{i}') for i in range(1, 1 + 7)]
        CZ_flap_wing_value = mavrik_setup.CZ_flap_wing_val
        self.CZ_flap_wing_lookup_table = JaxNDInterpolator(CZ_flap_wing_breakpoints, CZ_flap_wing_value)

        CZ_aileron_wing_breakpoints = [getattr(mavrik_setup, f'CZ_aileron_wing_{i}') for i in range(1, 1 + 7)]
        CZ_aileron_wing_value = mavrik_setup.CZ_aileron_wing_val
        self.CZ_aileron_wing_lookup_table = JaxNDInterpolator(CZ_aileron_wing_breakpoints, CZ_aileron_wing_value)

        CZ_elevator_tail_breakpoints = [getattr(mavrik_setup, f'CZ_elevator_tail_{i}') for i in range(1, 1 + 7)]
        CZ_elevator_tail_value = mavrik_setup.CZ_elevator_tail_val
        self.CZ_elevator_tail_lookup_table = JaxNDInterpolator(CZ_elevator_tail_breakpoints, CZ_elevator_tail_value)

        CZ_flap_wing_breakpoints = [getattr(mavrik_setup, f'CZ_flap_wing_{i}') for i in range(1, 1 + 7)]
        CZ_flap_wing_value = mavrik_setup.CZ_flap_wing_val
        self.CZ_flap_wing_lookup_table = JaxNDInterpolator(CZ_flap_wing_breakpoints, CZ_flap_wing_value)

        CZ_rudder_tail_breakpoints = [getattr(mavrik_setup, f'CZ_rudder_tail_{i}') for i in range(1, 1 + 7)]
        CZ_rudder_tail_value = mavrik_setup.CZ_rudder_tail_val
        self.CZ_rudder_tail_lookup_table = JaxNDInterpolator(CZ_rudder_tail_breakpoints, CZ_rudder_tail_value)

        CZ_tail_breakpoints = [getattr(mavrik_setup, f'CZ_tail_{i}') for i in range(1, 1 + 6)]
        CZ_tail_value = mavrik_setup.CZ_tail_val
        self.CZ_tail_lookup_table = JaxNDInterpolator(CZ_tail_breakpoints, CZ_tail_value)

        CZ_tail_damp_p_breakpoints = [getattr(mavrik_setup, f'CZ_tail_damp_p_{i}') for i in range(1, 1 + 6)]
        CZ_tail_damp_p_value = mavrik_setup.CZ_tail_damp_p_val
        self.CZ_tail_damp_p_lookup_table = JaxNDInterpolator(CZ_tail_damp_p_breakpoints, CZ_tail_damp_p_value)

        CZ_tail_damp_q_breakpoints = [getattr(mavrik_setup, f'CZ_tail_damp_q_{i}') for i in range(1, 1 + 6)]
        CZ_tail_damp_q_value = mavrik_setup.CZ_tail_damp_q_val
        self.CZ_tail_damp_q_lookup_table = JaxNDInterpolator(CZ_tail_damp_q_breakpoints, CZ_tail_damp_q_value)

        CZ_tail_damp_r_breakpoints = [getattr(mavrik_setup, f'CZ_tail_damp_r_{i}') for i in range(1, 1 + 6)]
        CZ_tail_damp_r_value = mavrik_setup.CZ_tail_damp_r_val
        self.CZ_tail_damp_r_lookup_table = JaxNDInterpolator(CZ_tail_damp_r_breakpoints, CZ_tail_damp_r_value)

        CZ_wing_breakpoints = [getattr(mavrik_setup, f'CZ_wing_{i}') for i in range(1, 1 + 6)]
        CZ_wing_value = mavrik_setup.CZ_wing_val
        self.CZ_wing_lookup_table = JaxNDInterpolator(CZ_wing_breakpoints, CZ_wing_value)

        CZ_wing_damp_p_breakpoints = [getattr(mavrik_setup, f'CZ_wing_damp_p_{i}') for i in range(1, 1 + 6)]
        CZ_wing_damp_p_value = mavrik_setup.CZ_wing_damp_p_val
        self.CZ_wing_damp_p_lookup_table = JaxNDInterpolator(CZ_wing_damp_p_breakpoints, CZ_wing_damp_p_value)

        CZ_wing_damp_q_breakpoints = [getattr(mavrik_setup, f'CZ_wing_damp_q_{i}') for i in range(1, 1 + 6)]
        CZ_wing_damp_q_value = mavrik_setup.CZ_wing_damp_q_val
        self.CZ_wing_damp_q_lookup_table = JaxNDInterpolator(CZ_wing_damp_q_breakpoints, CZ_wing_damp_q_value)

        CZ_wing_damp_r_breakpoints = [getattr(mavrik_setup, f'CZ_wing_damp_r_{i}') for i in range(1, 1 + 6)]
        CZ_wing_damp_r_value = mavrik_setup.CZ_wing_damp_r_val
        self.CZ_wing_damp_r_lookup_table = JaxNDInterpolator(CZ_wing_damp_r_breakpoints, CZ_wing_damp_r_value)

        CZ_hover_fuse_breakpoints = [getattr(mavrik_setup, f'CZ_hover_fuse_{i}') for i in range(1, 1 + 3)]
        CZ_hover_fuse_value = mavrik_setup.CZ_hover_fuse_val
        self.CZ_hover_fuse_lookup_table = JaxNDInterpolator(CZ_hover_fuse_breakpoints, CZ_hover_fuse_value)
    
    def Cz(self, u: ActuatorOutput, wing_transform: FloatScalar, tail_transform: FloatScalar) -> Forces:
        CZ_Scale = 0.5744 * u.Q
        CZ_Scale_r = 0.5744 * 2.8270 * 1.225 * 0.25 * u.U * u.r
        CZ_Scale_p = 0.5744 * 2.8270 * 1.225 * 0.25 * u.U * u.p
        CZ_Scale_q = 0.5744 * 0.2032 * 1.225 * 0.25 * u.U * u.q
            
        wing_transform = jnp.array([[jnp.cos(u.wing_tilt), 0, jnp.sin(u.wing_tilt)], [0, 1, 0], [-jnp.sin(u.wing_tilt), 0., jnp.cos(u.wing_tilt)]])
        tail_transform = jnp.array([[jnp.cos(u.tail_tilt), 0, jnp.sin(u.tail_tilt)], [0, 1, 0], [-jnp.sin(u.tail_tilt), 0., jnp.cos(u.tail_tilt)]])

         
        CZ_aileron_wing = self.CZ_aileron_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron
        ]))
        CZ_aileron_wing_padded = jnp.array([0.0, 0.0, CZ_aileron_wing])
        CZ_aileron_wing_padded_transformed = jnp.dot(wing_transform, CZ_aileron_wing_padded * CZ_Scale)

        CZ_elevator_tail = self.CZ_elevator_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator
        ]))
        CZ_elevator_tail_padded = jnp.array([0.0, 0.0, CZ_elevator_tail])
        CZ_elevator_tail_padded_transformed = jnp.dot(tail_transform, CZ_elevator_tail_padded * CZ_Scale)

        
        CZ_flap_wing = self.CZ_flap_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap
        ]))
        CZ_flap_wing_padded = jnp.array([0.0, 0.0, CZ_flap_wing])
        CZ_flap_wing_padded_transformed = jnp.dot(wing_transform, CZ_flap_wing_padded * CZ_Scale)

        CZ_rudder_tail = self.CZ_rudder_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder
        ]))
        CZ_rudder_tail_padded = jnp.array([0.0, 0.0, CZ_rudder_tail])
        CZ_rudder_tail_padded_transformed = jnp.dot(tail_transform, CZ_rudder_tail_padded * CZ_Scale)

        # Tail
        CZ_tail = self.CZ_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        CZ_tail_padded = jnp.array([0.0, 0.0, CZ_tail])
        CZ_tail_padded_transformed = jnp.dot(tail_transform, CZ_tail_padded * CZ_Scale)

        # Tail Damp p
        CZ_tail_damp_p = self.CZ_tail_damp_p_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        CZ_tail_damp_p_padded = jnp.array([0.0, 0.0, CZ_tail_damp_p])
        CZ_tail_damp_p_padded_transformed = jnp.dot(tail_transform, CZ_tail_damp_p_padded * CZ_Scale_p)

        # Tail Damp q
        CZ_tail_damp_q = self.CZ_tail_damp_q_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        CZ_tail_damp_q_padded = jnp.array([0.0, 0.0, CZ_tail_damp_q])
        CZ_tail_damp_q_padded_transformed = jnp.dot(tail_transform, CZ_tail_damp_q_padded * CZ_Scale_q)

        # Tail Damp r
        CZ_tail_damp_r = self.CZ_tail_damp_r_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        CZ_tail_damp_r_padded = jnp.array([0.0, 0.0, CZ_tail_damp_r])
        CZ_tail_damp_r_padded_transformed = jnp.dot(tail_transform, CZ_tail_damp_r_padded * CZ_Scale_r)

        # Wing
        CZ_wing = self.CZ_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        CZ_wing_padded = jnp.array([0.0, 0.0, CZ_wing])
        CZ_wing_padded_transformed = jnp.dot(wing_transform, CZ_wing_padded * CZ_Scale)

        # Wing Damp p
        CZ_wing_damp_p = self.CZ_wing_damp_p_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        CZ_wing_damp_p_padded = jnp.array([0.0, 0.0, CZ_wing_damp_p])
        CZ_wing_damp_p_padded_transformed = jnp.dot(wing_transform, CZ_wing_damp_p_padded * CZ_Scale_p)

        # Wing Damp q
        CZ_wing_damp_q = self.CZ_wing_damp_q_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        CZ_wing_damp_q_padded = jnp.array([0.0, 0.0, CZ_wing_damp_q])
        CZ_wing_damp_q_padded_transformed = jnp.dot(wing_transform, CZ_wing_damp_q_padded * CZ_Scale_q)

        # Wing Damp r
        CZ_wing_damp_r = self.CZ_wing_damp_r_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        CZ_wing_damp_r_padded = jnp.array([0.0, 0.0, CZ_wing_damp_r])
        CZ_wing_damp_r_padded_transformed = jnp.dot(wing_transform, CZ_wing_damp_r_padded * CZ_Scale_r)

        # Hover Fuse
        CZ_hover_fuse = self.CZ_hover_fuse_lookup_table(jnp.array([
            u.U, u.alpha, u.beta
        ]))
        CZ_hover_fuse_padded = jnp.array([0.0, 0.0, CZ_hover_fuse * CZ_Scale])

        return Forces(
            CZ_aileron_wing_padded_transformed[0] + CZ_elevator_tail_padded_transformed[0] + CZ_flap_wing_padded_transformed[0] + CZ_rudder_tail_padded_transformed[0] +
            CZ_tail_padded_transformed[0] + CZ_tail_damp_p_padded_transformed[0] + CZ_tail_damp_q_padded_transformed[0] + CZ_tail_damp_r_padded_transformed[0] +
            CZ_wing_padded_transformed[0] + CZ_wing_damp_p_padded_transformed[0] + CZ_wing_damp_q_padded_transformed[0] + CZ_wing_damp_r_padded_transformed[0] +
            CZ_hover_fuse_padded[0],
            CZ_aileron_wing_padded_transformed[1] + CZ_elevator_tail_padded_transformed[1] + CZ_flap_wing_padded_transformed[1] + CZ_rudder_tail_padded_transformed[1] +
            CZ_tail_padded_transformed[1] + CZ_tail_damp_p_padded_transformed[1] + CZ_tail_damp_q_padded_transformed[1] + CZ_tail_damp_r_padded_transformed[1] +
            CZ_wing_padded_transformed[1] + CZ_wing_damp_p_padded_transformed[1] + CZ_wing_damp_q_padded_transformed[1] + CZ_wing_damp_r_padded_transformed[1] +
            CZ_hover_fuse_padded[1],
            CZ_aileron_wing_padded_transformed[2] + CZ_elevator_tail_padded_transformed[2] + CZ_flap_wing_padded_transformed[2] + CZ_rudder_tail_padded_transformed[2] +
            CZ_tail_padded_transformed[2] + CZ_tail_damp_p_padded_transformed[2] + CZ_tail_damp_q_padded_transformed[2] + CZ_tail_damp_r_padded_transformed[2] +
            CZ_wing_padded_transformed[2] + CZ_wing_damp_p_padded_transformed[2] + CZ_wing_damp_q_padded_transformed[2] + CZ_wing_damp_r_padded_transformed[2] +
            CZ_hover_fuse_padded[2]
        )
    


    def get_Cl_table(self, mavrik_setup: MavrikSetup):
        Cl_aileron_wing_breakpoints = [getattr(mavrik_setup, f'Cl_aileron_wing_{i}') for i in range(1, 1 + 7)]
        Cl_aileron_wing_value = mavrik_setup.Cl_aileron_wing_val
        self.Cl_aileron_wing_lookup_table = JaxNDInterpolator(Cl_aileron_wing_breakpoints, Cl_aileron_wing_value)

        Cl_elevator_tail_breakpoints = [getattr(mavrik_setup, f'Cl_elevator_tail_{i}') for i in range(1, 1 + 7)]
        Cl_elevator_tail_value = mavrik_setup.Cl_elevator_tail_val
        self.Cl_elevator_tail_lookup_table = JaxNDInterpolator(Cl_elevator_tail_breakpoints, Cl_elevator_tail_value)

        Cl_flap_wing_breakpoints = [getattr(mavrik_setup, f'Cl_flap_wing_{i}') for i in range(1, 1 + 7)]
        Cl_flap_wing_value = mavrik_setup.Cl_flap_wing_val
        self.Cl_flap_wing_lookup_table = JaxNDInterpolator(Cl_flap_wing_breakpoints, Cl_flap_wing_value)

        Cl_rudder_tail_breakpoints = [getattr(mavrik_setup, f'Cl_rudder_tail_{i}') for i in range(1, 1 + 7)]
        Cl_rudder_tail_value = mavrik_setup.Cl_rudder_tail_val
        self.Cl_rudder_tail_lookup_table = JaxNDInterpolator(Cl_rudder_tail_breakpoints, Cl_rudder_tail_value)

        Cl_tail_breakpoints = [getattr(mavrik_setup, f'Cl_tail_{i}') for i in range(1, 1 + 6)]
        Cl_tail_value = mavrik_setup.Cl_tail_val
        self.Cl_tail_lookup_table = JaxNDInterpolator(Cl_tail_breakpoints, Cl_tail_value)

        Cl_tail_damp_p_breakpoints = [getattr(mavrik_setup, f'Cl_tail_damp_p_{i}') for i in range(1, 1 + 6)]
        Cl_tail_damp_p_value = mavrik_setup.Cl_tail_damp_p_val
        self.Cl_tail_damp_p_lookup_table = JaxNDInterpolator(Cl_tail_damp_p_breakpoints, Cl_tail_damp_p_value)

        Cl_tail_damp_q_breakpoints = [getattr(mavrik_setup, f'Cl_tail_damp_q_{i}') for i in range(1, 1 + 6)]
        Cl_tail_damp_q_value = mavrik_setup.Cl_tail_damp_q_val
        self.Cl_tail_damp_q_lookup_table = JaxNDInterpolator(Cl_tail_damp_q_breakpoints, Cl_tail_damp_q_value)

        Cl_tail_damp_r_breakpoints = [getattr(mavrik_setup, f'Cl_tail_damp_r_{i}') for i in range(1, 1 + 6)]
        Cl_tail_damp_r_value = mavrik_setup.Cl_tail_damp_r_val
        self.Cl_tail_damp_r_lookup_table = JaxNDInterpolator(Cl_tail_damp_r_breakpoints, Cl_tail_damp_r_value)

        Cl_wing_breakpoints = [getattr(mavrik_setup, f'Cl_wing_{i}') for i in range(1, 1 + 6)]
        Cl_wing_value = mavrik_setup.Cl_wing_val
        self.Cl_wing_lookup_table = JaxNDInterpolator(Cl_wing_breakpoints, Cl_wing_value)

        Cl_wing_damp_p_breakpoints = [getattr(mavrik_setup, f'Cl_wing_damp_p_{i}') for i in range(1, 1 + 6)]
        Cl_wing_damp_p_value = mavrik_setup.Cl_wing_damp_p_val
        self.Cl_wing_damp_p_lookup_table = JaxNDInterpolator(Cl_wing_damp_p_breakpoints, Cl_wing_damp_p_value)

        Cl_wing_damp_q_breakpoints = [getattr(mavrik_setup, f'Cl_wing_damp_q_{i}') for i in range(1, 1 + 6)]
        Cl_wing_damp_q_value = mavrik_setup.Cl_wing_damp_q_val
        self.Cl_wing_damp_q_lookup_table = JaxNDInterpolator(Cl_wing_damp_q_breakpoints, Cl_wing_damp_q_value)

        Cl_wing_damp_r_breakpoints = [getattr(mavrik_setup, f'Cl_wing_damp_r_{i}') for i in range(1, 1 + 6)]
        Cl_wing_damp_r_value = mavrik_setup.Cl_wing_damp_r_val
        self.Cl_wing_damp_r_lookup_table = JaxNDInterpolator(Cl_wing_damp_r_breakpoints, Cl_wing_damp_r_value)

        Cl_hover_fuse_breakpoints = [getattr(mavrik_setup, f'Cl_hover_fuse_{i}') for i in range(1, 1 + 3)]
        Cl_hover_fuse_value = mavrik_setup.Cl_hover_fuse_val
        self.Cl_hover_fuse_lookup_table = JaxNDInterpolator(Cl_hover_fuse_breakpoints, Cl_hover_fuse_value)

    def L(self, u: ActuatorOutput, wing_transform: FloatScalar, tail_transform: FloatScalar) -> Moments:
        Cl_Scale = 0.5744 * 2.8270 * u.Q
        Cl_Scale_p = 0.5744 * 2.8270**2 * 1.225 * 0.25 * u.U * u.p
        Cl_Scale_q = 0.5744 * 2.8270 * 0.2032 * 1.225 * 0.25 * u.U * u.q
        Cl_Scale_r = 0.5744 * 2.8270**2 * 1.225 * 0.25 * u.U * u.r
 
        Cl_aileron_wing = self.Cl_aileron_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron
        ]))
        Cl_aileron_wing_padded = jnp.array([Cl_aileron_wing, 0.0, 0.0])
        Cl_aileron_wing_padded_transformed = jnp.dot(wing_transform, Cl_aileron_wing_padded * Cl_Scale)

      
        Cl_elevator_tail = self.Cl_elevator_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator
        ]))
        Cl_elevator_tail_padded = jnp.array([Cl_elevator_tail, 0.0, 0.0])
        Cl_elevator_tail_padded_transformed = jnp.dot(tail_transform, Cl_elevator_tail_padded * Cl_Scale)

        Cl_flap_wing = self.Cl_flap_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap
        ]))
        Cl_flap_wing_padded = jnp.array([Cl_flap_wing, 0.0, 0.0])
        Cl_flap_wing_padded_transformed = jnp.dot(wing_transform, Cl_flap_wing_padded * Cl_Scale)

        Cl_rudder_tail = self.Cl_rudder_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder
        ]))
        Cl_rudder_tail_padded = jnp.array([Cl_rudder_tail, 0.0, 0.0])
        Cl_rudder_tail_padded_transformed = jnp.dot(tail_transform, Cl_rudder_tail_padded * Cl_Scale)

        # Tail
        Cl_tail = self.Cl_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        Cl_tail_padded = jnp.array([Cl_tail, 0.0, 0.0])
        Cl_tail_padded_transformed = jnp.dot(tail_transform, Cl_tail_padded * Cl_Scale)

        # Tail Damp p
        Cl_tail_damp_p = self.Cl_tail_damp_p_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        Cl_tail_damp_p_padded = jnp.array([Cl_tail_damp_p, 0.0, 0.0])
        Cl_tail_damp_p_padded_transformed = jnp.dot(tail_transform, Cl_tail_damp_p_padded * Cl_Scale_p)

        # Tail Damp q
        Cl_tail_damp_q = self.Cl_tail_damp_q_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        Cl_tail_damp_q_padded = jnp.array([Cl_tail_damp_q, 0.0, 0.0])
        Cl_tail_damp_q_padded_transformed = jnp.dot(tail_transform, Cl_tail_damp_q_padded * Cl_Scale_q)

        # Tail Damp r
        Cl_tail_damp_r = self.Cl_tail_damp_r_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        Cl_tail_damp_r_padded = jnp.array([Cl_tail_damp_r, 0.0, 0.0])
        Cl_tail_damp_r_padded_transformed = jnp.dot(tail_transform, Cl_tail_damp_r_padded * Cl_Scale_r)

        # Wing
        Cl_wing = self.Cl_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Cl_wing_padded = jnp.array([Cl_wing, 0.0, 0.0])
        Cl_wing_padded_transformed = jnp.dot(wing_transform, Cl_wing_padded * Cl_Scale)

        # Wing Damp p
        Cl_wing_damp_p = self.Cl_wing_damp_p_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Cl_wing_damp_p_padded = jnp.array([Cl_wing_damp_p, 0.0, 0.0])
        Cl_wing_damp_p_padded_transformed = jnp.dot(wing_transform, Cl_wing_damp_p_padded * Cl_Scale_p)

        # Wing Damp q
        Cl_wing_damp_q = self.Cl_wing_damp_q_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Cl_wing_damp_q_padded = jnp.array([Cl_wing_damp_q, 0.0, 0.0])
        Cl_wing_damp_q_padded_transformed = jnp.dot(wing_transform, Cl_wing_damp_q_padded * Cl_Scale_q)

        # Wing Damp r
        Cl_wing_damp_r = self.Cl_wing_damp_r_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Cl_wing_damp_r_padded = jnp.array([Cl_wing_damp_r, 0.0, 0.0])
        Cl_wing_damp_r_padded_transformed = jnp.dot(wing_transform, Cl_wing_damp_r_padded * Cl_Scale_r)

        # Hover Fuse
        Cl_hover_fuse = self.Cl_hover_fuse_lookup_table(jnp.array([
            u.U, u.alpha, u.beta
        ]))
        Cl_hover_fuse_padded = jnp.array([Cl_hover_fuse * Cl_Scale, 0.0, 0.0])

        return Moments(
            Cl_aileron_wing_padded_transformed[0] + Cl_elevator_tail_padded_transformed[0] + Cl_flap_wing_padded_transformed[0] + Cl_rudder_tail_padded_transformed[0] +
            Cl_tail_padded_transformed[0] + Cl_tail_damp_p_padded_transformed[0] + Cl_tail_damp_q_padded_transformed[0] + Cl_tail_damp_r_padded_transformed[0] +
            Cl_wing_padded_transformed[0] + Cl_wing_damp_p_padded_transformed[0] + Cl_wing_damp_q_padded_transformed[0] + Cl_wing_damp_r_padded_transformed[0] +
            Cl_hover_fuse_padded[0],
            Cl_aileron_wing_padded_transformed[1] + Cl_elevator_tail_padded_transformed[1] + Cl_flap_wing_padded_transformed[1] + Cl_rudder_tail_padded_transformed[1] +
            Cl_tail_padded_transformed[1] + Cl_tail_damp_p_padded_transformed[1] + Cl_tail_damp_q_padded_transformed[1] + Cl_tail_damp_r_padded_transformed[1] +
            Cl_wing_padded_transformed[1] + Cl_wing_damp_p_padded_transformed[1] + Cl_wing_damp_q_padded_transformed[1] + Cl_wing_damp_r_padded_transformed[1] +
            Cl_hover_fuse_padded[1],
            Cl_aileron_wing_padded_transformed[2] + Cl_elevator_tail_padded_transformed[2] + Cl_flap_wing_padded_transformed[2] + Cl_rudder_tail_padded_transformed[2] +
            Cl_tail_padded_transformed[2] + Cl_tail_damp_p_padded_transformed[2] + Cl_tail_damp_q_padded_transformed[2] + Cl_tail_damp_r_padded_transformed[2] +
            Cl_wing_padded_transformed[2] + Cl_wing_damp_p_padded_transformed[2] + Cl_wing_damp_q_padded_transformed[2] + Cl_wing_damp_r_padded_transformed[2] +
            Cl_hover_fuse_padded[2]
        )
    


    def get_Cm_table(self, mavrik_setup: MavrikSetup):
        Cm_aileron_wing_breakpoints = [getattr(mavrik_setup, f'Cm_aileron_wing_{i}') for i in range(1, 1 + 7)]
        Cm_aileron_wing_value = mavrik_setup.Cm_aileron_wing_val
        self.Cm_aileron_wing_lookup_table = JaxNDInterpolator(Cm_aileron_wing_breakpoints, Cm_aileron_wing_value)

        Cm_elevator_tail_breakpoints = [getattr(mavrik_setup, f'Cm_elevator_tail_{i}') for i in range(1, 1 + 7)]
        Cm_elevator_tail_value = mavrik_setup.Cm_elevator_tail_val
        self.Cm_elevator_tail_lookup_table = JaxNDInterpolator(Cm_elevator_tail_breakpoints, Cm_elevator_tail_value)

        Cm_flap_wing_breakpoints = [getattr(mavrik_setup, f'Cm_flap_wing_{i}') for i in range(1, 1 + 7)]
        Cm_flap_wing_value = mavrik_setup.Cm_flap_wing_val
        self.Cm_flap_wing_lookup_table = JaxNDInterpolator(Cm_flap_wing_breakpoints, Cm_flap_wing_value)

        Cm_rudder_tail_breakpoints = [getattr(mavrik_setup, f'Cm_rudder_tail_{i}') for i in range(1, 1 + 7)]
        Cm_rudder_tail_value = mavrik_setup.Cm_rudder_tail_val
        self.Cm_rudder_tail_lookup_table = JaxNDInterpolator(Cm_rudder_tail_breakpoints, Cm_rudder_tail_value)

        Cm_tail_breakpoints = [getattr(mavrik_setup, f'Cm_tail_{i}') for i in range(1, 1 + 6)]
        Cm_tail_value = mavrik_setup.Cm_tail_val
        self.Cm_tail_lookup_table = JaxNDInterpolator(Cm_tail_breakpoints, Cm_tail_value)

        Cm_tail_damp_p_breakpoints = [getattr(mavrik_setup, f'Cm_tail_damp_p_{i}') for i in range(1, 1 + 6)]
        Cm_tail_damp_p_value = mavrik_setup.Cm_tail_damp_p_val
        self.Cm_tail_damp_p_lookup_table = JaxNDInterpolator(Cm_tail_damp_p_breakpoints, Cm_tail_damp_p_value)

        Cm_tail_damp_q_breakpoints = [getattr(mavrik_setup, f'Cm_tail_damp_q_{i}') for i in range(1, 1 + 6)]
        Cm_tail_damp_q_value = mavrik_setup.Cm_tail_damp_q_val
        self.Cm_tail_damp_q_lookup_table = JaxNDInterpolator(Cm_tail_damp_q_breakpoints, Cm_tail_damp_q_value)

        Cm_tail_damp_r_breakpoints = [getattr(mavrik_setup, f'Cm_tail_damp_r_{i}') for i in range(1, 1 + 6)]
        Cm_tail_damp_r_value = mavrik_setup.Cm_tail_damp_r_val
        self.Cm_tail_damp_r_lookup_table = JaxNDInterpolator(Cm_tail_damp_r_breakpoints, Cm_tail_damp_r_value)

        Cm_wing_breakpoints = [getattr(mavrik_setup, f'Cm_wing_{i}') for i in range(1, 1 + 6)]
        Cm_wing_value = mavrik_setup.Cm_wing_val
        self.Cm_wing_lookup_table = JaxNDInterpolator(Cm_wing_breakpoints, Cm_wing_value)

        Cm_wing_damp_p_breakpoints = [getattr(mavrik_setup, f'Cm_wing_damp_p_{i}') for i in range(1, 1 + 6)]
        Cm_wing_damp_p_value = mavrik_setup.Cm_wing_damp_p_val
        self.Cm_wing_damp_p_lookup_table = JaxNDInterpolator(Cm_wing_damp_p_breakpoints, Cm_wing_damp_p_value)

        Cm_wing_damp_q_breakpoints = [getattr(mavrik_setup, f'Cm_wing_damp_q_{i}') for i in range(1, 1 + 6)]
        Cm_wing_damp_q_value = mavrik_setup.Cm_wing_damp_q_val
        self.Cm_wing_damp_q_lookup_table = JaxNDInterpolator(Cm_wing_damp_q_breakpoints, Cm_wing_damp_q_value)

        Cm_wing_damp_r_breakpoints = [getattr(mavrik_setup, f'Cm_wing_damp_r_{i}') for i in range(1, 1 + 6)]
        Cm_wing_damp_r_value = mavrik_setup.Cm_wing_damp_r_val
        self.Cm_wing_damp_r_lookup_table = JaxNDInterpolator(Cm_wing_damp_r_breakpoints, Cm_wing_damp_r_value)

        Cm_hover_fuse_breakpoints = [getattr(mavrik_setup, f'Cm_hover_fuse_{i}') for i in range(1, 1 + 3)]
        Cm_hover_fuse_value = mavrik_setup.Cm_hover_fuse_val
        self.Cm_hover_fuse_lookup_table = JaxNDInterpolator(Cm_hover_fuse_breakpoints, Cm_hover_fuse_value)

    def M(self, u: ActuatorOutput, wing_transform: FloatScalar, tail_transform: FloatScalar) -> Moments:
        Cm_Scale = 0.5744 * 0.2032 * u.Q
        Cm_Scale_p = 0.5744 * 0.2032 * 2.8270 * 1.225 * 0.25 * u.U * u.p
        Cm_Scale_q = 0.5744 * 0.2032**2 * 1.225 * 0.25 * u.U * u.q
        Cm_Scale_r = 0.5744 * 0.2032 * 2.8270 * 1.225 * 0.25 * u.U * u.r
 
        Cm_aileron_wing = self.Cm_aileron_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron
        ]))
        Cm_aileron_wing_padded = jnp.array([0.0, Cm_aileron_wing, 0.0])
        Cm_aileron_wing_padded_transformed = jnp.dot(wing_transform, Cm_aileron_wing_padded * Cm_Scale)

        Cm_elevator_tail = self.Cm_elevator_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator
        ]))
        Cm_elevator_tail_padded = jnp.array([0.0, Cm_elevator_tail, 0.0])
        Cm_elevator_tail_padded_transformed = jnp.dot(tail_transform, Cm_elevator_tail_padded * Cm_Scale)

        Cm_flap_wing = self.Cm_flap_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap
        ]))
        Cm_flap_wing_padded = jnp.array([0.0, Cm_flap_wing, 0.0])
        Cm_flap_wing_padded_transformed = jnp.dot(wing_transform, Cm_flap_wing_padded * Cm_Scale)

        Cm_rudder_tail = self.Cm_rudder_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder
        ]))
        Cm_rudder_tail_padded = jnp.array([0.0, Cm_rudder_tail, 0.0])
        Cm_rudder_tail_padded_transformed = jnp.dot(tail_transform, Cm_rudder_tail_padded * Cm_Scale)

        # Tail
        Cm_tail = self.Cm_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        Cm_tail_padded = jnp.array([0.0, Cm_tail, 0.0])
        Cm_tail_padded_transformed = jnp.dot(tail_transform, Cm_tail_padded * Cm_Scale)

        # Tail Damp p
        Cm_tail_damp_p = self.Cm_tail_damp_p_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        Cm_tail_damp_p_padded = jnp.array([0.0, Cm_tail_damp_p, 0.0])
        Cm_tail_damp_p_padded_transformed = jnp.dot(tail_transform, Cm_tail_damp_p_padded * Cm_Scale_p)

        # Tail Damp q
        Cm_tail_damp_q = self.Cm_tail_damp_q_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        Cm_tail_damp_q_padded = jnp.array([0.0, Cm_tail_damp_q, 0.0])
        Cm_tail_damp_q_padded_transformed = jnp.dot(tail_transform, Cm_tail_damp_q_padded * Cm_Scale_q)

        # Tail Damp r
        # Tail Damp r
        Cm_tail_damp_r = self.Cm_tail_damp_r_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        Cm_tail_damp_r_padded = jnp.array([0.0, Cm_tail_damp_r, 0.0])
        Cm_tail_damp_r_padded_transformed = jnp.dot(tail_transform, Cm_tail_damp_r_padded * Cm_Scale_r)

        # Wing
        Cm_wing = self.Cm_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Cm_wing_padded = jnp.array([0.0, Cm_wing, 0.0])
        Cm_wing_padded_transformed = jnp.dot(wing_transform, Cm_wing_padded * Cm_Scale)

        # Wing Damp p
        Cm_wing_damp_p = self.Cm_wing_damp_p_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Cm_wing_damp_p_padded = jnp.array([0.0, Cm_wing_damp_p, 0.0])
        Cm_wing_damp_p_padded_transformed = jnp.dot(wing_transform, Cm_wing_damp_p_padded * Cm_Scale_p)

        # Wing Damp q
        Cm_wing_damp_q = self.Cm_wing_damp_q_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Cm_wing_damp_q_padded = jnp.array([0.0, Cm_wing_damp_q, 0.0])
        Cm_wing_damp_q_padded_transformed = jnp.dot(wing_transform, Cm_wing_damp_q_padded * Cm_Scale_q)

        # Wing Damp r
        Cm_wing_damp_r = self.Cm_wing_damp_r_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Cm_wing_damp_r_padded = jnp.array([0.0, Cm_wing_damp_r, 0.0])
        Cm_wing_damp_r_padded_transformed = jnp.dot(wing_transform, Cm_wing_damp_r_padded * Cm_Scale_r)

        # Hover Fuse
        Cm_hover_fuse = self.Cm_hover_fuse_lookup_table(jnp.array([
            u.U, u.alpha, u.beta
        ]))
        Cm_hover_fuse_padded = jnp.array([0.0, Cm_hover_fuse * Cm_Scale, 0.0])
          

        return Moments(
            Cm_aileron_wing_padded_transformed[0] + Cm_elevator_tail_padded_transformed[0] + Cm_flap_wing_padded_transformed[0] + Cm_rudder_tail_padded_transformed[0] +
            Cm_tail_padded_transformed[0] + Cm_tail_damp_p_padded_transformed[0] + Cm_tail_damp_q_padded_transformed[0] + Cm_tail_damp_r_padded_transformed[0] +
            Cm_wing_padded_transformed[0] + Cm_wing_damp_p_padded_transformed[0] + Cm_wing_damp_q_padded_transformed[0] + Cm_wing_damp_r_padded_transformed[0] +
            Cm_hover_fuse_padded[0],
            Cm_aileron_wing_padded_transformed[1] + Cm_elevator_tail_padded_transformed[1] + Cm_flap_wing_padded_transformed[1] + Cm_rudder_tail_padded_transformed[1] +
            Cm_tail_padded_transformed[1] + Cm_tail_damp_p_padded_transformed[1] + Cm_tail_damp_q_padded_transformed[1] + Cm_tail_damp_r_padded_transformed[1] +
            Cm_wing_padded_transformed[1] + Cm_wing_damp_p_padded_transformed[1] + Cm_wing_damp_q_padded_transformed[1] + Cm_wing_damp_r_padded_transformed[1] +
            Cm_hover_fuse_padded[1],
            Cm_aileron_wing_padded_transformed[2] + Cm_elevator_tail_padded_transformed[2] + Cm_flap_wing_padded_transformed[2] + Cm_rudder_tail_padded_transformed[2] +
            Cm_tail_padded_transformed[2] + Cm_tail_damp_p_padded_transformed[2] + Cm_tail_damp_q_padded_transformed[2] + Cm_tail_damp_r_padded_transformed[2] +
            Cm_wing_padded_transformed[2] + Cm_wing_damp_p_padded_transformed[2] + Cm_wing_damp_q_padded_transformed[2] + Cm_wing_damp_r_padded_transformed[2] +
            Cm_hover_fuse_padded[2]
        )


    def get_Cn_table(self, mavrik_setup: MavrikSetup):
        Cn_aileron_wing_breakpoints = [getattr(mavrik_setup, f'Cn_aileron_wing_{i}') for i in range(1, 1 + 7)]
        Cn_aileron_wing_value = mavrik_setup.Cn_aileron_wing_val
        self.Cn_aileron_wing_lookup_table = JaxNDInterpolator(Cn_aileron_wing_breakpoints, Cn_aileron_wing_value)

        Cn_elevator_tail_breakpoints = [getattr(mavrik_setup, f'Cn_elevator_tail_{i}') for i in range(1, 1 + 7)]
        Cn_elevator_tail_value = mavrik_setup.Cn_elevator_tail_val
        self.Cn_elevator_tail_lookup_table = JaxNDInterpolator(Cn_elevator_tail_breakpoints, Cn_elevator_tail_value)

        Cn_flap_wing_breakpoints = [getattr(mavrik_setup, f'Cn_flap_wing_{i}') for i in range(1, 1 + 7)]
        Cn_flap_wing_value = mavrik_setup.Cn_flap_wing_val
        self.Cn_flap_wing_lookup_table = JaxNDInterpolator(Cn_flap_wing_breakpoints, Cn_flap_wing_value)

        Cn_rudder_tail_breakpoints = [getattr(mavrik_setup, f'Cn_rudder_tail_{i}') for i in range(1, 1 + 7)]
        Cn_rudder_tail_value = mavrik_setup.Cn_rudder_tail_val
        self.Cn_rudder_tail_lookup_table = JaxNDInterpolator(Cn_rudder_tail_breakpoints, Cn_rudder_tail_value)
        
        Cn_tail_breakpoints = [getattr(mavrik_setup, f'Cn_tail_{i}') for i in range(1, 1 + 6)]
        Cn_tail_value = mavrik_setup.Cn_tail_val
        self.Cn_tail_lookup_table = JaxNDInterpolator(Cn_tail_breakpoints, Cn_tail_value)

        Cn_tail_damp_p_breakpoints = [getattr(mavrik_setup, f'Cn_tail_damp_p_{i}') for i in range(1, 1 + 6)]
        Cn_tail_damp_p_value = mavrik_setup.Cn_tail_damp_p_val
        self.Cn_tail_damp_p_lookup_table = JaxNDInterpolator(Cn_tail_damp_p_breakpoints, Cn_tail_damp_p_value)

        Cn_tail_damp_q_breakpoints = [getattr(mavrik_setup, f'Cn_tail_damp_q_{i}') for i in range(1, 1 + 6)]
        Cn_tail_damp_q_value = mavrik_setup.Cn_tail_damp_q_val
        self.Cn_tail_damp_q_lookup_table = JaxNDInterpolator(Cn_tail_damp_q_breakpoints, Cn_tail_damp_q_value)

        Cn_tail_damp_r_breakpoints = [getattr(mavrik_setup, f'Cn_tail_damp_r_{i}') for i in range(1, 1 + 6)]
        Cn_tail_damp_r_value = mavrik_setup.Cn_tail_damp_r_val
        self.Cn_tail_damp_r_lookup_table = JaxNDInterpolator(Cn_tail_damp_r_breakpoints, Cn_tail_damp_r_value)

        Cn_wing_breakpoints = [getattr(mavrik_setup, f'Cn_wing_{i}') for i in range(1, 1 + 6)]
        Cn_wing_value = mavrik_setup.Cn_wing_val
        self.Cn_wing_lookup_table = JaxNDInterpolator(Cn_wing_breakpoints, Cn_wing_value)

        Cn_wing_damp_p_breakpoints = [getattr(mavrik_setup, f'Cn_wing_damp_p_{i}') for i in range(1, 1 + 6)]
        Cn_wing_damp_p_value = mavrik_setup.Cn_wing_damp_p_val
        self.Cn_wing_damp_p_lookup_table = JaxNDInterpolator(Cn_wing_damp_p_breakpoints, Cn_wing_damp_p_value)

        Cn_wing_damp_q_breakpoints = [getattr(mavrik_setup, f'Cn_wing_damp_q_{i}') for i in range(1, 1 + 6)]
        Cn_wing_damp_q_value = mavrik_setup.Cn_wing_damp_q_val
        self.Cn_wing_damp_q_lookup_table = JaxNDInterpolator(Cn_wing_damp_q_breakpoints, Cn_wing_damp_q_value)

        Cn_wing_damp_r_breakpoints = [getattr(mavrik_setup, f'Cn_wing_damp_r_{i}') for i in range(1, 1 + 6)]
        Cn_wing_damp_r_value = mavrik_setup.Cn_wing_damp_r_val
        self.Cn_wing_damp_r_lookup_table = JaxNDInterpolator(Cn_wing_damp_r_breakpoints, Cn_wing_damp_r_value)

        Cn_hover_fuse_breakpoints = [getattr(mavrik_setup, f'Cn_hover_fuse_{i}') for i in range(1, 1 + 3)]
        Cn_hover_fuse_value = mavrik_setup.Cn_hover_fuse_val
        self.Cn_hover_fuse_lookup_table = JaxNDInterpolator(Cn_hover_fuse_breakpoints, Cn_hover_fuse_value)
        
        
    def N(self, u: ActuatorOutput, wing_transform: FloatScalar, tail_transform: FloatScalar) -> Moments:
        Cn_Scale = 0.5744 * 2.8270 * u.Q
        Cn_Scale_p = 0.5744 * 2.8270**2 * 1.225 * 0.25 * u.U * u.p
        Cn_Scale_q = 0.5744 * 0.2032 * 2.8270 * 1.225 * 0.25 * u.U * u.q
        Cn_Scale_r = 0.5744 * 2.8270**2 * 1.225 * 0.25 * u.U * u.r
       
        Cn_aileron_wing = self.Cn_aileron_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron
        ]))
        Cn_aileron_wing_padded = jnp.array([0.0, 0.0, Cn_aileron_wing])
        Cn_aileron_wing_padded_transformed = jnp.dot(wing_transform, Cn_aileron_wing_padded * Cn_Scale)

        Cn_elevator_tail = self.Cn_elevator_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator
        ]))
        Cn_elevator_tail_padded = jnp.array([0.0, 0.0, Cn_elevator_tail])
        Cn_elevator_tail_padded_transformed = jnp.dot(tail_transform, Cn_elevator_tail_padded * Cn_Scale)

        Cn_flap_wing = self.Cn_flap_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap
        ]))
        Cn_flap_wing_padded = jnp.array([0.0, 0.0, Cn_flap_wing])
        Cn_flap_wing_padded_transformed = jnp.dot(wing_transform, Cn_flap_wing_padded * Cn_Scale)

        Cn_rudder_tail = self.Cn_rudder_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder
        ]))
        Cn_rudder_tail_padded = jnp.array([0.0, 0.0, Cn_rudder_tail])
        Cn_rudder_tail_padded_transformed = jnp.dot(tail_transform, Cn_rudder_tail_padded * Cn_Scale)
        # Tail
        Cn_tail = self.Cn_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        Cn_tail_padded = jnp.array([0.0, 0.0, Cn_tail])
        Cn_tail_padded_transformed = jnp.dot(tail_transform, Cn_tail_padded * Cn_Scale)

        # Tail Damp p
        Cn_tail_damp_p = self.Cn_tail_damp_p_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        Cn_tail_damp_p_padded = jnp.array([0.0, 0.0, Cn_tail_damp_p])
        Cn_tail_damp_p_padded_transformed = jnp.dot(tail_transform, Cn_tail_damp_p_padded * Cn_Scale_p)

        # Tail Damp q
        Cn_tail_damp_q = self.Cn_tail_damp_q_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        Cn_tail_damp_q_padded = jnp.array([0.0, 0.0, Cn_tail_damp_q])
        Cn_tail_damp_q_padded_transformed = jnp.dot(tail_transform, Cn_tail_damp_q_padded * Cn_Scale_q)

        # Tail Damp r
        Cn_tail_damp_r = self.Cn_tail_damp_r_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        Cn_tail_damp_r_padded = jnp.array([0.0, 0.0, Cn_tail_damp_r])
        Cn_tail_damp_r_padded_transformed = jnp.dot(tail_transform, Cn_tail_damp_r_padded * Cn_Scale_r)

        # Wing
        Cn_wing = self.Cn_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Cn_wing_padded = jnp.array([0.0, 0.0, Cn_wing])
        Cn_wing_padded_transformed = jnp.dot(wing_transform, Cn_wing_padded * Cn_Scale)

        # Wing Damp p
        Cn_wing_damp_p = self.Cn_wing_damp_p_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Cn_wing_damp_p_padded = jnp.array([0.0, 0.0, Cn_wing_damp_p])
        Cn_wing_damp_p_padded_transformed = jnp.dot(wing_transform, Cn_wing_damp_p_padded * Cn_Scale_p)

        # Wing Damp q
        Cn_wing_damp_q = self.Cn_wing_damp_q_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Cn_wing_damp_q_padded = jnp.array([0.0, 0.0, Cn_wing_damp_q])
        Cn_wing_damp_q_padded_transformed = jnp.dot(wing_transform, Cn_wing_damp_q_padded * Cn_Scale_q)

        # Wing Damp r
        Cn_wing_damp_r = self.Cn_wing_damp_r_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Cn_wing_damp_r_padded = jnp.array([0.0, 0.0, Cn_wing_damp_r])
        Cn_wing_damp_r_padded_transformed = jnp.dot(wing_transform, Cn_wing_damp_r_padded * Cn_Scale_r)

        # Hover Fuse
        Cn_hover_fuse = self.Cn_hover_fuse_lookup_table(jnp.array([
            u.U, u.alpha, u.beta
        ]))
        Cn_hover_fuse_padded = jnp.array([0.0, 0.0, Cn_hover_fuse * Cn_Scale]) 

        return Moments(
            Cn_aileron_wing_padded_transformed[0] + Cn_elevator_tail_padded_transformed[0] + Cn_flap_wing_padded_transformed[0] + Cn_rudder_tail_padded_transformed[0] +
            Cn_tail_padded_transformed[0] + Cn_tail_damp_p_padded_transformed[0] + Cn_tail_damp_q_padded_transformed[0] + Cn_tail_damp_r_padded_transformed[0] +
            Cn_wing_padded_transformed[0] + Cn_wing_damp_p_padded_transformed[0] + Cn_wing_damp_q_padded_transformed[0] + Cn_wing_damp_r_padded_transformed[0] +
            Cn_hover_fuse_padded[0],
            Cn_aileron_wing_padded_transformed[1] + Cn_elevator_tail_padded_transformed[1] + Cn_flap_wing_padded_transformed[1] + Cn_rudder_tail_padded_transformed[1] +
            Cn_tail_padded_transformed[1] + Cn_tail_damp_p_padded_transformed[1] + Cn_tail_damp_q_padded_transformed[1] + Cn_tail_damp_r_padded_transformed[1] +
            Cn_wing_padded_transformed[1] + Cn_wing_damp_p_padded_transformed[1] + Cn_wing_damp_q_padded_transformed[1] + Cn_wing_damp_r_padded_transformed[1] +
            Cn_hover_fuse_padded[1],
            Cn_aileron_wing_padded_transformed[2] + Cn_elevator_tail_padded_transformed[2] + Cn_flap_wing_padded_transformed[2] + Cn_rudder_tail_padded_transformed[2] +
            Cn_tail_padded_transformed[2] + Cn_tail_damp_p_padded_transformed[2] + Cn_tail_damp_q_padded_transformed[2] + Cn_tail_damp_r_padded_transformed[2] +
            Cn_wing_padded_transformed[2] + Cn_wing_damp_p_padded_transformed[2] + Cn_wing_damp_q_padded_transformed[2] + Cn_wing_damp_r_padded_transformed[2] +
            Cn_hover_fuse_padded[2]
        )
    

    def get_Ct_table(self, mavrik_setup: MavrikSetup):
        Ct_tail_left_breakpoints = [getattr(mavrik_setup, f'Ct_tail_left_{i}') for i in range(1, 1 + 4)]
        Ct_tail_left_value = mavrik_setup.Ct_tail_left_val
        self.Ct_tail_left_lookup_table = JaxNDInterpolator(Ct_tail_left_breakpoints, Ct_tail_left_value)

        Ct_tail_right_breakpoints = [getattr(mavrik_setup, f'Ct_tail_right_{i}') for i in range(1, 1 + 4)]
        Ct_tail_right_value = mavrik_setup.Ct_tail_right_val
        self.Ct_tail_right_lookup_table = JaxNDInterpolator(Ct_tail_right_breakpoints, Ct_tail_right_value)

        Ct_left_out1_breakpoints = [getattr(mavrik_setup, f'Ct_left_out_{i}') for i in range(1, 1 + 4)]
        Ct_left_out1_value = mavrik_setup.Ct_left_out_val
        self.Ct_left_out1_lookup_table = JaxNDInterpolator(Ct_left_out1_breakpoints, Ct_left_out1_value)

        Ct_left_2_breakpoints = [getattr(mavrik_setup, f'Ct_left_2_{i}') for i in range(1, 1 + 4)]
        Ct_left_2_value = mavrik_setup.Ct_left_2_val
        self.Ct_left_2_lookup_table = JaxNDInterpolator(Ct_left_2_breakpoints, Ct_left_2_value)

        Ct_left_3_breakpoints = [getattr(mavrik_setup, f'Ct_left_3_{i}') for i in range(1, 1 + 4)]
        Ct_left_3_value = mavrik_setup.Ct_left_3_val
        self.Ct_left_3_lookup_table = JaxNDInterpolator(Ct_left_3_breakpoints, Ct_left_3_value)

        Ct_left_4_breakpoints = [getattr(mavrik_setup, f'Ct_left_4_{i}') for i in range(1, 1 + 4)]
        Ct_left_4_value = mavrik_setup.Ct_left_4_val
        self.Ct_left_4_lookup_table = JaxNDInterpolator(Ct_left_4_breakpoints, Ct_left_4_value)

        Ct_left_5_breakpoints = [getattr(mavrik_setup, f'Ct_left_5_{i}') for i in range(1, 1 + 4)]
        Ct_left_5_value = mavrik_setup.Ct_left_5_val
        self.Ct_left_5_lookup_table = JaxNDInterpolator(Ct_left_5_breakpoints, Ct_left_5_value)

        Ct_left_6_in_breakpoints = [getattr(mavrik_setup, f'Ct_left_6_in_{i}') for i in range(1, 1 + 4)]
        Ct_left_6_in_value = mavrik_setup.Ct_left_6_in_val
        self.Ct_left_6_in_lookup_table = JaxNDInterpolator(Ct_left_6_in_breakpoints, Ct_left_6_in_value)

        Ct_right_7_in_breakpoints = [getattr(mavrik_setup, f'Ct_right_7_in_{i}') for i in range(1, 1 + 4)]
        Ct_right_7_in_value = mavrik_setup.Ct_right_7_in_val
        self.Ct_right_7_in_lookup_table = JaxNDInterpolator(Ct_right_7_in_breakpoints, Ct_right_7_in_value)

        Ct_right_8_breakpoints = [getattr(mavrik_setup, f'Ct_right_8_{i}') for i in range(1, 1 + 4)]
        Ct_right_8_value = mavrik_setup.Ct_right_8_val
        self.Ct_right_8_lookup_table = JaxNDInterpolator(Ct_right_8_breakpoints, Ct_right_8_value)

        Ct_right_9_breakpoints = [getattr(mavrik_setup, f'Ct_right_9_{i}') for i in range(1, 1 + 4)]
        Ct_right_9_value = mavrik_setup.Ct_right_9_val
        self.Ct_right_9_lookup_table = JaxNDInterpolator(Ct_right_9_breakpoints, Ct_right_9_value)

        Ct_right_10_breakpoints = [getattr(mavrik_setup, f'Ct_right_10_{i}') for i in range(1, 1 + 4)]
        Ct_right_10_value = mavrik_setup.Ct_right_10_val
        self.Ct_right_10_lookup_table = JaxNDInterpolator(Ct_right_10_breakpoints, Ct_right_10_value)

        Ct_right_11_breakpoints = [getattr(mavrik_setup, f'Ct_right_11_{i}') for i in range(1, 1 + 4)]
        Ct_right_11_value = mavrik_setup.Ct_right_11_val
        self.Ct_right_11_lookup_table = JaxNDInterpolator(Ct_right_11_breakpoints, Ct_right_11_value)

        Ct_right_12_out_breakpoints = [getattr(mavrik_setup, f'Ct_right_12_out_{i}') for i in range(1, 1 + 4)]
        Ct_right_12_out_value = mavrik_setup.Ct_right_12_out_val
        self.Ct_right_12_out_lookup_table = JaxNDInterpolator(Ct_right_12_out_breakpoints, Ct_right_12_out_value)

        self.RPM_tail_left_trans = mavrik_setup.RPM_tail_left_trans 
        self.RPM_tail_right_trans = mavrik_setup.RPM_tail_right_trans 
        self.RPM_left_out1_trans = mavrik_setup.RPM_left_out1_trans 
        self.RPM_left_2_trans = mavrik_setup.RPM_left_2_trans
        self.RPM_left_3_trans = mavrik_setup.RPM_left_3_trans
        self.RPM_left_4_trans = mavrik_setup.RPM_left_4_trans
        self.RPM_left_5_trans = mavrik_setup.RPM_left_5_trans
        self.RPM_left_6_in_trans = mavrik_setup.RPM_left_6_in_trans
        self.RPM_right_7_in_trans = mavrik_setup.RPM_right_7_in_trans
        self.RPM_right_8_trans = mavrik_setup.RPM_right_8_trans
        self.RPM_right_9_trans = mavrik_setup.RPM_right_9_trans
        self.RPM_right_10_trans = mavrik_setup.RPM_right_10_trans
        self.RPM_right_11_trans = mavrik_setup.RPM_right_11_trans
        self.RPM_right_12_out_trans = mavrik_setup.RPM_right_12_out_trans 
        
    def Ct(self, u: ActuatorOutput, wing_transform: FloatScalar, tail_transform: FloatScalar) -> Tuple[Forces, Moments]:
        Ct_tail_left = self.Ct_tail_left_lookup_table(jnp.array([
            u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        Ct_tail_left_padded = jnp.array([Ct_tail_left, 0., 0.])
        Ct_tail_left_transformed = jnp.dot(tail_transform, Ct_tail_left_padded * (1.225 * u.RPM_tailLeft**2 * 0.005059318992632 * 2.777777777777778e-4))

        Ct_tail_right = self.Ct_tail_right_lookup_table(jnp.array([
            u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        Ct_tail_right_padded = jnp.array([Ct_tail_right, 0., 0.])
        Ct_tail_right_transformed = jnp.dot(tail_transform, Ct_tail_right_padded * (1.225 * u.RPM_tailRight**2 * 0.005059318992632 * 2.777777777777778e-4))

        Ct_left_out1 = self.Ct_left_out1_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Ct_left_out1_padded = jnp.array([Ct_left_out1, 0., 0.])
        Ct_left_out1_transformed = jnp.dot(wing_transform, Ct_left_out1_padded * (1.225 * u.RPM_leftOut1**2 * 0.021071715921 * 2.777777777777778e-4))

        Ct_left_2 = self.Ct_left_2_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Ct_left_2_padded = jnp.array([Ct_left_2, 0., 0.])
        Ct_left_2_transformed = jnp.dot(wing_transform, Ct_left_2_padded * (1.225 * u.RPM_left2**2 * 0.021071715921 * 2.777777777777778e-4))

        Ct_left_3 = self.Ct_left_3_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Ct_left_3_padded = jnp.array([Ct_left_3, 0., 0.])
        Ct_left_3_transformed = jnp.dot(wing_transform, Ct_left_3_padded * (1.225 * u.RPM_left3**2 * 0.021071715921 * 2.777777777777778e-4))

        Ct_left_4 = self.Ct_left_4_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Ct_left_4_padded = jnp.array([Ct_left_4, 0., 0.])
        Ct_left_4_transformed = jnp.dot(wing_transform, Ct_left_4_padded * (1.225 * u.RPM_left4**2 * 0.021071715921 * 2.777777777777778e-4))

        Ct_left_5 = self.Ct_left_5_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Ct_left_5_padded = jnp.array([Ct_left_5, 0., 0.])
        Ct_left_5_transformed = jnp.dot(wing_transform, Ct_left_5_padded * (1.225 * u.RPM_left5**2 * 0.021071715921 * 2.777777777777778e-4))

        Ct_left_6_in = self.Ct_left_6_in_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Ct_left_6_in_padded = jnp.array([Ct_left_6_in, 0., 0.])
        Ct_left_6_in_transformed = jnp.dot(wing_transform, Ct_left_6_in_padded * (1.225 * u.RPM_left6In**2 * 0.021071715921 * 2.777777777777778e-4))

        Ct_right_7_in = self.Ct_right_7_in_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Ct_right_7_in_padded = jnp.array([Ct_right_7_in, 0., 0.])
        Ct_right_7_in_transformed = jnp.dot(wing_transform, Ct_right_7_in_padded * (1.225 * u.RPM_right7In**2 * 0.021071715921 * 2.777777777777778e-4))

        Ct_right_8 = self.Ct_right_8_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Ct_right_8_padded = jnp.array([Ct_right_8, 0., 0.])
        Ct_right_8_transformed = jnp.dot(wing_transform, Ct_right_8_padded * (1.225 * u.RPM_right8**2 * 0.021071715921 * 2.777777777777778e-4))

        Ct_right_9 = self.Ct_right_9_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Ct_right_9_padded = jnp.array([Ct_right_9, 0., 0.])
        Ct_right_9_transformed = jnp.dot(wing_transform, Ct_right_9_padded * (1.225 * u.RPM_right9**2 * 0.021071715921 * 2.777777777777778e-4))

        Ct_right_10 = self.Ct_right_10_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Ct_right_10_padded = jnp.array([Ct_right_10, 0., 0.])
        Ct_right_10_transformed = jnp.dot(wing_transform, Ct_right_10_padded * (1.225 * u.RPM_right10**2 * 0.021071715921 * 2.777777777777778e-4))

        Ct_right_11 = self.Ct_right_11_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Ct_right_11_padded = jnp.array([Ct_right_11, 0., 0.])
        Ct_right_11_transformed = jnp.dot(wing_transform, Ct_right_11_padded * (1.225 * u.RPM_right11**2 * 0.021071715921 * 2.777777777777778e-4))

        Ct_right_12_out = self.Ct_right_12_out_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Ct_right_12_out_padded = jnp.array([Ct_right_12_out, 0., 0.])
        Ct_right_12_out_transformed = jnp.dot(wing_transform, Ct_right_12_out_padded * (1.225 * u.RPM_right12Out**2 * 0.021071715921 * 2.777777777777778e-4))

        forces = Forces(
            Ct_tail_left_transformed[0] + Ct_tail_right_transformed[0] + Ct_left_out1_transformed[0] + Ct_left_2_transformed[0] + Ct_left_3_transformed[0] + 
            Ct_left_4_transformed[0] + Ct_left_5_transformed[0] + Ct_left_6_in_transformed[0] + Ct_right_7_in_transformed[0] + Ct_right_8_transformed[0] + 
            Ct_right_9_transformed[0] + Ct_right_10_transformed[0] + Ct_right_11_transformed[0] + Ct_right_12_out_transformed[0],
            Ct_tail_left_transformed[1] + Ct_tail_right_transformed[1] + Ct_left_out1_transformed[1] + Ct_left_2_transformed[1] + Ct_left_3_transformed[1] + 
            Ct_left_4_transformed[1] + Ct_left_5_transformed[1] + Ct_left_6_in_transformed[1] + Ct_right_7_in_transformed[1] + Ct_right_8_transformed[1] + 
            Ct_right_9_transformed[1] + Ct_right_10_transformed[1] + Ct_right_11_transformed[1] + Ct_right_12_out_transformed[1],
            Ct_tail_left_transformed[2] + Ct_tail_right_transformed[2] + Ct_left_out1_transformed[2] + Ct_left_2_transformed[2] + Ct_left_3_transformed[2] + 
            Ct_left_4_transformed[2] + Ct_left_5_transformed[2] + Ct_left_6_in_transformed[2] + Ct_right_7_in_transformed[2] + Ct_right_8_transformed[2] + 
            Ct_right_9_transformed[2] + Ct_right_10_transformed[2] + Ct_right_11_transformed[2] + Ct_right_12_out_transformed[2]
            )

        Ct_tail_left_transformed = jnp.cross(self.RPM_tail_left_trans, Ct_tail_left_transformed)
        Ct_tail_right_transformed = jnp.cross(self.RPM_tail_right_trans, Ct_tail_right_transformed)
        Ct_left_out1_transformed = jnp.cross(self.RPM_left_out1_trans, Ct_left_out1_transformed)
        Ct_left_2_transformed = jnp.cross(self.RPM_left_2_trans, Ct_left_2_transformed)
        Ct_left_3_transformed = jnp.cross(self.RPM_left_3_trans, Ct_left_3_transformed)
        Ct_left_4_transformed = jnp.cross(self.RPM_left_4_trans, Ct_left_4_transformed)
        Ct_left_5_transformed = jnp.cross(self.RPM_left_5_trans, Ct_left_5_transformed)
        Ct_left_6_in_transformed = jnp.cross(self.RPM_left_6_in_trans, Ct_left_6_in_transformed)
        Ct_right_7_in_transformed = jnp.cross(self.RPM_right_7_in_trans, Ct_right_7_in_transformed)
        Ct_right_8_transformed = jnp.cross(self.RPM_right_8_trans, Ct_right_8_transformed)
        Ct_right_9_transformed = jnp.cross(self.RPM_right_9_trans, Ct_right_9_transformed)
        Ct_right_10_transformed = jnp.cross(self.RPM_right_10_trans, Ct_right_10_transformed)
        Ct_right_11_transformed = jnp.cross(self.RPM_right_11_trans, Ct_right_11_transformed)
        Ct_right_12_out_transformed = jnp.cross(self.RPM_right_12_out_trans, Ct_right_12_out_transformed)
        

        moments = Moments(
            Ct_tail_left_transformed[0] + Ct_tail_right_transformed[0] + Ct_left_out1_transformed[0] + Ct_left_2_transformed[0] + Ct_left_3_transformed[0] + 
            Ct_left_4_transformed[0] + Ct_left_5_transformed[0] + Ct_left_6_in_transformed[0] + Ct_right_7_in_transformed[0] + Ct_right_8_transformed[0] + 
            Ct_right_9_transformed[0] + Ct_right_10_transformed[0] + Ct_right_11_transformed[0] + Ct_right_12_out_transformed[0],
            Ct_tail_left_transformed[1] + Ct_tail_right_transformed[1] + Ct_left_out1_transformed[1] + Ct_left_2_transformed[1] + Ct_left_3_transformed[1] + 
            Ct_left_4_transformed[1] + Ct_left_5_transformed[1] + Ct_left_6_in_transformed[1] + Ct_right_7_in_transformed[1] + Ct_right_8_transformed[1] + 
            Ct_right_9_transformed[1] + Ct_right_10_transformed[1] + Ct_right_11_transformed[1] + Ct_right_12_out_transformed[1],
            Ct_tail_left_transformed[2] + Ct_tail_right_transformed[2] + Ct_left_out1_transformed[2] + Ct_left_2_transformed[2] + Ct_left_3_transformed[2] + 
            Ct_left_4_transformed[2] + Ct_left_5_transformed[2] + Ct_left_6_in_transformed[2] + Ct_right_7_in_transformed[2] + Ct_right_8_transformed[2] + 
            Ct_right_9_transformed[2] + Ct_right_10_transformed[2] + Ct_right_11_transformed[2] + Ct_right_12_out_transformed[2]
            )
        
        return forces, moments
 
    def get_Kq_table(self, mavrik_setup: MavrikSetup):
        Kq_tail_left_breakpoints = [getattr(mavrik_setup, f'Kq_tail_left_{i}') for i in range(1, 1 + 4)]
        Kq_tail_left_value = mavrik_setup.Kq_tail_left_val
        self.Kq_tail_left_lookup_table = JaxNDInterpolator(Kq_tail_left_breakpoints, Kq_tail_left_value)

        Kq_tail_right_breakpoints = [getattr(mavrik_setup, f'Kq_tail_right_{i}') for i in range(1, 1 + 4)]
        Kq_tail_right_value = mavrik_setup.Kq_tail_right_val
        self.Kq_tail_right_lookup_table = JaxNDInterpolator(Kq_tail_right_breakpoints, Kq_tail_right_value)

        Kq_left_out_breakpoints = [getattr(mavrik_setup, f'Kq_left_out_{i}') for i in range(1, 1 + 4)]
        Kq_left_out_value = mavrik_setup.Kq_left_out_val
        self.Kq_left_out_lookup_table = JaxNDInterpolator(Kq_left_out_breakpoints, Kq_left_out_value)

        Kq_left_2_breakpoints = [getattr(mavrik_setup, f'Kq_left_2_{i}') for i in range(1, 1 + 4)]
        Kq_left_2_value = mavrik_setup.Kq_left_2_val
        self.Kq_left_2_lookup_table = JaxNDInterpolator(Kq_left_2_breakpoints, Kq_left_2_value)

        Kq_left_3_breakpoints = [getattr(mavrik_setup, f'Kq_left_3_{i}') for i in range(1, 1 + 4)]
        Kq_left_3_value = mavrik_setup.Kq_left_3_val
        self.Kq_left_3_lookup_table = JaxNDInterpolator(Kq_left_3_breakpoints, Kq_left_3_value)

        Kq_left_4_breakpoints = [getattr(mavrik_setup, f'Kq_left_4_{i}') for i in range(1, 1 + 4)]
        Kq_left_4_value = mavrik_setup.Kq_left_4_val
        self.Kq_left_4_lookup_table = JaxNDInterpolator(Kq_left_4_breakpoints, Kq_left_4_value)

        Kq_left_5_breakpoints = [getattr(mavrik_setup, f'Kq_left_5_{i}') for i in range(1, 1 + 4)]
        Kq_left_5_value = mavrik_setup.Kq_left_5_val
        self.Kq_left_5_lookup_table = JaxNDInterpolator(Kq_left_5_breakpoints, Kq_left_5_value)

        Kq_left_6_in_breakpoints = [getattr(mavrik_setup, f'Kq_left_6_in_{i}') for i in range(1, 1 + 4)]
        Kq_left_6_in_value = mavrik_setup.Kq_left_6_in_val
        self.Kq_left_6_in_lookup_table = JaxNDInterpolator(Kq_left_6_in_breakpoints, Kq_left_6_in_value)

        Kq_right_7_in_breakpoints = [getattr(mavrik_setup, f'Kq_right_7_in_{i}') for i in range(1, 1 + 4)]
        Kq_right_7_in_value = mavrik_setup.Kq_right_7_in_val
        self.Kq_right_7_in_lookup_table = JaxNDInterpolator(Kq_right_7_in_breakpoints, Kq_right_7_in_value)

        Kq_right_8_breakpoints = [getattr(mavrik_setup, f'Kq_right_8_{i}') for i in range(1, 1 + 4)]
        Kq_right_8_value = mavrik_setup.Kq_right_8_val
        self.Kq_right_8_lookup_table = JaxNDInterpolator(Kq_right_8_breakpoints, Kq_right_8_value)

        Kq_right_9_breakpoints = [getattr(mavrik_setup, f'Kq_right_9_{i}') for i in range(1, 1 + 4)]
        Kq_right_9_value = mavrik_setup.Kq_right_9_val
        self.Kq_right_9_lookup_table = JaxNDInterpolator(Kq_right_9_breakpoints, Kq_right_9_value)

        Kq_right_10_breakpoints = [getattr(mavrik_setup, f'Kq_right_10_{i}') for i in range(1, 1 + 4)]
        Kq_right_10_value = mavrik_setup.Kq_right_10_val
        self.Kq_right_10_lookup_table = JaxNDInterpolator(Kq_right_10_breakpoints, Kq_right_10_value)

        Kq_right_11_breakpoints = [getattr(mavrik_setup, f'Kq_right_11_{i}') for i in range(1, 1 + 4)]
        Kq_right_11_value = mavrik_setup.Kq_right_11_val
        self.Kq_right_11_lookup_table = JaxNDInterpolator(Kq_right_11_breakpoints, Kq_right_11_value)

        Kq_right_12_out_breakpoints = [getattr(mavrik_setup, f'Kq_right_12_out_{i}') for i in range(1, 1 + 4)]
        Kq_right_12_out_value = mavrik_setup.Kq_right_12_out_val
        self.Kq_right_12_out_lookup_table = JaxNDInterpolator(Kq_right_12_out_breakpoints, Kq_right_12_out_value)

    def Kq(self, u: ActuatorOutput, wing_transform: FloatScalar, tail_transform: FloatScalar) -> Moments:
        Kq_tail_left = self.Kq_tail_left_lookup_table(jnp.array([
            u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        Kq_tail_left_padded = jnp.array([Kq_tail_left, 0., 0.])
        Kq_tail_left_transformed = jnp.dot(tail_transform, Kq_tail_left_padded * (-1.225 * u.RPM_tailLeft**2 * 0.001349320375335 * 2.777777777777778e-4))

        Kq_tail_right = self.Kq_tail_right_lookup_table(jnp.array([
            u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        Kq_tail_right_padded = jnp.array([Kq_tail_right, 0., 0.])
        Kq_tail_right_transformed = jnp.dot(tail_transform, Kq_tail_right_padded * (1.225 * u.RPM_tailRight**2 * 0.001349320375335 * 2.777777777777778e-4))

        Kq_left_out = self.Kq_left_out_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Kq_left_out_padded = jnp.array([Kq_left_out, 0., 0.])
        Kq_left_out_transformed = jnp.dot(wing_transform, Kq_left_out_padded * (1.225 * u.RPM_leftOut1**2 * 0.008028323765901 * 2.777777777777778e-4))

        Kq_left_2 = self.Kq_left_2_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Kq_left_2_padded = jnp.array([Kq_left_2, 0., 0.])
        Kq_left_2_transformed = jnp.dot(wing_transform, Kq_left_2_padded * (-1.225 * u.RPM_left2**2 * 0.008028323765901 * 2.777777777777778e-4))

        Kq_left_3 = self.Kq_left_3_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Kq_left_3_padded = jnp.array([Kq_left_3, 0., 0.])
        Kq_left_3_transformed = jnp.dot(wing_transform, Kq_left_3_padded * (1.225 * u.RPM_left3**2 * 0.008028323765901 * 2.777777777777778e-4))

        Kq_left_4 = self.Kq_left_4_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Kq_left_4_padded = jnp.array([Kq_left_4, 0., 0.])
        Kq_left_4_transformed = jnp.dot(wing_transform, Kq_left_4_padded * (-1.225 * u.RPM_left4**2 * 0.008028323765901 * 2.777777777777778e-4))

        Kq_left_5 = self.Kq_left_5_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Kq_left_5_padded = jnp.array([Kq_left_5, 0., 0.])
        Kq_left_5_transformed = jnp.dot(wing_transform, Kq_left_5_padded * (1.225 * u.RPM_left5**2 * 0.008028323765901 * 2.777777777777778e-4))

        Kq_left_6_in = self.Kq_left_6_in_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Kq_left_6_in_padded = jnp.array([Kq_left_6_in, 0., 0.])
        Kq_left_6_in_transformed = jnp.dot(wing_transform, Kq_left_6_in_padded * (-1.225 * u.RPM_left6In**2 * 0.008028323765901 * 2.777777777777778e-4))

        Kq_right_7_in = self.Kq_right_7_in_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Kq_right_7_in_padded = jnp.array([Kq_right_7_in, 0., 0.])
        Kq_right_7_in_transformed = jnp.dot(wing_transform, Kq_right_7_in_padded * (-1.225 * u.RPM_right7In**2 * 0.008028323765901 * 2.777777777777778e-4))

        Kq_right_8 = self.Kq_right_8_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Kq_right_8_padded = jnp.array([Kq_right_8, 0., 0.])
        Kq_right_8_transformed = jnp.dot(wing_transform, Kq_right_8_padded * (1.225 * u.RPM_right8**2 * 0.008028323765901 * 2.777777777777778e-4))

        Kq_right_9 = self.Kq_right_9_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Kq_right_9_padded = jnp.array([Kq_right_9, 0., 0.])
        Kq_right_9_transformed = jnp.dot(wing_transform, Kq_right_9_padded * (-1.225 * u.RPM_right9**2 * 0.008028323765901 * 2.777777777777778e-4))

        Kq_right_10 = self.Kq_right_10_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Kq_right_10_padded = jnp.array([Kq_right_10, 0., 0.])
        Kq_right_10_transformed = jnp.dot(wing_transform, Kq_right_10_padded * (1.225 * u.RPM_right10**2 * 0.008028323765901 * 2.777777777777778e-4))

        Kq_right_11 = self.Kq_right_11_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Kq_right_11_padded = jnp.array([Kq_right_11, 0., 0.])
        Kq_right_11_transformed = jnp.dot(wing_transform, Kq_right_11_padded * (-1.225 * u.RPM_right11**2 * 0.008028323765901 * 2.777777777777778e-4))

        Kq_right_12_out = self.Kq_right_12_out_lookup_table(jnp.array([
            u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        Kq_right_12_out_padded = jnp.array([Kq_right_12_out, 0., 0.])
        Kq_right_12_out_transformed = jnp.dot(wing_transform, Kq_right_12_out_padded * (1.225 * u.RPM_right12Out**2 * 0.008028323765901 * 2.777777777777778e-4))

        return Moments(
            Kq_tail_left_transformed[0] + Kq_tail_right_transformed[0] + Kq_left_out_transformed[0] + Kq_left_2_transformed[0] + Kq_left_3_transformed[0] + 
            Kq_left_4_transformed[0] + Kq_left_5_transformed[0] + Kq_left_6_in_transformed[0] + Kq_right_7_in_transformed[0] + Kq_right_8_transformed[0] + 
            Kq_right_9_transformed[0] + Kq_right_10_transformed[0] + Kq_right_11_transformed[0] + Kq_right_12_out_transformed[0],
            Kq_tail_left_transformed[1] + Kq_tail_right_transformed[1] + Kq_left_out_transformed[1] + Kq_left_2_transformed[1] + Kq_left_3_transformed[1] + 
            Kq_left_4_transformed[1] + Kq_left_5_transformed[1] + Kq_left_6_in_transformed[1] + Kq_right_7_in_transformed[1] + Kq_right_8_transformed[1] + 
            Kq_right_9_transformed[1] + Kq_right_10_transformed[1] + Kq_right_11_transformed[1] + Kq_right_12_out_transformed[1],
            Kq_tail_left_transformed[2] + Kq_tail_right_transformed[2] + Kq_left_out_transformed[2] + Kq_left_2_transformed[2] + Kq_left_3_transformed[2] + 
            Kq_left_4_transformed[2] + Kq_left_5_transformed[2] + Kq_left_6_in_transformed[2] + Kq_right_7_in_transformed[2] + Kq_right_8_transformed[2] + 
            Kq_right_9_transformed[2] + Kq_right_10_transformed[2] + Kq_right_11_transformed[2] + Kq_right_12_out_transformed[2]
            )
         


if __name__ == "__main__":
    # Example usage of MavrikAero class

    # Initialize MavrikSetup with appropriate values
    mavrik_setup = MavrikSetup(file_path="/Users/weichaozhou/Workspace/Mavrik_JAX/jax_mavrik/aero_export.mat")

    # Initialize MavrikAero with mass and setup
    mavrik_aero = MavrikAero(mavrik_setup=mavrik_setup)

    # Define initial state variables
    state = StateVariables(
        u=29.9269, v=0.0, w=2.0927,
        Xe=0.0, Ye=0.0, Ze=0.0,
        roll=0.0, pitch=0.069813, yaw=0.0,
        VXe=30.0, VYe=0.0, VZe=0.0,
        p=0.0, q=0.0, r=0.0,
        Fx=0.0, Fy=0.0, Fz=0.0,
        L=0.0, M=0.0, N=0.0
    )

    # Define control inputs
    control = ControlInputs(
        wing_tilt=0.0, tail_tilt=0.0, aileron=0.0,
        elevator=0.0, flap=0.0, rudder=0.0,
        RPM_tailLeft=7500, RPM_tailRight=7500,
        RPM_leftOut1=7500, RPM_left2=7500,
        RPM_left3=7500, RPM_left4=7500,
        RPM_left5=7500, RPM_left6In=7500,
        RPM_right7In=7500, RPM_right8=7500,
        RPM_right9=7500, RPM_right10=7500,
        RPM_right11=7500, RPM_right12Out=7500
    )

    # Calculate forces and moments
    forces, moments, actuator_outputs = mavrik_aero(state, control)

    # Print the results
    print("Forces:", forces)
    print("Moments:", moments)
    print("Actuator Outputs:", actuator_outputs)