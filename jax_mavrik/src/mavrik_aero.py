# mavrik_aero.py
import functools as ft


from jax_mavrik.mavrik_types import StateVariables, ControlInputs, AeroState, Forces, Moments
from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.src.actuator import ActuatorInutState, ActuatorInput, ActuatorOutput, actuate

import jax.numpy as jnp
from jax import jit
from jax import vmap

from typing import Tuple, List


@jit
def linear_interpolate(v0, v1, weight):
    return v0 * (1 - weight) + v1 * weight


class JaxNDInterpolator:
    def __init__(self, breakpoints: List[jnp.ndarray], values: jnp.ndarray):
        """
        Initialize the n-D interpolator.

        Args:
            breakpoints (list of jnp.ndarray): Each array contains the breakpoints for one dimension.
            values (jnp.ndarray): The values at each grid point with shape matching the breakpoints.
        """
        self.breakpoints = breakpoints
        self.values = values
        self.ndim = len(breakpoints)  # Number of dimensions

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Perform n-dimensional interpolation using vectorized JAX operations.

        Args:
            inputs (jnp.ndarray): The input coordinates at which to interpolate.

        Returns:
            jnp.ndarray: Interpolated value.
        """
        indices = []
        weights = []

        # For each dimension, calculate the index and interpolation weight
        for i in range(self.ndim):
            bp = self.breakpoints[i]
            value = inputs[i]
            
            # Find the interval in breakpoints where the input value lies
            idx = jnp.clip(jnp.searchsorted(bp, value) - 1, 0, len(bp) - 2)
            weight = (value - bp[idx]) / (bp[idx + 1] - bp[idx])

            indices.append(idx)
            weights.append(weight)

        # Generate all corner combinations for interpolation
        corner_indices = jnp.array(jnp.meshgrid(*[[0, 1]] * self.ndim, indexing="ij")).reshape(self.ndim, -1).T

        # Gather values from each corner of the interpolation "hypercube"
        interpolated_value = 0.0
        for corner in corner_indices:
            corner_idx = [indices[dim] + corner[dim] for dim in range(self.ndim)]
            corner_value = self.values[tuple(corner_idx)]
            
            # Compute the weight for this corner
            corner_weight = jnp.prod(
                [weights[dim] if corner[dim] else (1 - weights[dim]) for dim in range(self.ndim)]
            )
            
            # Accumulate weighted value
            interpolated_value += corner_value * corner_weight

        return interpolated_value
    

class MavrikAero:
    def __init__(self, mass: float, mavrik_setup: MavrikSetup):
        self.mass = mass
        self.mavrik_setup = mavrik_setup
        
    def __call__(self, state: StateVariables, control: ControlInputs) -> Tuple[Forces, Moments]:
        # Calculate forces and moments using Mavrik Aero model
        actuator_input_state = ActuatorInutState(
            U = jnp.sqrt(state.Vx**2 + state.Vy**2 + state.Vz**2),
            alpha = jnp.arctan2(state.Vz, state.Vx),
            beta = jnp.arctan2(state.Vy, jnp.sqrt(state.Vx**2 + state.Vz**2)),
            p = state.wx,
            q = state.wy,
            r = state.wz,
            phi = state.roll,
            theta = state.pitch,
            psi = state.yaw
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

        F0, M0 = self.Ct(actuator_outputs)

        F1 = self.Cx(actuator_outputs)
        F2 = self.Cy(actuator_outputs)
        F3 = self.Cz(actuator_outputs)

        M1 = self.L(actuator_outputs)
        M2 = self.M(actuator_outputs)
        M3 = self.N(actuator_outputs)
        M5 = self.Kq(actuator_outputs)

        Fx = F0.Fx + F1.Fx + F2.Fx + F3.Fx
        Fy = F0.Fy + F1.Fy + F2.Fy + F3.Fy
        Fz = F0.Fz + F1.Fz + F2.Fz + F3.Fz

        forces = Forces(Fx, Fy, Fz)
        moments_by_forces = jnp.cross(jnp.array([state.X, state.Y, state.Z]), jnp.array([forces.Fx, forces.Fy, forces.Fz]))[0]
       
        L = M0.L + M1.L + M2.L + M3.L + M5.L
        M = M0.M + M1.M + M2.M + M3.M + M5.M
        N = M0.N + M1.N + M2.N + M3.N + M5.N

        moments = Moments(L + moments_by_forces[0], M + moments_by_forces[1], N + moments_by_forces[2])

        return forces, moments



    def Cx(self, u: ActuatorOutput) -> Forces:
        CX_Scale = 0.5744 * u.Q
        wing_transform = jnp.array([[jnp.cos( u.wing_tilt), 0, jnp.sin( u.wing_tilt)], [0, 1, 0], [-jnp.sin(u.wing_tilt), 0., jnp.cos(u.wing_tilt)]]);
        tail_transform = jnp.array([[jnp.cos(u.tail_tilt), 0, jnp.sin(u.tail_tilt)], [0, 1, 0], [-jnp.sin(u.tail_tilt), 0., jnp.cos(u.tail_tilt)]])

        CX_aileron_wing_breakpoints = [getattr(self.mavrik_setup, f'CY_aileron_wing_{i}') for i in range(7)]
        CX_aileron_wing_value = self.mavrik_setup.Cl_aileron_wing_()
        CX_aileron_wing_lookup_table = JaxNDInterpolator(CX_aileron_wing_breakpoints, CX_aileron_wing_value)
        CX_aileron_wing = CX_aileron_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron
        ]))
        CX_aileron_wing_00 = jnp.concatenate([CX_aileron_wing, jnp.array([0.0, 0.0])])
        CX_aileron_wing_00_transformed = jnp.dot(wing_transform, CX_aileron_wing_00 * CX_Scale)


        CX_elevator_tail_breakpoints = [getattr(self.mavrik_setup, f'CY_elevator_tail_{i}') for i in range(7)]
        CX_elevator_tail_value = self.mavrik_setup.Cl_elevator_tail_val()
        CX_elevator_tail_lookup_table = JaxNDInterpolator(CX_elevator_tail_breakpoints, CX_elevator_tail_value)
        CX_elevator_tail = CX_elevator_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator
        ]))
        CX_elevator_tail_00 = jnp.concatenate([CX_elevator_tail, jnp.array([0.0, 0.0])])
        CX_elevator_tail_00_transformed = jnp.dot(tail_transform, CX_elevator_tail_00 * CX_Scale)


        CX_flap_wing_breakpoints = [getattr(self.mavrik_setup, f'CY_flap_wing_{i}') for i in range(7)]
        CX_flap_wing_value = self.mavrik_setup.Cl_flap_wing_val()
        CX_flap_wing_lookup_table = JaxNDInterpolator(CX_flap_wing_breakpoints, CX_flap_wing_value)
        CX_flap_wing = CX_flap_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap
        ]))
        CX_flap_wing_00 = jnp.concatenate([CX_flap_wing, jnp.array([0.0, 0.0])])
        Cx_flap_wing_00_Scaled_transformed = jnp.dot(wing_transform, CX_flap_wing_00 * CX_Scale)


        CX_rudder_tail_breakpoints = [getattr(self.mavrik_setup, f'CY_rudder_tail_{i}') for i in range(7)]
        CX_rudder_tail_value = self.mavrik_setup.Cl_rudder_tail_val()
        CX_rudder_tail_lookup_table = JaxNDInterpolator(CX_rudder_tail_breakpoints, CX_rudder_tail_value)
        CX_rudder_tail = CX_rudder_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder
        ]))
        CX_rudder_tail_00 = jnp.concatenate([CX_rudder_tail, jnp.array([0.0, 0.0])])
        CX_rudder_tail_00_transformed = jnp.dot(tail_transform, CX_rudder_tail_00 * CX_Scale)

        # Tail
        CX_tail_breakpoints = [getattr(self.mavrik_setup, f'CY_tail_{i}') for i in range(6)]
        CX_tail_value = self.mavrik_setup.Cl_tail_val()
        CX_tail_lookup_table = JaxNDInterpolator(CX_tail_breakpoints, CX_tail_value)
        CX_tail = CX_tail_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        CX_tail_00 = jnp.concatenate([CX_tail, jnp.array([0.0, 0.0])])
        CX_tail_00_transformed = jnp.dot(tail_transform, CX_tail_00 * CX_Scale)


        # Tail Damp p
        CX_tail_damp_p_breakpoints = [getattr(self.mavrik_setup, f'CY_tail_damp_p_{i}') for i in range(6)]
        CX_tail_damp_p_value = self.mavrik_setup.Cl_tail_damp_p_val()
        CX_tail_damp_p_lookup_table = JaxNDInterpolator(CX_tail_damp_p_breakpoints, CX_tail_damp_p_value)
        CX_tail_damp_p = CX_tail_damp_p_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        CX_tail_damp_p_00 = jnp.concatenate([CX_tail_damp_p, jnp.array([0.0, 0.0])])
        CX_tail_damp_p_00_transformed = jnp.dot(tail_transform, CX_tail_damp_p_00  * CX_Scale)


        # Tail Damp q
        CX_tail_damp_q_breakpoints = [getattr(self.mavrik_setup, f'CY_tail_damp_q_{i}') for i in range(6)]
        CX_tail_damp_q_value = self.mavrik_setup.Cl_tail_damp_q_val()
        CX_tail_damp_q_lookup_table = JaxNDInterpolator(CX_tail_damp_q_breakpoints, CX_tail_damp_q_value)
        CX_tail_damp_q = CX_tail_damp_q_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        CX_tail_damp_q_00 = jnp.concatenate([CX_tail_damp_q, jnp.array([0.0, 0.0])])
        CX_tail_damp_q_00_transformed = jnp.dot(tail_transform, CX_tail_damp_q_00 * CX_Scale)

        # Tail Damp r
        CX_tail_damp_r_breakpoints = [getattr(self.mavrik_setup, f'CY_tail_damp_r_{i}') for i in range(6)]
        CX_tail_damp_r_value = self.mavrik_setup.Cl_tail_damp_r_val()
        CX_tail_damp_r_lookup_table = JaxNDInterpolator(CX_tail_damp_r_breakpoints, CX_tail_damp_r_value)
        CX_tail_damp_r = CX_tail_damp_r_lookup_table(jnp.array([
            u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
        ]))
        CX_tail_damp_r_00 = jnp.concatenate([CX_tail_damp_r, jnp.array([0.0, 0.0])])
        CX_tail_damp_r_00_transformed = jnp.dot(tail_transform, CX_tail_damp_r_00 * CX_Scale)

        # Wing
        CX_wing_breakpoints = [getattr(self.mavrik_setup, f'CY_wing_{i}') for i in range(6)]
        CX_wing_value = self.mavrik_setup.Cl_wing_val()
        CX_wing_lookup_table = JaxNDInterpolator(CX_wing_breakpoints, CX_wing_value)
        CX_wing = CX_wing_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        CX_wing_00 = jnp.concatenate([CX_wing, jnp.array([0.0, 0.0])])
        CX_wing_00_transformed = jnp.dot(wing_transform, CX_wing_00 * CX_Scale)

        # Wing Damp p
        CX_wing_damp_p_breakpoints = [getattr(self.mavrik_setup, f'CY_wing_damp_p_{i}') for i in range(6)]
        CX_wing_damp_p_value = self.mavrik_setup.Cl_wing_damp_p_val()
        CX_wing_damp_p_lookup_table = JaxNDInterpolator(CX_wing_damp_p_breakpoints, CX_wing_damp_p_value)
        CX_wing_damp_p = CX_wing_damp_p_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        CX_wing_damp_p_00 = jnp.concatenate([CX_wing_damp_p, jnp.array([0.0, 0.0])])
        CX_wing_damp_p_00_transformed = jnp.dot(wing_transform, CX_wing_damp_p_00 * CX_Scale)

        # Wing Damp q
        CX_wing_damp_q_breakpoints = [getattr(self.mavrik_setup, f'CY_wing_damp_q_{i}') for i in range(6)]
        CX_wing_damp_q_value = self.mavrik_setup.Cl_wing_damp_q_val()
        CX_wing_damp_q_lookup_table = JaxNDInterpolator(CX_wing_damp_q_breakpoints, CX_wing_damp_q_value)
        CX_wing_damp_q = CX_wing_damp_q_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        CX_wing_damp_q_00 = jnp.concatenate([CX_wing_damp_q, jnp.array([0.0, 0.0])])
        CX_wing_damp_q_00_transformed = jnp.dot(wing_transform, CX_wing_damp_q_00 * CX_Scale)

        # Wing Damp r
        CX_wing_damp_r_breakpoints = [getattr(self.mavrik_setup, f'CY_wing_damp_r_{i}') for i in range(6)]
        CX_wing_damp_r_value = self.mavrik_setup.Cl_wing_damp_r_val()
        CX_wing_damp_r_lookup_table = JaxNDInterpolator(CX_wing_damp_r_breakpoints, CX_wing_damp_r_value)
        CX_wing_damp_r = CX_wing_damp_r_lookup_table(jnp.array([
            u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
        ]))
        CX_wing_damp_r_00 = jnp.concatenate([CX_wing_damp_r, jnp.array([0.0, 0.0])])
        CX_wing_damp_r_00_transformed = jnp.dot(wing_transform, CX_wing_damp_r_00 * CX_Scale)

        # Hover Fuse
        CX_hover_fuse_breakpoints = [getattr(self.mavrik_setup, f'CY_hover_fuse_{i}') for i in range(3)]
        CX_hover_fuse_value = self.mavrik_setup.Cl_hover_fuse_val()
        CX_hover_fuse_lookup_table = JaxNDInterpolator(CX_hover_fuse_breakpoints, CX_hover_fuse_value)
        CX_hover_fuse = CX_hover_fuse_lookup_table(jnp.array([
            u.U, u.alpha, u.beta
        ]))
        CX_hover_fuse_00 = jnp.concatenate([CX_hover_fuse, jnp.array([0.0, 0.0])])
         
        return Forces(
            CX_aileron_wing_00_transformed[0] + CX_elevator_tail_00_transformed[0] + Cx_flap_wing_00_Scaled_transformed[0] + CX_rudder_tail_00_transformed[0] +
            CX_tail_00_transformed[0] + CX_tail_damp_p_00_transformed[0] + CX_tail_damp_q_00_transformed[0] + CX_tail_damp_r_00_transformed[0] +
            CX_wing_00_transformed[0] + CX_wing_damp_p_00_transformed[0] + CX_wing_damp_q_00_transformed[0] + CX_wing_damp_r_00_transformed[0] +
            CX_hover_fuse_00[0], 
            CX_aileron_wing_00_transformed[1] + CX_elevator_tail_00_transformed[1] + Cx_flap_wing_00_Scaled_transformed[1] + CX_rudder_tail_00_transformed[1] +
            CX_tail_00_transformed[1] + CX_tail_damp_p_00_transformed[1] + CX_tail_damp_q_00_transformed[1] + CX_tail_damp_r_00_transformed[1] +
            CX_wing_00_transformed[1] + CX_wing_damp_p_00_transformed[1] + CX_wing_damp_q_00_transformed[1] + CX_wing_damp_r_00_transformed[1] +
            CX_hover_fuse_00[1], 
            CX_aileron_wing_00_transformed[2] + CX_elevator_tail_00_transformed[2] + Cx_flap_wing_00_Scaled_transformed[2] + CX_rudder_tail_00_transformed[2] +
            CX_tail_00_transformed[2] + CX_tail_damp_p_00_transformed[2] + CX_tail_damp_q_00_transformed[2] + CX_tail_damp_r_00_transformed[2] +
            CX_wing_00_transformed[2] + CX_wing_damp_p_00_transformed[2] + CX_wing_damp_q_00_transformed[2] + CX_wing_damp_r_00_transformed[2] +
            CX_hover_fuse_00[2]
        )
  
    def Cy(self, u: ActuatorOutput) -> Forces:
        raise NotImplementedError
    
    def Cz(self, u: ActuatorOutput) -> Forces:
        raise NotImplementedError
    
    def L(self, u: ActuatorOutput) -> Moments:
        raise NotImplementedError
    
    def M(self, u: ActuatorOutput) -> Moments:
        raise NotImplementedError

    def N(self, u: ActuatorOutput) -> Moments:
        raise NotImplementedError
    
    def Ct(self, u: ActuatorOutput) -> Tuple[Forces, Moments]:
        raise NotImplementedError
    
    def Kq(self, u: ActuatorOutput) -> Moments:
        raise NotImplementedError