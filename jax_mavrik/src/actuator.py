from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from jax import jit
from jax_mavrik.src.utils.jax_types import FloatScalar
from jax import vmap

class ActuatorInput(NamedTuple):
    wing_tilt: FloatScalar
    tail_tilt: FloatScalar
    aileron: FloatScalar
    elevator: FloatScalar
    flap: FloatScalar
    rudder: FloatScalar
    RPM_tailLeft: FloatScalar
    RPM_tailRight: FloatScalar
    RPM_leftOut1: FloatScalar
    RPM_left2: FloatScalar
    RPM_left3: FloatScalar
    RPM_left4: FloatScalar
    RPM_left5: FloatScalar
    RPM_left6In: FloatScalar
    RPM_right7In: FloatScalar
    RPM_right8: FloatScalar
    RPM_right9: FloatScalar
    RPM_right10: FloatScalar
    RPM_right11: FloatScalar
    RPM_right12Out: FloatScalar

class ActuatorInutState(NamedTuple):
    U: FloatScalar
    alpha: FloatScalar
    beta: FloatScalar
    p: FloatScalar
    q: FloatScalar
    r: FloatScalar
    rho: FloatScalar = 1.225

    
    @staticmethod
    def from_input(u: np.ndarray) -> 'ActuatorInutState':
        U = np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
        alpha = np.arctan2(u[2], u[0])
        beta = np.arctan2(u[1], np.sqrt(u[0]**2 + u[2]**2))
        p = u[3]
        q = u[4]
        r = u[5]
        return ActuatorInutState(U=U, alpha=alpha, beta=beta, p=p, q=q, r=r)
    
class ActuatorOutput(NamedTuple):
    U: FloatScalar
    alpha: FloatScalar
    beta: FloatScalar
    p: FloatScalar
    q: FloatScalar
    r: FloatScalar
    wing_alpha: FloatScalar
    wing_beta: FloatScalar
    wing_RPM: FloatScalar
    left_alpha: FloatScalar
    right_alpha: FloatScalar
    left_beta: FloatScalar
    right_beta: FloatScalar
    wing_prop_alpha: FloatScalar
    wing_prop_beta: FloatScalar
    tail_alpha: FloatScalar
    tail_beta: FloatScalar
    tail_RPM: FloatScalar
    tailLeft_alpha: FloatScalar
    tailRight_alpha: FloatScalar
    tailLeft_beta: FloatScalar
    tailRight_beta: FloatScalar
    tail_prop_alpha: FloatScalar
    tail_prop_beta: FloatScalar
    Q: FloatScalar
    aileron: FloatScalar
    elevator: FloatScalar
    flap: FloatScalar
    rudder: FloatScalar
    wing_tilt: FloatScalar
    tail_tilt: FloatScalar
    RPM_tailLeft: FloatScalar
    RPM_tailRight: FloatScalar
    RPM_leftOut1: FloatScalar
    RPM_left2: FloatScalar
    RPM_left3: FloatScalar
    RPM_left4: FloatScalar
    RPM_left5: FloatScalar
    RPM_left6In: FloatScalar
    RPM_right7In: FloatScalar
    RPM_right8: FloatScalar
    RPM_right9: FloatScalar
    RPM_right10: FloatScalar
    RPM_right11: FloatScalar
    RPM_right12Out: FloatScalar

def actuate(state: ActuatorInutState, actuators: ActuatorInput) -> ActuatorOutput:
    # Calculate alpha/beta for local tables
    wing_alpha: float = state.alpha + actuators.wing_tilt
    wing_beta: float = state.beta
    wing_RPM: float = (1 / 12) * (actuators.RPM_leftOut1 + actuators.RPM_left2 + actuators.RPM_left3 + actuators.RPM_left4 + actuators.RPM_left5 + actuators.RPM_left6In +
                                  actuators.RPM_right7In + actuators.RPM_right8 + actuators.RPM_right9 + actuators.RPM_right10 + actuators.RPM_right11 + actuators.RPM_right12Out)
    left_alpha: float = state.alpha + actuators.wing_tilt
    right_alpha: float = state.alpha + actuators.wing_tilt
    left_beta: float = state.beta
    right_beta: float = state.beta
    wing_prop_alpha: float = (1 / 12) * (left_alpha + right_alpha)
    wing_prop_beta: float = (1 / 12) * (left_beta + right_beta)
    tail_alpha: float = state.alpha + actuators.tail_tilt
    tail_beta: float = state.beta
    tail_RPM: float = 0.5 * (actuators.RPM_tailRight + actuators.RPM_tailLeft)
    tailLeft_alpha: float = state.alpha + actuators.tail_tilt
    tailRight_alpha: float = state.alpha + actuators.tail_tilt
    tailLeft_beta: float = state.beta
    tailRight_beta: float = state.beta
    tail_prop_alpha: float = 0.5 * (tailLeft_alpha + tailRight_alpha)
    tail_prop_beta: float = 0.5 * (tailLeft_beta + tailRight_beta)
    Q: float = 0.5 * state.rho * state.U**2

    return ActuatorOutput(
        U=state.U,
        alpha=state.alpha,
        beta=state.beta,
        p=state.p,
        q=state.q,
        r=state.r,
        wing_alpha=wing_alpha,
        wing_beta=wing_beta,
        wing_RPM=wing_RPM,
        left_alpha=left_alpha,
        right_alpha=right_alpha,
        left_beta=left_beta,
        right_beta=right_beta,
        wing_prop_alpha=wing_prop_alpha,
        wing_prop_beta=wing_prop_beta,
        tail_alpha=tail_alpha,
        tail_beta=tail_beta,
        tail_RPM=tail_RPM,
        tailLeft_alpha=tailLeft_alpha,
        tailLeft_beta=tailLeft_beta,
        tailRight_alpha=tailRight_alpha,
        tailRight_beta=tailRight_beta,
        tail_prop_alpha=tail_prop_alpha,
        tail_prop_beta=tail_prop_beta,
        Q=Q,
        aileron=actuators.aileron,
        elevator=actuators.elevator,
        flap=actuators.flap,
        rudder=actuators.rudder,
        wing_tilt=actuators.wing_tilt,
        tail_tilt=actuators.tail_tilt,
        RPM_tailLeft=actuators.RPM_tailLeft,
        RPM_tailRight=actuators.RPM_tailRight,
        RPM_leftOut1=actuators.RPM_leftOut1,
        RPM_left2=actuators.RPM_left2,
        RPM_left3=actuators.RPM_left3,
        RPM_left4=actuators.RPM_left4,
        RPM_left5=actuators.RPM_left5,
        RPM_left6In=actuators.RPM_left6In,
        RPM_right7In=actuators.RPM_right7In,
        RPM_right8=actuators.RPM_right8,
        RPM_right9=actuators.RPM_right9,
        RPM_right10=actuators.RPM_right10,
        RPM_right11=actuators.RPM_right11,
        RPM_right12Out=actuators.RPM_right12Out
    )

if __name__ == '__main__':

    # Example usage with arrays
    input_states = ActuatorInutState(
        U=np.array([100.0, 110.0]), alpha=np.array([0.1, 0.2]), beta=np.array([0.05, 0.06]),
        p=np.array([0.01, 0.02]), q=np.array([0.02, 0.03]), r=np.array([0.03, 0.04]), rho=np.array([1.225, 1.226])
    )

    # Example usage with from_input method
    # Vectorize the from_input method
    input_states = vmap(ActuatorInutState.from_input, in_axes=(0,))(np.array([
        [100.0, 10.0, 5.0, 0.01, 0.02, 0.03],
        [110.0, 11.0, 6.0, 0.02, 0.03, 0.04]
    ])) 

    actuator_inputs = ActuatorInput(
        wing_tilt=np.array([0.1, 0.2]), tail_tilt=np.array([0.1, 0.2]), aileron=np.array([0.1, 0.2]),
        elevator=np.array([0.1, 0.2]), flap=np.array([0.1, 0.2]), rudder=np.array([0.1, 0.2]),
        RPM_tailLeft=np.array([1000.0, 1100.0]), RPM_tailRight=np.array([1000.0, 1100.0]),
        RPM_leftOut1=np.array([1000.0, 1100.0]), RPM_left2=np.array([1000.0, 1100.0]),
        RPM_left3=np.array([1000.0, 1100.0]), RPM_left4=np.array([1000.0, 1100.0]),
        RPM_left5=np.array([1000.0, 1100.0]), RPM_left6In=np.array([1000.0, 1100.0]),
        RPM_right7In=np.array([1000.0, 1100.0]), RPM_right8=np.array([1000.0, 1100.0]),
        RPM_right9=np.array([1000.0, 1100.0]), RPM_right10=np.array([1000.0, 1100.0]),
        RPM_right11=np.array([1000.0, 1100.0]), RPM_right12Out=np.array([1000.0, 1100.0])
    )

    # JIT compile the actuate function
    actuator_outputs = jit(actuate)(input_states, actuator_inputs)
    print(actuator_outputs)