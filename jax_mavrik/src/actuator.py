import jax.numpy as jnp
from jax import jit
from dataclasses import dataclass
from typing import List
 
@dataclass
class Actuator:
    wing_tilt: float
    tail_tilt: float
    aileron: float
    elevator: float
    flap: float
    rudder: float
    RPM_tailLeft: float
    RPM_tailRight: float
    RPM_leftOut1: float
    RPM_left2: float
    RPM_left3: float
    RPM_left4: float
    RPM_left5: float
    RPM_left6In: float
    RPM_right7In: float
    RPM_right8: float
    RPM_right9: float
    RPM_right10: float
    RPM_right11: float
    RPM_right12Out: float

    @staticmethod
    def from_array(actuators):
        return Actuator(
            wing_tilt=actuators[0],
            tail_tilt=actuators[1],
            aileron=actuators[2],
            elevator=actuators[3],
            flap=actuators[4],
            rudder=actuators[5],
            RPM_tailLeft=actuators[6],
            RPM_tailRight=actuators[7],
            RPM_leftOut1=actuators[8],
            RPM_left2=actuators[9],
            RPM_left3=actuators[10],
            RPM_left4=actuators[11],
            RPM_left5=actuators[12],
            RPM_left6In=actuators[13],
            RPM_right7In=actuators[14],
            RPM_right8=actuators[15],
            RPM_right9=actuators[16],
            RPM_right10=actuators[17],
            RPM_right11=actuators[18],
            RPM_right12Out=actuators[19]
        )

@dataclass
class ActuatorState:
    U: float
    alpha: float
    beta: float
    p: float
    q: float
    r: float
    wing_alpha: float
    wing_beta: float
    wing_RPM: float
    left_alpha: float
    right_alpha: float
    left_beta: float
    right_beta: float
    wing_prop_alpha: float
    wing_prop_beta: float
    tail_alpha: float
    tail_beta: float
    tail_RPM: float
    tailLeft_alpha: float
    tailRight_alpha: float
    tailLeft_beta: float
    tailRight_beta: float
    tail_prop_alpha: float
    tail_prop_beta: float
    Q: float
    aileron: float
    elevator: float
    flap: float
    rudder: float
    wing_tilt: float
    tail_tilt: float
    RPM_tailLeft: float
    RPM_tailRight: float
    RPM_leftOut1: float
    RPM_left2: float
    RPM_left3: float
    RPM_left4: float
    RPM_left5: float
    RPM_left6In: float
    RPM_right7In: float
    RPM_right8: float
    RPM_right9: float
    RPM_right10: float
    RPM_right11: float
    RPM_right12Out: float
    rho: float

    @staticmethod
    def from_array(y):
        return ActuatorState(
            U=y[0],
            alpha=y[1],
            beta=y[2],
            p=y[3],
            q=y[4],
            r=y[5],
            wing_alpha=y[6],
            wing_beta=y[7],
            wing_RPM=y[8],
            left_alpha=y[9],
            right_alpha=y[10],
            left_beta=y[11],
            right_beta=y[12],
            wing_prop_alpha=y[13],
            wing_prop_beta=y[14],
            tail_alpha=y[15],
            tail_beta=y[16],
            tail_RPM=y[17],
            tailLeft_alpha=y[18],
            tailRight_alpha=y[19],
            tailLeft_beta=y[20],
            tailRight_beta=y[21],
            tail_prop_alpha=y[22],
            tail_prop_beta=y[23],
            Q=y[24],
            aileron=y[25],
            elevator=y[26],
            flap=y[27],
            rudder=y[28],
            wing_tilt=y[29],
            tail_tilt=y[30],
            RPM_tailLeft=y[31],
            RPM_tailRight=y[32],
            RPM_leftOut1=y[33],
            RPM_left2=y[34],
            RPM_left3=y[35],
            RPM_left4=y[36],
            RPM_left5=y[37],
            RPM_left6In=y[38],
            RPM_right7In=y[39],
            RPM_right8=y[40],
            RPM_right9=y[41],
            RPM_right10=y[42],
            RPM_right11=y[43],
            RPM_right12Out=y[44],
            rho=y[45]
        )
    

@dataclass
class ActuatorOutput:
    U: float
    alpha: float
    beta: float
    p: float
    q: float
    r: float
    wing_alpha: float
    wing_beta: float
    wing_RPM: float
    left_alpha: float
    right_alpha: float
    left_beta: float
    right_beta: float
    wing_prop_alpha: float
    wing_prop_beta: float
    tail_alpha: float
    tail_beta: float
    tail_RPM: float
    tailLeft_alpha: float
    tailRight_alpha: float
    tailLeft_beta: float
    tailRight_beta: float
    tail_prop_alpha: float
    tail_prop_beta: float
    Q: float
    aileron: float
    elevator: float
    flap: float
    rudder: float
    wing_tilt: float
    tail_tilt: float
    RPM_tailLeft: float
    RPM_tailRight: float
    RPM_leftOut1: float
    RPM_left2: float
    RPM_left3: float
    RPM_left4: float
    RPM_left5: float
    RPM_left6In: float
    RPM_right7In: float
    RPM_right8: float
    RPM_right9: float
    RPM_right10: float
    RPM_right11: float
    RPM_right12Out: float



    @staticmethod
    def from_array(y):
        return ActuatorOutput(
            U=y[0],
            alpha=y[1],
            beta=y[2],
            p=y[3],
            q=y[4],
            r=y[5],
            wing_alpha=y[6],
            wing_beta=y[7],
            wing_RPM=y[8],
            left_alpha=y[9],
            right_alpha=y[10],
            left_beta=y[11],
            right_beta=y[12],
            wing_prop_alpha=y[13],
            wing_prop_beta=y[14],
            tail_alpha=y[15],
            tail_beta=y[16],
            tail_RPM=y[17],
            tailLeft_alpha=y[18],
            tailRight_alpha=y[19],
            tailLeft_beta=y[20],
            tailRight_beta=y[21],
            tail_prop_alpha=y[22],
            tail_prop_beta=y[23], 
            Q=y[24],
            aileron=y[25],
            elevator=y[26],
            flap=y[27],
            rudder=y[28],
            wing_tilt=y[29],
            tail_tilt=y[30],
            RPM_tailLeft=y[31],
            RPM_tailRight=y[32],
            RPM_leftOut1=y[33],
            RPM_left2=y[34],
            RPM_left3=y[35],
            RPM_left4=y[36],
            RPM_left5=y[37],
            RPM_left6In=y[38],
            RPM_right7In=y[39],
            RPM_right8=y[40],
            RPM_right9=y[41],
            RPM_right10=y[42],
            RPM_right11=y[43],
            RPM_right12Out=y[44]
        )

@jit
def fcn(state_array: jnp.ndarray, actuators_array: jnp.ndarray) -> ActuatorOutput:
    state = ActuatorState.from_array(state_array)
    actuators = Actuator.from_array(actuators_array)
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

    
    y: jnp.ndarray = jnp.array([state.U, state.alpha, state.beta, state.p, state.q, state.r, wing_alpha, wing_beta, wing_RPM, left_alpha, right_alpha, left_beta, right_beta,
                                wing_prop_alpha, wing_prop_beta, tail_alpha, tail_beta, tail_RPM, tailLeft_alpha, tailRight_alpha,
                                tailLeft_beta, tailRight_beta, tail_prop_alpha, tail_prop_beta, Q, actuators.aileron, actuators.elevator, actuators.flap, actuators.rudder,
                                actuators.wing_tilt, actuators.tail_tilt, actuators.RPM_tailLeft, actuators.RPM_tailRight, actuators.RPM_leftOut1, actuators.RPM_left2, actuators.RPM_left3, actuators.RPM_left4,
                                actuators.RPM_left5, actuators.RPM_left6In, actuators.RPM_right7In, actuators.RPM_right8, actuators.RPM_right9, actuators.RPM_right10, actuators.RPM_right11, actuators.RPM_right12Out])
    
    return ActuatorOutput.from_array(y)


if __name__ == "__main__":
    # Example state and actuator inputs
    
    state = ActuatorState(
        U=100.0, alpha=5.0, beta=2.0, p=0.1, q=0.2, r=0.3, wing_alpha=0.0, wing_beta=0.0, wing_RPM=0.0,
        tailRight_beta=0.0, tail_prop_alpha=0.0, tail_prop_beta=0.0, left_alpha=0.0, right_alpha=0.0, left_beta=0.0, right_beta=0.0, wing_prop_alpha=0.0, wing_prop_beta=0.0, Q=0.0, aileron=0.0, elevator=0.0, flap=0.0,
        tail_alpha=0.0, tail_beta=0.0, tail_RPM=0.0, tailLeft_alpha=0.0, tailRight_alpha=0.0, tailLeft_beta=0.0,
        rudder=0.0, wing_tilt=1.0, tail_tilt=1.0, RPM_tailLeft=1000.0, RPM_tailRight=1000.0, RPM_leftOut1=1000.0,
        RPM_left2=1000.0, RPM_left3=1000.0, RPM_left4=1000.0, RPM_left5=1000.0, RPM_left6In=1000.0, RPM_right7In=1000.0,
        RPM_right8=1000.0, RPM_right9=1000.0, RPM_right10=1000.0, RPM_right11=1000.0, RPM_right12Out=1000.0, rho=1.225
    )

    actuators = Actuator(
        wing_tilt=1.0, tail_tilt=1.0, aileron=0.1, elevator=0.2, flap=0.3, rudder=0.4, RPM_tailLeft=1000.0,
        RPM_tailRight=1000.0, RPM_leftOut1=1000.0, RPM_left2=1000.0, RPM_left3=1000.0, RPM_left4=1000.0, RPM_left5=1000.0,
        RPM_left6In=1000.0, RPM_right7In=1000.0, RPM_right8=1000.0, RPM_right9=1000.0, RPM_right10=1000.0,
        RPM_right11=1000.0, RPM_right12Out=1000.0
    )
    state_array = jnp.array([state.U, state.alpha, state.beta, state.p, state.q, state.r, state.wing_alpha, state.wing_beta, state.wing_RPM, state.left_alpha, state.right_alpha, state.left_beta, state.right_beta,
                             state.wing_prop_alpha, state.wing_prop_beta, state.tail_alpha, state.tail_beta, state.tail_RPM, state.tailLeft_alpha, state.tailRight_alpha,
                             state.tailLeft_beta, state.tailRight_beta, state.tail_prop_alpha, state.tail_prop_beta, state.Q, state.aileron, state.elevator, state.flap, state.rudder,
                             state.wing_tilt, state.tail_tilt, state.RPM_tailLeft, state.RPM_tailRight, state.RPM_leftOut1, state.RPM_left2, state.RPM_left3, state.RPM_left4, 
                             state.RPM_left5, state.RPM_left6In, state.RPM_right7In, state.RPM_right8, state.RPM_right9, state.RPM_right10, state.RPM_right11, state.RPM_right12Out])
    
    actuators_array = jnp.array([actuators.wing_tilt, actuators.tail_tilt, actuators.aileron, actuators.elevator, actuators.flap, actuators.rudder, actuators.RPM_tailLeft,
                                 actuators.RPM_tailRight, actuators.RPM_leftOut1, actuators.RPM_left2, actuators.RPM_left3, actuators.RPM_left4, actuators.RPM_left5,
                                 actuators.RPM_left6In, actuators.RPM_right7In, actuators.RPM_right8, actuators.RPM_right9, actuators.RPM_right10, actuators.RPM_right11, actuators.RPM_right12Out])
    
    output = fcn(state_array, actuators_array)
     
    # Print the output
    print(output)
            