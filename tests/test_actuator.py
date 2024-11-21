
import pytest

import os
import sys

from jax_mavrik.src.mavrik_aero import ActuatorInutState, ActuatorInput, ActuatorOutput, actuate

from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.mavrik_types import StateVariables, ControlInputs

from jax_mavrik.src.utils.mat_tools import euler_to_dcm

import jax.numpy as jnp

from .test_mavrik_aero import vned_values, xned_values, euler_values, vb_values, pqr_values, expected_actuator_outputs_values, control_inputs

 
expected_actuator_vb_values = jnp.array([
    [29.9269, 0, 2.0927], [29.8923, 0.0011, 1.9667], [29.8605, 0.0046, 1.8037],
    [29.8309, 0.0103, 1.6083], [29.8028, 0.0183, 1.3853], [29.7755, 0.0285, 1.1393],
    [29.7484, 0.0411, 0.8750], [29.7208, 0.0560, 0.5969], [29.6923, 0.0732, 0.3092],
    [29.6625, 0.0929, 0.0163], [29.6311, 0.1150, -0.2781]
])

expected_R_values = jnp.array([
    [0.997564050259824, 0, -0.0697564737441253, 0, 1, 0, 0.0697564737441253, 0, 0.997564050259824],
    [0.997618543741988, -3.82190337974867e-05, -0.0689727462221143, 3.8372682291554e-05, 0.999999999263361, 9.02757411463207e-07, 0.0689727461368038, -3.54727681173338e-06, 0.99761854447367],
    [0.997772728892127, -0.000153210072062912, -0.0667050073520098, 0.000153821813565908, 0.999999988161285, 4.0347860884043e-06, 0.0667050059441383, -1.42864847307365e-05, 0.997772740646831],
    [0.998008215871425, -0.000345634488166369, -0.0630831323725712, 0.000346983887392833, 0.999999939746639, 1.04354845354693e-05, 0.0630831249647371, -3.2303529802549e-05, 0.998008275667675],
    [0.998302190878637, -0.000616396048634007, -0.0582439330988611, 0.000618709429892039, 0.999999808364172, 2.16855009758966e-05, 0.0582439085703797, -5.76847537768049e-05, 0.998302380938218],
    [0.998629424943671, -0.000966634429662038, -0.0523291243434447, 0.00097004812514636, 0.999999528709719, 3.98368463475195e-05, 0.0523290611735698, -9.05440159195021e-05, 0.998629891981245],
    [0.998964117400584, -0.00139772535325718, -0.0454833871859625, 0.00140223948357805, 0.999999014594243, 6.73422178015324e-05, 0.0454832482404458, -0.00013105106052893, 0.998965092961269],
    [0.99928150857961, -0.00191128733726218, -0.0378525242430893, 0.00191671062506308, 0.999998157385549, 0.000106985452306516, 0.0378522500155409, -0.000179460919579114, 0.999283330673808],
    [0.999559216105274, -0.00250919480762391, -0.0295817078686759, 0.00251508220987494, 0.999996824083651, 0.000161814981510342, 0.0295812078943349, -0.000236144083270791, 0.99956235242004],
    [0.999778268173431, -0.0031935971120001, -0.0208138277503381, 0.00319918299898061, 0.999994854969047, 0.000235082057240932, 0.020812969905171, -0.000301617175949578, 0.999783341184882],
    [0.999923825321829, -0.00396702700061673, -0.0116878676642502, 0.00397115719783872, 0.999992060411721, 0.000330187872793571, 0.0116864650031864, -0.000376577080640898, 0.999931640026173]
])

 
assert expected_actuator_outputs_values.shape == (11, 45)
assert expected_actuator_vb_values.shape == (11, 3)
 
@pytest.mark.parametrize(
    "id, vned, xned, euler, vb, pqr, expected_R_values, expected_actuator_vb_values, expected_actuator_outputs_values",
    zip(
        list(range(11)),
        vned_values, xned_values, euler_values, vb_values, pqr_values, 
        expected_R_values, expected_actuator_vb_values, expected_actuator_outputs_values 
    )
)

def test_actuator(id, control_inputs, vned, xned, euler, vb, pqr, \
                     expected_R_values, expected_actuator_vb_values, expected_actuator_outputs_values):
    state = StateVariables(
        u=vb[0], v=vb[1], w=vb[2],
        Xe=xned[0], Ye=xned[1], Ze=xned[2],
        roll=euler[0], pitch=euler[1], yaw=euler[2],
        VXe=vned[0], VYe=vned[1], VZe=vned[2],
        p=pqr[0], q=pqr[1], r=pqr[2], 
        Fx = 0, Fy = 0, Fz = 0,
        L = 0, M = 0, N = 0
    )
    control = control_inputs
    print(f">>>>>>>>>>>>>>>>>>>> Test ID: {id} <<<<<<<<<<<<<<<<<<<<<<")


    # Calculate forces and moments using Mavrik Aero model
    # Transform body frame velocities (u, v, w) to inertial frame velocities (Vz, Vy, Vx)
    R = euler_to_dcm(state.roll, state.pitch, state.yaw)
    
    R_close = jnp.allclose(R.flatten(), expected_R_values, atol=0.001)
    print('R close???', R_close)
    if not R_close:
        print(f"\n  Expected: {expected_R_values}\n  Got: {R.flatten()}")


    
    # Body frame velocities
    body_velocities = jnp.array([state.VXe, state.VYe, state.VZe])
    #print(body_velocities)
    # Inertial frame velocities
    inertial_velocities = R @ body_velocities
    u, v, w = inertial_velocities

    actuator_vb_close = jnp.allclose(inertial_velocities, expected_actuator_vb_values, atol=0.001)
    print('actuator vb (uvw) Outputs close???', actuator_vb_close)
    if not actuator_vb_close:
        print(f"\n  Expected: {expected_actuator_vb_values}\n  Got: {inertial_velocities}")
 
     
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
    
    actuator_outputs_array = jnp.array([
        actuator_outputs.U, actuator_outputs.alpha, actuator_outputs.beta,
        actuator_outputs.p, actuator_outputs.q, actuator_outputs.r,
        actuator_outputs.wing_alpha, actuator_outputs.wing_beta, actuator_outputs.wing_RPM,
        actuator_outputs.left_alpha, actuator_outputs.right_alpha,
        actuator_outputs.left_beta, actuator_outputs.right_beta,
        actuator_outputs.wing_prop_alpha, actuator_outputs.wing_prop_beta,
        actuator_outputs.tail_alpha, actuator_outputs.tail_beta, actuator_outputs.tail_RPM,
        actuator_outputs.tailLeft_alpha, actuator_outputs.tailRight_alpha,
        actuator_outputs.tailLeft_beta, actuator_outputs.tailRight_beta,
        actuator_outputs.tail_prop_alpha, actuator_outputs.tail_prop_beta,
        actuator_outputs.Q, actuator_outputs.aileron, actuator_outputs.elevator,
        actuator_outputs.flap, actuator_outputs.rudder, actuator_outputs.wing_tilt,
        actuator_outputs.tail_tilt, actuator_outputs.RPM_tailLeft, actuator_outputs.RPM_tailRight,
        actuator_outputs.RPM_leftOut1, actuator_outputs.RPM_left2, actuator_outputs.RPM_left3,
        actuator_outputs.RPM_left4, actuator_outputs.RPM_left5, actuator_outputs.RPM_left6In,
        actuator_outputs.RPM_right7In, actuator_outputs.RPM_right8, actuator_outputs.RPM_right9,
        actuator_outputs.RPM_right10, actuator_outputs.RPM_right11, actuator_outputs.RPM_right12Out
    ])

    
    actuator_close = jnp.allclose(actuator_outputs_array, expected_actuator_outputs_values, atol=0.001)
    print('Actuator Outputs close???', actuator_close)
    if not actuator_close:
        print(f"\n  Expected: {expected_actuator_outputs_values}\n  Got: {actuator_outputs_array}")
        max_diff_index = jnp.argmax(jnp.abs(actuator_outputs_array - expected_actuator_outputs_values))
        print(f"\n  Max difference at index {max_diff_index}: Expected {expected_actuator_outputs_values[max_diff_index]}, Got {actuator_outputs_array[max_diff_index]}\n\n")
     
 