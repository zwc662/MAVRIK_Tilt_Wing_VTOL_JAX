
import pytest

import os
import sys

from jax_mavrik.src.mavrik_aero import MavrikAero, Cy, interpolate_nd, CY_LOOKUP_TABLES, RPM_TRANSFORMS

from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.mavrik_types import StateVariables, ControlInputs, Forces
from jax_mavrik.src.actuator import ActuatorOutput 

from jax_mavrik.src.utils.mat_tools import euler_to_dcm

import jax.numpy as jnp

from .test_mavrik_aero import mavrik_aero, expected_actuator_outputs_values as actuator_outputs_values, expected_Cy_outputs_values as expected_CY_outputs_values


expected_CY_alieron_wing_values = jnp.zeros([11])
expected_CY_elevator_tail_values = jnp.zeros([11])
expected_CY_flap_wing_values = jnp.zeros([11])
expected_CY_ruder_tail_values = jnp.zeros([11])
expected_CY_tail_values = jnp.array([
    4.85722573273506e-18, -4.90857242800799e-05, -0.000196133125002159, -0.000441027566660364,
    -0.000783969348094822, -0.00122546837445442, -0.00176634575785855, -0.00240774360983464,
    -0.00315114348806775, -0.00399839324796852, -0.00495184708598806
])
expected_CY_tail_damp_p_values = jnp.array([
    0.039085708734669, 0.0359384277886069, 0.0318334639049069, 0.0268870182275255,
    0.0212173506980488, 0.0149431911815976, 0.00818225642291378, 0.00104988313890477,
    -0.00634221384206797, -0.0138870511651818, -0.0214836725107944
])
expected_CY_tail_damp_q_values = jnp.array([
    -1.37045023477593e-11, 6.17677734266998e-05, 0.000233570517212275, 0.00048931257812345,
    0.000796584083274779, 0.00111838761719309, 0.00141480800708394, 0.00164459986609511,
    0.0017666674207698, 0.00174141266160428, 0.00153193724005878
])
expected_CY_tail_damp_r_values = jnp.array([
    1.30283125845563, 1.30013419634504, 1.29661644792414, 1.29237759175534,
    1.28751897068831, 1.28214233027425, 1.27634854787135, 1.27023646127338,
    1.26390180449709, 1.25743625692327, 1.25092633306522
])
expected_CY_wing_values = jnp.zeros([11])
expected_CY_wing_damp_p_values = jnp.zeros([11])
expected_CY_wing_damp_q_values = jnp.zeros([11])
expected_CY_wing_damp_r_values = jnp.zeros([11])
expected_CY_hover_fuse_values = jnp.array([
    2.71050543121376e-20, 1.5657298357323e-08, 6.31668654995293e-08, 1.4380919966996e-07,
    2.59413007864601e-07, 4.12233519323272e-07, 6.04838629215398e-07, 8.40004433528792e-07,
    1.12062145748568e-06, 1.4496126441746e-06, 1.82991019546717e-06
])

expected_CY_Scale_values = jnp.array([
    316.638, 315.729883945804, 314.845190058716, 313.988838147529, 313.164400714939,
    312.374104928985, 311.618896081704, 310.898550595885, 310.211826405883, 309.556638785613,
    308.930807904554
])

expected_CY_Scale_p_values = jnp.array([
    0.0, -0.11446510667421, -0.229402121005288, -0.345172422576589, -0.46212187051964, 
    -0.580587743826303, -0.70090610852723, -0.82341932343948, -0.948483433834421, 
    -1.07647523887386, -1.20818303955046
])


expected_CY_Scale_q_values = jnp.array([
    0.0, -0.16596442266699, -0.317940535744807, -0.455119391134645, -0.576940294487692, 
    -0.683074267479587, -0.773406673460475, -0.84801948103289, -0.907173485108433, 
    -0.951290675908402, -0.980963804822868
])

expected_CY_Scale_r_values = jnp.array([
    0.0, 0.0027518656446275, 0.00610265683220155, 0.0104814908383701, 0.0163052131445058,
    0.0239797646628719, 0.0339017233709519, 0.0464601114471251, 0.0620385210874145,
    0.081017576558698, 0.103787909352465
])
 
expected_wind_transform = jnp.repeat(jnp.diag(jnp.array([1., 1., 1.]))[None, :, :], 11, axis=0)
expected_tail_transform = jnp.repeat(jnp.diag(jnp.array([1., 1., 1.]))[None, :, :], 11, axis=0) 


 

@pytest.mark.parametrize(
    "id, actuator_outputs_values, \
        expected_CY_outputs_values, expected_CY_alieron_wing_values, expected_CY_elevator_tail_values, expected_CY_flap_wing_values, expected_CY_ruder_tail_values, expected_CY_tail_values, expected_CY_tail_damp_p_values, expected_CY_tail_damp_q_values, expected_CY_tail_damp_r_values, expected_CY_wing_values, expected_CY_wing_damp_p_values, expected_CY_wing_damp_q_values, expected_CY_wing_damp_r_values, expected_CY_hover_fuse_values, \
            expected_CY_Scale_values, expected_CY_Scale_p_values, expected_CY_Scale_q_values, expected_CY_Scale_r_values, \
                expected_wind_transform, expected_tail_transform",
    zip(
        list(range(11)), actuator_outputs_values, \
            expected_CY_outputs_values, expected_CY_alieron_wing_values, expected_CY_elevator_tail_values, expected_CY_flap_wing_values, expected_CY_ruder_tail_values, expected_CY_tail_values, expected_CY_tail_damp_p_values, expected_CY_tail_damp_q_values, expected_CY_tail_damp_r_values, expected_CY_wing_values, expected_CY_wing_damp_p_values, expected_CY_wing_damp_q_values, expected_CY_wing_damp_r_values, expected_CY_hover_fuse_values, \
            expected_CY_Scale_values, expected_CY_Scale_p_values, expected_CY_Scale_q_values, expected_CY_Scale_r_values,
            expected_wind_transform, expected_tail_transform
    )
)
 
def test_mavrik_aero(id, mavrik_aero, actuator_outputs_values, \
                     expected_CY_outputs_values, expected_CY_alieron_wing_values, expected_CY_elevator_tail_values, expected_CY_flap_wing_values, expected_CY_ruder_tail_values, expected_CY_tail_values, expected_CY_tail_damp_p_values, expected_CY_tail_damp_q_values, expected_CY_tail_damp_r_values, expected_CY_wing_values, expected_CY_wing_damp_p_values, expected_CY_wing_damp_q_values, expected_CY_wing_damp_r_values, expected_CY_hover_fuse_values, \
                     expected_CY_Scale_values, expected_CY_Scale_p_values, expected_CY_Scale_q_values, expected_CY_Scale_r_values,
                     expected_wind_transform, expected_tail_transform):
    u = ActuatorOutput(*actuator_outputs_values)
        
    print(f">>>>>>>>>>>>>>>>>>>> Test ID: {id} <<<<<<<<<<<<<<<<<<<<<<")
    

    wing_transform = jnp.array([[jnp.cos(u.wing_tilt), 0, jnp.sin( u.wing_tilt)], [0, 1, 0], [-jnp.sin(u.wing_tilt), 0., jnp.cos(u.wing_tilt)]]);
    tail_transform = jnp.array([[jnp.cos(u.tail_tilt), 0, jnp.sin(u.tail_tilt)], [0, 1, 0], [-jnp.sin(u.tail_tilt), 0., jnp.cos(u.tail_tilt)]])
    

    F2 = mavrik_aero.Cy(u, wing_transform, tail_transform)
    F2_array = jnp.array([F2.Fx, F2.Fy, F2.Fz])
    CY_outputs_values_close = jnp.allclose(F2_array, expected_CY_outputs_values, atol=0.001)
    print("CY_outputs_values_close???", CY_outputs_values_close)
    if not CY_outputs_values_close:
        print(f"\n  Expected: {expected_CY_outputs_values}\n  Got: {F2_array}")
        max_diff_index_CY_outputs_values = jnp.argmax(jnp.abs(F2_array - expected_CY_outputs_values))
        print(f"\n  Max difference in CY_outputs_values at index {max_diff_index_CY_outputs_values}: Expected {expected_CY_outputs_values[max_diff_index_CY_outputs_values]}, Got {F2_array[max_diff_index_CY_outputs_values]}")
 

    CY_Scale = 0.5744 * u.Q
    CY_Scale_r = 0.5744 * 2.8270 * 1.225 * 0.25 * u.U * u.r
    CY_Scale_p = 0.5744 * 2.8270 * 1.225 * 0.25 * u.U * u.p
    CY_Scale_q = 0.5744 * 0.2032 * 1.225 * 0.25 * u.U * u.q

    
    print("CY_Scale_close???", jnp.allclose(CY_Scale, expected_CY_Scale_values, atol=0.001))
    if not jnp.allclose(CY_Scale, expected_CY_Scale_values, atol=0.001):
        print(f"\n  Expected: {expected_CY_Scale_values}\n  Got: {CY_Scale}")
    print("CY_Scale_p_close???", jnp.allclose(CY_Scale_p, expected_CY_Scale_p_values, atol=0.001))
    if not jnp.allclose(CY_Scale_p, expected_CY_Scale_p_values, atol=0.001):
        print(f"\n  Expected: {expected_CY_Scale_p_values}\n  Got: {CY_Scale_p}") 
    print("CY_Scale_q_close???", jnp.allclose(CY_Scale_q, expected_CY_Scale_q_values, atol=0.001))
    if not jnp.allclose(CY_Scale_q, expected_CY_Scale_q_values, atol=0.001):
        print(f"\n  Expected: {expected_CY_Scale_q_values}\n  Got: {CY_Scale_q}")
    print("CY_Scale_r_close???", jnp.allclose(CY_Scale_r, expected_CY_Scale_r_values, atol=0.001))
    if not jnp.allclose(CY_Scale_r, expected_CY_Scale_r_values, atol=0.001):
        print(f"\n  Expected: {expected_CY_Scale_r_values}\n  Got: {CY_Scale_r}") 
    
    print("wing_transform_close???", jnp.allclose(wing_transform, expected_wind_transform, atol=0.001))
    if not jnp.allclose(wing_transform, expected_wind_transform, atol=0.001):
        print(f"\n  Expected: {expected_wind_transform}\n  Got: {wing_transform}")
        max_diff_index_wing_transform = jnp.argmax(jnp.abs(wing_transform - expected_wind_transform))
        print(f"\n  Max difference in wing_transform at index {max_diff_index_wing_transform}: Expected {expected_wind_transform[max_diff_index_wing_transform]}, Got {wing_transform[max_diff_index_wing_transform]}")

    print("tail_transform_close???", jnp.allclose(tail_transform, expected_tail_transform, atol=0.001))
    if not jnp.allclose(tail_transform, expected_tail_transform, atol=0.001):
        print(f"\n  Expected: {expected_tail_transform}\n  Got: {tail_transform}")
        max_diff_index_tail_transform = jnp.argmax(jnp.abs(tail_transform - expected_tail_transform))
        print(f"\n  Max difference in tail_transform at index {max_diff_index_tail_transform}: Expected {expected_tail_transform[max_diff_index_tail_transform]}, Got {tail_transform[max_diff_index_tail_transform]}")
   
    Cy_lookup_tables = mavrik_aero.Cy_lookup_tables

    CY_aileron_wing = interpolate_nd(
        jnp.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron]),
        breakpoints=Cy_lookup_tables.CY_aileron_wing_lookup_table.breakpoints,
        values=Cy_lookup_tables.CY_aileron_wing_lookup_table.values
    )
    CY_aileron_wing_padded = jnp.array([0.0, CY_aileron_wing, 0.0])
    CY_aileron_wing_padded_transformed = jnp.dot(wing_transform, CY_aileron_wing_padded * CY_Scale)

    CY_elevator_tail = interpolate_nd(
        jnp.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator]),
        breakpoints=Cy_lookup_tables.CY_elevator_tail_lookup_table.breakpoints,
        values=Cy_lookup_tables.CY_elevator_tail_lookup_table.values
    )
    CY_elevator_tail_padded = jnp.array([0.0, CY_elevator_tail, 0.0])
    CY_elevator_tail_padded_transformed = jnp.dot(tail_transform, CY_elevator_tail_padded * CY_Scale)

    CY_flap_wing = interpolate_nd(
        jnp.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap]),
        breakpoints=Cy_lookup_tables.CY_flap_wing_lookup_table.breakpoints,
        values=Cy_lookup_tables.CY_flap_wing_lookup_table.values
    )
    CY_flap_wing_padded = jnp.array([0.0, CY_flap_wing, 0.0])
    CY_flap_wing_padded_transformed = jnp.dot(wing_transform, CY_flap_wing_padded * CY_Scale)

    CY_rudder_tail = interpolate_nd(
        jnp.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder]),
        breakpoints=Cy_lookup_tables.CY_rudder_tail_lookup_table.breakpoints,
        values=Cy_lookup_tables.CY_rudder_tail_lookup_table.values
    )
    CY_rudder_tail_padded = jnp.array([0.0, CY_rudder_tail, 0.0])
    CY_rudder_tail_padded_transformed = jnp.dot(tail_transform, CY_rudder_tail_padded * CY_Scale)

    # Tail
    CY_tail = interpolate_nd(
        jnp.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
        breakpoints=Cy_lookup_tables.CY_tail_lookup_table.breakpoints,
        values=Cy_lookup_tables.CY_tail_lookup_table.values
    )
    CY_tail_padded = jnp.array([0.0, CY_tail, 0.0])
    CY_tail_padded_transformed = jnp.dot(tail_transform, CY_tail_padded * CY_Scale)

    # Tail Damp p
    CY_tail_damp_p = interpolate_nd(
        jnp.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
        breakpoints=Cy_lookup_tables.CY_tail_damp_p_lookup_table.breakpoints,
        values=Cy_lookup_tables.CY_tail_damp_p_lookup_table.values
    )
    CY_tail_damp_p_padded = jnp.array([0.0, CY_tail_damp_p, 0.0])
    CY_tail_damp_p_padded_transformed = jnp.dot(tail_transform, CY_tail_damp_p_padded * CY_Scale_p)

    # Tail Damp q
    CY_tail_damp_q = interpolate_nd(
        jnp.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
        breakpoints=Cy_lookup_tables.CY_tail_damp_q_lookup_table.breakpoints,
        values=Cy_lookup_tables.CY_tail_damp_q_lookup_table.values
    )
    CY_tail_damp_q_padded = jnp.array([0.0, CY_tail_damp_q, 0.0])
    CY_tail_damp_q_padded_transformed = jnp.dot(tail_transform, CY_tail_damp_q_padded * CY_Scale_q)

    # Tail Damp r
    CY_tail_damp_r = interpolate_nd(
        jnp.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
        breakpoints=Cy_lookup_tables.CY_tail_damp_r_lookup_table.breakpoints,
        values=Cy_lookup_tables.CY_tail_damp_r_lookup_table.values
    )
    CY_tail_damp_r_padded = jnp.array([0.0, CY_tail_damp_r, 0.0])
    CY_tail_damp_r_padded_transformed = jnp.dot(tail_transform, CY_tail_damp_r_padded * CY_Scale_r)

    # Wing
    CY_wing = interpolate_nd(
        jnp.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
        breakpoints=Cy_lookup_tables.CY_wing_lookup_table.breakpoints,
        values=Cy_lookup_tables.CY_wing_lookup_table.values
    )
    CY_wing_padded = jnp.array([0.0, CY_wing, 0.0])
    CY_wing_padded_transformed = jnp.dot(wing_transform, CY_wing_padded * CY_Scale)

    # Wing Damp p
    CY_wing_damp_p = interpolate_nd(
        jnp.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
        breakpoints=Cy_lookup_tables.CY_wing_damp_p_lookup_table.breakpoints,
        values=Cy_lookup_tables.CY_wing_damp_p_lookup_table.values
    )
    CY_wing_damp_p_padded = jnp.array([0.0, CY_wing_damp_p, 0.0])
    CY_wing_damp_p_padded_transformed = jnp.dot(wing_transform, CY_wing_damp_p_padded * CY_Scale_p)

    # Wing Damp q
    CY_wing_damp_q = interpolate_nd(
        jnp.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
        breakpoints=Cy_lookup_tables.CY_wing_damp_q_lookup_table.breakpoints,
        values=Cy_lookup_tables.CY_wing_damp_q_lookup_table.values
    )
    CY_wing_damp_q_padded = jnp.array([0.0, CY_wing_damp_q, 0.0])
    CY_wing_damp_q_padded_transformed = jnp.dot(wing_transform, CY_wing_damp_q_padded * CY_Scale_q)

    # Wing Damp r
    CY_wing_damp_r = interpolate_nd(
        jnp.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
        breakpoints=Cy_lookup_tables.CY_wing_damp_r_lookup_table.breakpoints,
        values=Cy_lookup_tables.CY_wing_damp_r_lookup_table.values
    )
    CY_wing_damp_r_padded = jnp.array([0.0, CY_wing_damp_r, 0.0])
    CY_wing_damp_r_padded_transformed = jnp.dot(wing_transform, CY_wing_damp_r_padded * CY_Scale_r)

    # Hover Fuse
    CY_hover_fuse = interpolate_nd(
        jnp.array([u.U, u.alpha, u.beta]),
        breakpoints=Cy_lookup_tables.CY_hover_fuse_lookup_table.breakpoints,
        values=Cy_lookup_tables.CY_hover_fuse_lookup_table.values
    )
    CY_hover_fuse_padded = jnp.array([0.0, CY_hover_fuse * CY_Scale, 0.0])
     
    CY_aileron_wing_close = jnp.allclose(CY_aileron_wing, expected_CY_alieron_wing_values, atol=0.001)
    print("CY_aileron_wing_close???", CY_aileron_wing_close)
    if not CY_aileron_wing_close:
        print(f"\n  Expected: {expected_CY_alieron_wing_values}\n  Got: {CY_aileron_wing}")
        max_diff_index_CY_aileron_wing = jnp.argmax(jnp.abs(CY_aileron_wing - expected_CY_alieron_wing_values))
        print(f"\n  Max difference in CY_aileron_wing at index {max_diff_index_CY_aileron_wing}: Expected {expected_CY_alieron_wing_values[max_diff_index_CY_aileron_wing]}, Got {CY_aileron_wing[max_diff_index_CY_aileron_wing]}")

    CY_elevator_tail_close = jnp.allclose(CY_elevator_tail, expected_CY_elevator_tail_values, atol=0.001)
    print("CY_elevator_tail_close???", CY_elevator_tail_close)
    if not CY_elevator_tail_close:
        print(f"\n  Expected: {expected_CY_elevator_tail_values}\n  Got: {CY_elevator_tail}")
        max_diff_index_CY_elevator_tail = jnp.argmax(jnp.abs(CY_elevator_tail - expected_CY_elevator_tail_values))
        print(f"\n  Max difference in CY_elevator_tail at index {max_diff_index_CY_elevator_tail}: Expected {expected_CY_elevator_tail_values[max_diff_index_CY_elevator_tail]}, Got {CY_elevator_tail[max_diff_index_CY_elevator_tail]}")

    CY_flap_wing_close = jnp.allclose(CY_flap_wing, expected_CY_flap_wing_values, atol=0.001)
    print("CY_flap_wing_close???", CY_flap_wing_close)
    if not CY_flap_wing_close:
        print(f"\n  Expected: {expected_CY_flap_wing_values}\n  Got: {CY_flap_wing}")
        max_diff_index_CY_flap_wing = jnp.argmax(jnp.abs(CY_flap_wing - expected_CY_flap_wing_values))
        print(f"\n  Max difference in CY_flap_wing at index {max_diff_index_CY_flap_wing}: Expected {expected_CY_flap_wing_values[max_diff_index_CY_flap_wing]}, Got {CY_flap_wing[max_diff_index_CY_flap_wing]}")

    CY_rudder_tail_close = jnp.allclose(CY_rudder_tail, expected_CY_ruder_tail_values, atol=0.001)
    print("CY_rudder_tail_close???", CY_rudder_tail_close)
    if not CY_rudder_tail_close:
        print(f"\n  Expected: {expected_CY_ruder_tail_values}\n  Got: {CY_rudder_tail}")
        max_diff_index_CY_rudder_tail = jnp.argmax(jnp.abs(CY_rudder_tail - expected_CY_ruder_tail_values))
        print(f"\n  Max difference in CY_rudder_tail at index {max_diff_index_CY_rudder_tail}: Expected {expected_CY_ruder_tail_values[max_diff_index_CY_rudder_tail]}, Got {CY_rudder_tail[max_diff_index_CY_rudder_tail]}")

    CY_tail_close = jnp.allclose(CY_tail, expected_CY_tail_values, atol=0.001)
    print("CY_tail_close???", CY_tail_close)
    if not CY_tail_close:
        print(f"\n  Expected: {expected_CY_tail_values}\n  Got: {CY_tail}")
        max_diff_index_CY_tail = jnp.argmax(jnp.abs(CY_tail - expected_CY_tail_values))
        print(f"\n  Max difference in CY_tail at index {max_diff_index_CY_tail}: Expected {expected_CY_tail_values[max_diff_index_CY_tail]}, Got {CY_tail[max_diff_index_CY_tail]}")

    CY_tail_damp_p_close = jnp.allclose(CY_tail_damp_p, expected_CY_tail_damp_p_values, atol=0.001)
    print("CY_tail_damp_p_close???", CY_tail_damp_p_close)
    if not CY_tail_damp_p_close:
        print(f"\n  Expected: {expected_CY_tail_damp_p_values}\n  Got: {CY_tail_damp_p}")
        max_diff_index_CY_tail_damp_p = jnp.argmax(jnp.abs(CY_tail_damp_p - expected_CY_tail_damp_p_values))
        print(f"\n  Max difference in CY_tail_damp_p at index {max_diff_index_CY_tail_damp_p}: Expected {expected_CY_tail_damp_p_values[max_diff_index_CY_tail_damp_p]}, Got {CY_tail_damp_p[max_diff_index_CY_tail_damp_p]}")

    CY_tail_damp_q_close = jnp.allclose(CY_tail_damp_q, expected_CY_tail_damp_q_values, atol=0.001)
    print("CY_tail_damp_q_close???", CY_tail_damp_q_close)
    if not CY_tail_damp_q_close:
        print(f"\n  Expected: {expected_CY_tail_damp_q_values}\n  Got: {CY_tail_damp_q}")
        max_diff_index_CY_tail_damp_q = jnp.argmax(jnp.abs(CY_tail_damp_q - expected_CY_tail_damp_q_values))
        print(f"\n  Max difference in CY_tail_damp_q at index {max_diff_index_CY_tail_damp_q}: Expected {expected_CY_tail_damp_q_values[max_diff_index_CY_tail_damp_q]}, Got {CY_tail_damp_q[max_diff_index_CY_tail_damp_q]}")

    CY_tail_damp_r_close = jnp.allclose(CY_tail_damp_r, expected_CY_tail_damp_r_values, atol=0.001)
    print("CY_tail_damp_r_close???", CY_tail_damp_r_close)
    if not CY_tail_damp_r_close:
        print(f"\n  Expected: {expected_CY_tail_damp_r_values}\n  Got: {CY_tail_damp_r}")
        max_diff_index_CY_tail_damp_r = jnp.argmax(jnp.abs(CY_tail_damp_r - expected_CY_tail_damp_r_values))
        print(f"\n  Max difference in CY_tail_damp_r at index {max_diff_index_CY_tail_damp_r}: Expected {expected_CY_tail_damp_r_values[max_diff_index_CY_tail_damp_r]}, Got {CY_tail_damp_r[max_diff_index_CY_tail_damp_r]}")

    CY_wing_close = jnp.allclose(CY_wing, expected_CY_wing_values, atol=0.001)
    print("CY_wing_close???", CY_wing_close)
    if not CY_wing_close:
        print(f"\n  Expected: {expected_CY_wing_values}\n  Got: {CY_wing}")
        max_diff_index_CY_wing = jnp.argmax(jnp.abs(CY_wing - expected_CY_wing_values))
        print(f"\n  Max difference in CY_wing at index {max_diff_index_CY_wing}: Expected {expected_CY_wing_values[max_diff_index_CY_wing]}, Got {CY_wing[max_diff_index_CY_wing]}")

    CY_wing_damp_p_close = jnp.allclose(CY_wing_damp_p, expected_CY_wing_damp_p_values, atol=0.001)
    print("CY_wing_damp_p_close???", CY_wing_damp_p_close)
    if not CY_wing_damp_p_close:
        print(f"\n  Expected: {expected_CY_wing_damp_p_values}\n  Got: {CY_wing_damp_p}")
        max_diff_index_CY_wing_damp_p = jnp.argmax(jnp.abs(CY_wing_damp_p - expected_CY_wing_damp_p_values))
        print(f"\n  Max difference in CY_wing_damp_p at index {max_diff_index_CY_wing_damp_p}: Expected {expected_CY_wing_damp_p_values[max_diff_index_CY_wing_damp_p]}, Got {CY_wing_damp_p[max_diff_index_CY_wing_damp_p]}")

    CY_wing_damp_q_close = jnp.allclose(CY_wing_damp_q, expected_CY_wing_damp_q_values, atol=0.001)
    print("CY_wing_damp_q_close???", CY_wing_damp_q_close)
    if not CY_wing_damp_q_close:
        print(f"\n  Expected: {expected_CY_wing_damp_q_values}\n  Got: {CY_wing_damp_q}")
        max_diff_index_CY_wing_damp_q = jnp.argmax(jnp.abs(CY_wing_damp_q - expected_CY_wing_damp_q_values))
        print(f"\n  Max difference in CY_wing_damp_q at index {max_diff_index_CY_wing_damp_q}: Expected {expected_CY_wing_damp_q_values[max_diff_index_CY_wing_damp_q]}, Got {CY_wing_damp_q[max_diff_index_CY_wing_damp_q]}")

    CY_wing_damp_r_close = jnp.allclose(CY_wing_damp_r, expected_CY_wing_damp_r_values, atol=0.001)
    print("CY_wing_damp_r_close???", CY_wing_damp_r_close)
    if not CY_wing_damp_r_close:
        print(f"\n  Expected: {expected_CY_wing_damp_r_values}\n  Got: {CY_wing_damp_r}")
        max_diff_index_CY_wing_damp_r = jnp.argmax(jnp.abs(CY_wing_damp_r - expected_CY_wing_damp_r_values))
        print(f"\n  Max difference in CY_wing_damp_r at index {max_diff_index_CY_wing_damp_r}: Expected {expected_CY_wing_damp_r_values[max_diff_index_CY_wing_damp_r]}, Got {CY_wing_damp_r[max_diff_index_CY_wing_damp_r]}")

    CY_hover_fuse_close = jnp.allclose(CY_hover_fuse, expected_CY_hover_fuse_values, atol=0.001)
    print("CY_hover_fuse_close???", CY_hover_fuse_close)
    if not CY_hover_fuse_close:
        print(f"\n  Expected: {expected_CY_hover_fuse_values}\n  Got: {CY_hover_fuse}")
        max_diff_index_CY_hover_fuse = jnp.argmax(jnp.abs(CY_hover_fuse - expected_CY_hover_fuse_values))
        print(f"\n  Max difference in CY_hover_fuse at index {max_diff_index_CY_hover_fuse}: Expected {expected_CY_hover_fuse_values[max_diff_index_CY_hover_fuse]}, Got {CY_hover_fuse[max_diff_index_CY_hover_fuse]}")
    
    F2_array = jnp.array([
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
    ])


    CY_outputs_values_close = jnp.allclose(F2_array, expected_CY_outputs_values, atol=0.001)
    print("CY_outputs_values_close???", CY_outputs_values_close)
    if not CY_outputs_values_close:
        print(f"\n  Expected: {expected_CY_outputs_values}\n  Got: {F2_array}")
        max_diff_index_CY_outputs_values = jnp.argmax(jnp.abs(F2_array - expected_CY_outputs_values))
        print(f"\n  Max difference in CY_outputs_values at index {max_diff_index_CY_outputs_values}: Expected {expected_CY_outputs_values[max_diff_index_CY_outputs_values]}, Got {F2_array[max_diff_index_CY_outputs_values]}")