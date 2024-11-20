
import pytest

import os
import sys

from jax_mavrik.src.mavrik_aero import MavrikAero

from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.mavrik_types import StateVariables, ControlInputs, Forces
from jax_mavrik.src.actuator import ActuatorOutput 

from jax_mavrik.src.utils.mat_tools import euler_to_dcm

import jax.numpy as jnp

from .test_mavrik_aero import expected_actuator_outputs_values as actuator_outputs_values, expected_Cz_outputs_values as expected_CZ_outputs_values


expected_CZ_alieron_wing_values = jnp.zeros([11])
expected_CZ_elevator_tail_values = jnp.zeros([11])
expected_CZ_flap_wing_values = jnp.zeros([11])
expected_CZ_ruder_tail_values = jnp.zeros([11])
expected_CZ_tail_values = jnp.array([
    -0.0983031998676459, -0.0928494715686514, -0.0857298137670177, -0.0771462267136948,
    -0.0673043211986089, -0.0564105545245675, -0.0446696493356847, -0.0322822134893826,
    -0.0194425767329141, -0.00633685701119404, 0.00679586595655535
])
expected_CZ_tail_damp_p_values = jnp.array([
    2.93872512845128e-08, 4.24020586992793e-05, 0.000169553998579096, 0.000381809050199371,
    0.00067987237414503, 0.00106479758155673, 0.00153796091600428, 0.00210104401357566,
    0.00275602606570379, 0.00350518555542699, 0.00435118141099692
])
expected_CZ_tail_damp_q_values = jnp.array([
    -15.2152299608785, -15.216085177439, -15.2172088063061, -15.218568462886,
    -15.2201311510576, -15.2218637067614, -15.2237332137208, -15.2257073881476,
    -15.2277549296496, -15.2298458360193, -15.2318701574886
])
expected_CZ_tail_damp_r_values = jnp.array([
    2.32062322256972e-07, 2.47500282899995e-06, 8.23450726453566e-06, 1.56276642356999e-05,
    2.23004226364384e-05, 2.55527638986203e-05, 2.24594425745047e-05, 9.98440478165311e-06,
    -1.49129795685734e-05, -5.51815930795918e-05, -0.000113597965003634
])
expected_CZ_wing_values = jnp.array([
    -0.738276930070935, -0.716811907838301, -0.688730877215672, -0.654834363541563,
    -0.615937504036622, -0.572859081141687, -0.526411267728718, -0.477390158953044,
    -0.426567156001053, -0.3746812553226, -0.322542239712922
])
expected_CZ_wing_damp_p_values = jnp.array([
    0.000581384298128804, 0.000649453994904211, 0.000899058789927153, 0.00133133867506664,
    0.00194823929816894, 0.00275245133609409, 0.00374736301516206, 0.00493703026212144,
    0.00632616721675468, 0.00792015812697629, 0.00972474046939384
])
expected_CZ_wing_damp_q_values = jnp.array([
    -34.8203898117822, -34.8189229296317, -34.8170484052042, -34.8148167773859,
    -34.8122793575576, -34.8094875339084, -34.8064921285648, -34.8033428112388,
    -34.8000875723789, -34.7967722579726, -34.7933869853641
])
expected_CZ_wing_damp_r_values = jnp.array([
    0.00843460325395249, 0.00851576863627858, 0.00858989581744224, 0.00865701460342469,
    0.00871731619320841, 0.00877114484245488, 0.00881898315658857, 0.00886143242647348,
    0.00889918945491791, 0.00893302128925923, 0.00902414572047538
])
expected_CZ_hover_fuse_values = jnp.array([
    1.6940658945086e-21, 9.22333490188002e-10, 3.15340005077091e-09, 5.66832780519565e-09,
    7.20785524220984e-09, 6.34848138983108e-09, 1.56771482257802e-09, -8.69602902611033e-09,
    -2.59892584035238e-08, -5.17952917164802e-08, -8.74956162983197e-08
])

expected_CZ_Scale_values = jnp.array([
    316.638, 315.729883945804, 314.845190058716, 313.988838147529, 313.164400714939,
    312.374104928985, 311.618896081704, 310.898550595885, 310.211826405883, 309.556638785613,
    308.930807904554
])

expected_CZ_Scale_p_values = jnp.array([
    0.0, -0.11446510667421, -0.229402121005288, -0.345172422576589, -0.46212187051964, 
    -0.580587743826303, -0.70090610852723, -0.82341932343948, -0.948483433834421, 
    -1.07647523887386, -1.20818303955046
])


expected_CZ_Scale_q_values = jnp.array([
    0.0, -0.16596442266699, -0.317940535744807, -0.455119391134645, -0.576940294487692, 
    -0.683074267479587, -0.773406673460475, -0.84801948103289, -0.907173485108433, 
    -0.951290675908402, -0.980963804822868
])

expected_CZ_Scale_r_values = jnp.array([
    0.0, 0.0027518656446275, 0.00610265683220155, 0.0104814908383701, 0.0163052131445058,
    0.0239797646628719, 0.0339017233709519, 0.0464601114471251, 0.0620385210874145,
    0.081017576558698, 0.103787909352465
])
 
expected_wind_transform = jnp.repeat(jnp.diag(jnp.array([1., 1., 1.]))[None, :, :], 11, axis=0)
expected_tail_transform = jnp.repeat(jnp.diag(jnp.array([1., 1., 1.]))[None, :, :], 11, axis=0) 



@pytest.fixture
def mavrik_aero():
    mavrik_setup = MavrikSetup(file_path=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "jax_mavrik/aero_export.mat")
    )
    return MavrikAero(mavrik_setup=mavrik_setup)


@pytest.mark.parametrize(
    "id, actuator_outputs_values, \
        expected_CZ_outputs_values, expected_CZ_alieron_wing_values, expected_CZ_elevator_tail_values, expected_CZ_flap_wing_values, expected_CZ_ruder_tail_values, expected_CZ_tail_values, expected_CZ_tail_damp_p_values, expected_CZ_tail_damp_q_values, expected_CZ_tail_damp_r_values, expected_CZ_wing_values, expected_CZ_wing_damp_p_values, expected_CZ_wing_damp_q_values, expected_CZ_wing_damp_r_values, expected_CZ_hover_fuse_values, \
            expected_CZ_Scale_values, expected_CZ_Scale_p_values, expected_CZ_Scale_q_values, expected_CZ_Scale_r_values, \
                expected_wind_transform, expected_tail_transform",
    zip(
        list(range(11)), actuator_outputs_values, \
            expected_CZ_outputs_values, expected_CZ_alieron_wing_values, expected_CZ_elevator_tail_values, expected_CZ_flap_wing_values, expected_CZ_ruder_tail_values, expected_CZ_tail_values, expected_CZ_tail_damp_p_values, expected_CZ_tail_damp_q_values, expected_CZ_tail_damp_r_values, expected_CZ_wing_values, expected_CZ_wing_damp_p_values, expected_CZ_wing_damp_q_values, expected_CZ_wing_damp_r_values, expected_CZ_hover_fuse_values, \
            expected_CZ_Scale_values, expected_CZ_Scale_p_values, expected_CZ_Scale_q_values, expected_CZ_Scale_r_values,
            expected_wind_transform, expected_tail_transform
    )
)
 
def test_mavrik_aero(id, mavrik_aero, actuator_outputs_values, \
                     expected_CZ_outputs_values, expected_CZ_alieron_wing_values, expected_CZ_elevator_tail_values, expected_CZ_flap_wing_values, expected_CZ_ruder_tail_values, expected_CZ_tail_values, expected_CZ_tail_damp_p_values, expected_CZ_tail_damp_q_values, expected_CZ_tail_damp_r_values, expected_CZ_wing_values, expected_CZ_wing_damp_p_values, expected_CZ_wing_damp_q_values, expected_CZ_wing_damp_r_values, expected_CZ_hover_fuse_values, \
                     expected_CZ_Scale_values, expected_CZ_Scale_p_values, expected_CZ_Scale_q_values, expected_CZ_Scale_r_values,
                     expected_wind_transform, expected_tail_transform):
    u = ActuatorOutput(*actuator_outputs_values)
        
    print(f">>>>>>>>>>>>>>>>>>>> Test ID: {id} <<<<<<<<<<<<<<<<<<<<<<")
    
    F3 = mavrik_aero.Cz(u)
    F3_array = jnp.array([F3.Fx, F3.Fy, F3.Fz])
    CZ_outputs_values_close = jnp.allclose(F3_array, expected_CZ_outputs_values, atol=0.001)
    print("CZ_outputs_values_close???", CZ_outputs_values_close)
    if not CZ_outputs_values_close:
        print(f"\n  Expected: {expected_CZ_outputs_values}\n  Got: {F3_array}")
        max_diff_index_CZ_outputs_values = jnp.argmax(jnp.abs(F3_array - expected_CZ_outputs_values))
        print(f"\n  Max difference in CZ_outputs_values at index {max_diff_index_CZ_outputs_values}: Expected {expected_CZ_outputs_values[max_diff_index_CZ_outputs_values]}, Got {F3_array[max_diff_index_CZ_outputs_values]}")
 

    CZ_Scale = 0.5744 * u.Q
    CZ_Scale_r = 0.5744 * 2.8270 * 1.225 * 0.25 * u.U * u.r
    CZ_Scale_p = 0.5744 * 2.8270 * 1.225 * 0.25 * u.U * u.p
    CZ_Scale_q = 0.5744 * 0.2032 * 1.225 * 0.25 * u.U * u.q

    
    print("CZ_Scale_close???", jnp.allclose(CZ_Scale, expected_CZ_Scale_values, atol=0.001))
    if not jnp.allclose(CZ_Scale, expected_CZ_Scale_values, atol=0.001):
        print(f"\n  Expected: {expected_CZ_Scale_values}\n  Got: {CZ_Scale}")
    print("CZ_Scale_p_close???", jnp.allclose(CZ_Scale_p, expected_CZ_Scale_p_values, atol=0.001))
    if not jnp.allclose(CZ_Scale_p, expected_CZ_Scale_p_values, atol=0.001):
        print(f"\n  Expected: {expected_CZ_Scale_p_values}\n  Got: {CZ_Scale_p}") 
    print("CZ_Scale_q_close???", jnp.allclose(CZ_Scale_q, expected_CZ_Scale_q_values, atol=0.001))
    if not jnp.allclose(CZ_Scale_q, expected_CZ_Scale_q_values, atol=0.001):
        print(f"\n  Expected: {expected_CZ_Scale_q_values}\n  Got: {CZ_Scale_q}")
    print("CZ_Scale_r_close???", jnp.allclose(CZ_Scale_r, expected_CZ_Scale_r_values, atol=0.001))
    if not jnp.allclose(CZ_Scale_r, expected_CZ_Scale_r_values, atol=0.001):
        print(f"\n  Expected: {expected_CZ_Scale_r_values}\n  Got: {CZ_Scale_r}") 
   
    wing_transform = jnp.array([[jnp.cos(u.wing_tilt), 0, jnp.sin(u.wing_tilt)], [0, 1, 0], [-jnp.sin(u.wing_tilt), 0., jnp.cos(u.wing_tilt)]])
    tail_transform = jnp.array([[jnp.cos(u.tail_tilt), 0, jnp.sin(u.tail_tilt)], [0, 1, 0], [-jnp.sin(u.tail_tilt), 0., jnp.cos(u.tail_tilt)]])

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
   
        
    CZ_aileron_wing = mavrik_aero.CZ_aileron_wing_lookup_table(jnp.array([
        u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron
    ]))
    CZ_aileron_wing_padded = jnp.array([0.0, 0.0, CZ_aileron_wing])
    CZ_aileron_wing_padded_transformed = jnp.dot(wing_transform, CZ_aileron_wing_padded * CZ_Scale)

    CZ_elevator_tail = mavrik_aero.CZ_elevator_tail_lookup_table(jnp.array([
        u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator
    ]))
    CZ_elevator_tail_padded = jnp.array([0.0, 0.0, CZ_elevator_tail])
    CZ_elevator_tail_padded_transformed = jnp.dot(tail_transform, CZ_elevator_tail_padded * CZ_Scale)

    
    CZ_flap_wing = mavrik_aero.CZ_flap_wing_lookup_table(jnp.array([
        u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap
    ]))
    CZ_flap_wing_padded = jnp.array([0.0, 0.0, CZ_flap_wing])
    CZ_flap_wing_padded_transformed = jnp.dot(wing_transform, CZ_flap_wing_padded * CZ_Scale)

    CZ_rudder_tail = mavrik_aero.CZ_rudder_tail_lookup_table(jnp.array([
        u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder
    ]))
    CZ_rudder_tail_padded = jnp.array([0.0, 0.0, CZ_rudder_tail])
    CZ_rudder_tail_padded_transformed = jnp.dot(tail_transform, CZ_rudder_tail_padded * CZ_Scale)

    # Tail
    CZ_tail = mavrik_aero.CZ_tail_lookup_table(jnp.array([
        u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
    ]))
    CZ_tail_padded = jnp.array([0.0, 0.0, CZ_tail])
    CZ_tail_padded_transformed = jnp.dot(tail_transform, CZ_tail_padded * CZ_Scale)

    # Tail Damp p
    CZ_tail_damp_p = mavrik_aero.CZ_tail_damp_p_lookup_table(jnp.array([
        u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
    ]))
    CZ_tail_damp_p_padded = jnp.array([0.0, 0.0, CZ_tail_damp_p])
    CZ_tail_damp_p_padded_transformed = jnp.dot(tail_transform, CZ_tail_damp_p_padded * CZ_Scale_p)

    # Tail Damp q
    CZ_tail_damp_q = mavrik_aero.CZ_tail_damp_q_lookup_table(jnp.array([
        u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
    ]))
    CZ_tail_damp_q_padded = jnp.array([0.0, 0.0, CZ_tail_damp_q])
    CZ_tail_damp_q_padded_transformed = jnp.dot(tail_transform, CZ_tail_damp_q_padded * CZ_Scale_q)

    # Tail Damp r
    CZ_tail_damp_r = mavrik_aero.CZ_tail_damp_r_lookup_table(jnp.array([
        u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
    ]))
    CZ_tail_damp_r_padded = jnp.array([0.0, 0.0, CZ_tail_damp_r])
    CZ_tail_damp_r_padded_transformed = jnp.dot(tail_transform, CZ_tail_damp_r_padded * CZ_Scale_r)

    # Wing
    CZ_wing = mavrik_aero.CZ_wing_lookup_table(jnp.array([
        u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
    ]))
    CZ_wing_padded = jnp.array([0.0, 0.0, CZ_wing])
    CZ_wing_padded_transformed = jnp.dot(wing_transform, CZ_wing_padded * CZ_Scale)

    # Wing Damp p
    CZ_wing_damp_p = mavrik_aero.CZ_wing_damp_p_lookup_table(jnp.array([
        u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
    ]))
    CZ_wing_damp_p_padded = jnp.array([0.0, 0.0, CZ_wing_damp_p])
    CZ_wing_damp_p_padded_transformed = jnp.dot(wing_transform, CZ_wing_damp_p_padded * CZ_Scale_p)

    # Wing Damp q
    CZ_wing_damp_q = mavrik_aero.CZ_wing_damp_q_lookup_table(jnp.array([
        u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
    ]))
    CZ_wing_damp_q_padded = jnp.array([0.0, 0.0, CZ_wing_damp_q])
    CZ_wing_damp_q_padded_transformed = jnp.dot(wing_transform, CZ_wing_damp_q_padded * CZ_Scale_q)

    # Wing Damp r
    CZ_wing_damp_r = mavrik_aero.CZ_wing_damp_r_lookup_table(jnp.array([
        u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
    ]))
    CZ_wing_damp_r_padded = jnp.array([0.0, 0.0, CZ_wing_damp_r])
    CZ_wing_damp_r_padded_transformed = jnp.dot(wing_transform, CZ_wing_damp_r_padded * CZ_Scale_r)

    # Hover Fuse
    CZ_hover_fuse = mavrik_aero.CZ_hover_fuse_lookup_table(jnp.array([
        u.U, u.alpha, u.beta
    ]))
    CZ_hover_fuse_padded = jnp.array([0.0, 0.0, CZ_hover_fuse * CZ_Scale])
    
    F3_array = jnp.array([
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
    ])


    CZ_outputs_values_close = jnp.allclose(F3_array, expected_CZ_outputs_values, atol=0.001)
    print("CZ_outputs_values_close???", CZ_outputs_values_close)
    if not CZ_outputs_values_close:
        print(f"\n  Expected: {expected_CZ_outputs_values}\n  Got: {F3_array}")
        max_diff_index_CZ_outputs_values = jnp.argmax(jnp.abs(F3_array - expected_CZ_outputs_values))
        print(f"\n  Max difference in CZ_outputs_values at index {max_diff_index_CZ_outputs_values}: Expected {expected_CZ_outputs_values[max_diff_index_CZ_outputs_values]}, Got {F3_array[max_diff_index_CZ_outputs_values]}")