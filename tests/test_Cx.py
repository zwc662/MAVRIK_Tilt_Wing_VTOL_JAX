
import pytest

import os
import sys

from jax_mavrik.src.mavrik_aero import MavrikAero

from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.mavrik_types import StateVariables, ControlInputs, Forces
from jax_mavrik.src.actuator import ActuatorOutput 

from jax_mavrik.src.utils.mat_tools import euler_to_dcm

from .test_mavrik_aero import expected_actuator_outputs_values as actuator_outputs_values

import jax.numpy as jnp
 
expected_CX_outputs_values = jnp.array([
    [-114.295761551278, 0, 0],
    [-114.449641091491, 0, 0],
    [-114.163852031687, 0, 0],
    [-113.441654414852, 0, 0],
    [-112.303261884019, 0, 0],
    [-110.78348892127, 0, 0],
    [-108.929021611756, 0, 0],
    [-106.795484478394, 0, 0],
    [-104.444458270136, 0, 0],
    [-101.940582730151, 0, 0],
    [-99.3547737877012, 0, 0]
])

expected_CX_alieron_wing_values = jnp.zeros([11])
expected_CX_elevator_tail_values = jnp.zeros([11])
expected_CX_flap_wing_values = jnp.zeros([11])
expected_CX_ruder_tail_values = jnp.zeros([11])
expected_CX_tail_values = jnp.array([
    -0.200935724251243, -0.200345607746314, -0.199576005784025, -0.198648672918778,
    -0.197585742666469, -0.196409434490178, -0.19514178008106, -0.193804370684883,
    -0.192418127020352, -0.191003093080959, -0.189579949385223
])
expected_CX_tail_damp_p_values = jnp.array([
    2.21393697049284e-07, -5.55945927582145e-06, -2.11950411074508e-05, -4.33662481646095e-05,
    -6.79358506547596e-05, -9.01687575664461e-05, -0.000104944744572204, -0.000106960361549783,
    -9.09167524952728e-05, -5.16903433683209e-05, 1.54376205982127e-05
])
expected_CX_tail_damp_q_values = jnp.array([
    1.8574115303102, 1.76875869206609, 1.65310041084851, 1.51371276414411,
    1.35393001190805, 1.17709979620799, 0.986541318975133, 0.785506789512281,
    0.577146394097683, 0.364476992624244, 0.150637597247265
])
expected_CX_tail_damp_r_values = jnp.array([
    8.41278677769832e-06, -2.00457074485548e-05, -0.000105195570046368, -0.000246818826888079, 
    -0.000444840995718077, -0.000699338072688702, -0.00101054715264127, -0.00137888125838211, 
    -0.00180494849542245, -0.0022895752501057, -0.00283389330364904
])
expected_CX_wing_values = jnp.array([
    -0.160432261153205, -0.158754236234397, -0.156562494170812, -0.15391981387405, 
    -0.150889938855056, -0.147536746082496, -0.143923475565989, -0.140112023503898, 
    -0.136162301324544, -0.132131662500254, -0.128088670988319
])
expected_CX_wing_damp_p_values = jnp.array([
    0.000221227999251527, 0.000195430200071627, 0.000138174843276469, 5.64270050948338e-05,
    -4.12041751702652e-05, -0.000144914004998831, -0.000244137019700513, -0.000327960850099901,
    -0.000385512039473838, -0.000406307188606199, -0.000381000732989396
])
expected_CX_wing_damp_q_values = jnp.array([
    5.68300121379039, 5.45253663411631, 5.15155326156364, 4.78859784668773,
    4.37237047251646, 3.91160761150427, 3.4149728892398, 2.89095632895936,
    2.34778274878057, 1.79332985971524, 1.2355612208852 
])
expected_CX_wing_damp_r_values = jnp.array([ 
    0.00021037872884806, 0.00022343754043074, 0.000240423906369049, 0.000260954102632588,
    0.000284602280587126, 0.00031091337841736, 0.000339416267364585, 0.000369636441132416,
    0.000401107636824296, 0.000433381902895385, 0.000468444099174172 
]) 
expected_CX_hover_fuse_values = jnp.array([
    0.000401324576377633, 0.000403541962794507, 0.000407002940470102, 0.000411553638085984,
    0.000417038934320398, 0.000423304268434168, 0.000430197254534993, 0.000437569128903729,
    0.000445276059254511, 0.000453180339650044, 0.000461153278755525
]) 

expected_CX_Scale_values = jnp.array([
    316.638, 315.729883945804, 314.845190058716, 313.988838147529, 313.164400714939,
    312.374104928985, 311.618896081704, 310.898550595885, 310.211826405883, 309.556638785613,
    308.930807904554
])

expected_CX_Scale_p_values = jnp.array([
    0, -0.11446510667421, -0.229402121005288, -0.345172422576589, -0.46212187051964,
    -0.580587743826303, -0.70090610852723, -0.82341932343948, -0.948483433834421,
    -1.07647523887386, -1.20818303955046
])

expected_CX_Scale_q_values = jnp.array([
    0, -0.16596442266699, -0.317940535744807, -0.455119391134645, -0.576940294487692,
    -0.683074267479587, -0.773406673460475, -0.84801948103289, -0.907173485108433,
    -0.951290675908402, -0.980963804822868
])

expected_CX_Scale_r_values = jnp.array([
    0, 0.0027518656446275, 0.00610265683220155, 0.0104814908383701, 0.0163052131445058,
    0.0239797646628719, 0.0339017233709519, 0.0464601114471251, 0.0620385210874145,
    0.081017576558698, 0.103787909352465
])
 
expected_wind_transform = jnp.repeat(jnp.diag(jnp.array([1., 1., 1.]))[None, :, :], 11, axis=0)
expected_tail_transform = jnp.repeat(jnp.diag(jnp.array([1., 1., 1.]))[None, :, :], 11, axis=0) 

expected_CX_aileron_wing_padded_transformed_values = jnp.zeros([11, 3])
expected_CX_elevator_tail_padded_transformed_values = jnp.zeros([11, 3])
expected_CX_flap_wing_padded_transformed_values = jnp.zeros([11, 3])
expected_CX_ruder_tail_padded_transformed_values = jnp.zeros([11, 3]) 
expected_CX_tail_padded_transformed_values = jnp.array([
    [-63.623885855465, 0, 0], [-63.2550954827951, 0, 0], [-62.8355454722308, 0, 0], [-62.3734660093156, 0, 0],
    [-61.8768206919609, 0, 0], [-61.3532212984774, 0, 0], [-60.8098660882786, 0, 0], [-60.2534979450777, 0, 0],
    [-59.6903786165827, 0, 0], [-59.1262754917973, 0, 0], [-58.5670869260815, 0, 0]
]) 
expected_CX_tail_damp_p_padded_transformed_values = jnp.array([
    [0, 0, 0], [6.36364099057827e-07, 0, 0], [4.86218738484348e-06, 0, 0], [1.49688329370358e-05, 0, 0],
    [3.13946423799204e-05, 0, 0], [5.23508755191238e-05, 0, 0], [7.35564125284874e-05, 0, 0], [8.80732285421646e-05, 0, 0],
    [8.62330335997906e-05, 0, 0], [5.56433747248851e-05, 0, 0], [-1.86514713777755e-05, 0, 0]
])
expected_CX_tail_damp_q_padded_transformed_values = jnp.array([
    [0., 0., 0.], [-0.293551015165968, 0., 0.], [-0.525587630265134, 0., 0.], [-0.688920031570008, 0., 0.],
    [-0.781136779785952, 0., 0.], [-0.804046581045146, 0., 0.], [-0.762997639739867, 0., 0.], [-0.666125059990016, 0., 0.],
    [-0.52357190575136, 0., 0.], [-0.346723564666579, 0., 0.], [-0.147770030545052, 0., 0.]
])
expected_CX_tail_damp_r_padded_transformed_values = jnp.array([
    [0, 0, 0], [-5.51630936499317e-08, 0, 0], [-6.41972464260803e-07, 0, 0], [-2.58702927276466e-06, 0, 0],
    [-7.25322725059744e-06, 0, 0], [-1.67699624028615e-05, 0, 0], [-3.42592900221475e-05, 0, 0],
    [-6.4062976936785e-05, 0, 0], [-0.000111976335294963, 0, 0], [-0.000185495838112339, 0, 0],
    [-0.000294123861313685, 0, 0]
])
expected_CX_wing_padded_transformed_values = jnp.array([
    [-50.7989503070284, 0, 0], [-50.1234565821909, 0, 0], [-49.2929482332759, 0, 0], [-48.3291035261969, 0, 0],
    [-47.2533572754573, 0, 0], [-46.0866590016547, 0, 0], [-44.8492745761155, 0, 0], [-43.5606250284183, 0, 0],
    [-42.2391561815149, 0, 0], [-40.9022333207335, 0, 0], [-39.570536611842, 0, 0]
])
expected_CX_wing_damp_p_padded_transformed_values = jnp.array([
    [0, 0, 0], [-2.23699386985609e-05, 0, 0], [-3.16976021171951e-05, 0, 0], [-1.94770460473253e-05, 0, 0],
    [1.90413505029019e-05, 0, 0], [8.4135295211105e-05, 0, 0], [0.000171117128425723, 0, 0], [0.000270049301303897, 0, 0],
    [0.000365651782984657, 0, 0], [0.000437379627911026, 0, 0], [0.000460318623654083, 0, 0]
]) 
expected_CX_wing_damp_q_padded_transformed_values = jnp.array([
    [0., 0., 0.], [-0.904927094551725, 0., 0.], [-1.63788760389945, 0., 0.], [-2.17938373637319, 0., 0.],
    [-2.52259670802293, 0., 0.], [-2.67191850389586, 0., 0.], [-2.64116282222466, 0., 0.], [-2.45158728577286, 0., 0.],
    [-2.12984625848873, 0., 0.], [-1.70597797437523, 0., 0.], [-1.21204083633113, 0., 0.]
]) 
expected_CX_wing_damp_r_padded_transformed_values = jnp.array([
    [0., 0., 0.], [6.14870091231421e-07, 0., 0.], [1.46722459482766e-06, 0., 0.], [2.73518803597856e-06, 0., 0.],
    [4.64050084638555e-06, 0., 0.], [7.45562964498673e-06, 0., 0.], [1.15067964037952e-05, 0., 0.], [1.71733502499308e-05, 0., 0.],
    [2.48841245854471e-05, 0., 0.], [3.51115514969811e-05, 0., 0.], [4.86188337017861e-05, 0., 0.]
]) 

expected_CX_hover_fuse_padded_values = jnp.array([
    [0.127074611215061, 0, 0], [0.127410257080372, 0, 0], [0.128142918146765, 0, 0], [0.129223248658007, 0, 0],
    [0.130601747941244, 0, 0], [0.132229291964742, 0, 0], [0.134057593555574, 0, 0], [0.136039607961673, 0, 0],
    [0.138129899596156, 0, 0], [0.14028498270579, 0, 0], [0.142464454973778, 0, 0]
])

 
@pytest.fixture
def mavrik_aero():
    mavrik_setup = MavrikSetup(file_path=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "jax_mavrik/aero_export.mat")
    )
    return MavrikAero(mavrik_setup=mavrik_setup)


@pytest.mark.parametrize(
    "id, actuator_outputs_values, \
        expected_CX_outputs_values, expected_CX_alieron_wing_values, expected_CX_elevator_tail_values, expected_CX_flap_wing_values, expected_CX_ruder_tail_values, expected_CX_tail_values, expected_CX_tail_damp_p_values, expected_CX_tail_damp_q_values, expected_CX_tail_damp_r_values, expected_CX_wing_values, expected_CX_wing_damp_p_values, expected_CX_wing_damp_q_values, expected_CX_wing_damp_r_values, expected_CX_hover_fuse_values, \
            expected_CX_Scale_values, expected_CX_Scale_p_values, expected_CX_Scale_q_values, expected_CX_Scale_r_values, \
                expected_wind_transform, expected_tail_transform, \
                expected_CX_tail_padded_transformed_values, expected_CX_tail_damp_p_padded_transformed_values, expected_CX_tail_damp_q_padded_transformed_values, expected_CX_tail_damp_r_padded_transformed_values, expected_CX_wing_padded_transformed_values, expected_CX_wing_damp_p_padded_transformed_values, expected_CX_wing_damp_q_padded_transformed_values, expected_CX_wing_damp_r_padded_transformed_values, expected_CX_hover_fuse_padded_values",
    zip(
        list(range(11)), actuator_outputs_values, \
            expected_CX_outputs_values, expected_CX_alieron_wing_values, expected_CX_elevator_tail_values, expected_CX_flap_wing_values, expected_CX_ruder_tail_values, expected_CX_tail_values, expected_CX_tail_damp_p_values, expected_CX_tail_damp_q_values, expected_CX_tail_damp_r_values, expected_CX_wing_values, expected_CX_wing_damp_p_values, expected_CX_wing_damp_q_values, expected_CX_wing_damp_r_values, expected_CX_hover_fuse_values, \
            expected_CX_Scale_values, expected_CX_Scale_p_values, expected_CX_Scale_q_values, expected_CX_Scale_r_values,
            expected_wind_transform, expected_tail_transform, \
            expected_CX_tail_padded_transformed_values, expected_CX_tail_damp_p_padded_transformed_values, expected_CX_tail_damp_q_padded_transformed_values, expected_CX_tail_damp_r_padded_transformed_values, expected_CX_wing_padded_transformed_values, expected_CX_wing_damp_p_padded_transformed_values, expected_CX_wing_damp_q_padded_transformed_values, expected_CX_wing_damp_r_padded_transformed_values, expected_CX_hover_fuse_padded_values
    )
)
 
def test_mavrik_aero(id, mavrik_aero, actuator_outputs_values, \
                     expected_CX_outputs_values, expected_CX_alieron_wing_values, expected_CX_elevator_tail_values, expected_CX_flap_wing_values, expected_CX_ruder_tail_values, expected_CX_tail_values, expected_CX_tail_damp_p_values, expected_CX_tail_damp_q_values, expected_CX_tail_damp_r_values, expected_CX_wing_values, expected_CX_wing_damp_p_values, expected_CX_wing_damp_q_values, expected_CX_wing_damp_r_values, expected_CX_hover_fuse_values, \
                     expected_CX_Scale_values, expected_CX_Scale_p_values, expected_CX_Scale_q_values, expected_CX_Scale_r_values,
                     expected_wind_transform, expected_tail_transform, \
                        expected_CX_tail_padded_transformed_values, expected_CX_tail_damp_p_padded_transformed_values, expected_CX_tail_damp_q_padded_transformed_values, expected_CX_tail_damp_r_padded_transformed_values, expected_CX_wing_padded_transformed_values, expected_CX_wing_damp_p_padded_transformed_values, expected_CX_wing_damp_q_padded_transformed_values, expected_CX_wing_damp_r_padded_transformed_values, expected_CX_hover_fuse_padded_values):
    u = ActuatorOutput(*actuator_outputs_values)
        
    
    print(f">>>>>>>>>>>>>>>>>>>> Test ID: {id} <<<<<<<<<<<<<<<<<<<<<<")
      
    wing_transform = jnp.array([[jnp.cos(u.wing_tilt), 0, jnp.sin( u.wing_tilt)], [0, 1, 0], [-jnp.sin(u.wing_tilt), 0., jnp.cos(u.wing_tilt)]]);
    tail_transform = jnp.array([[jnp.cos(u.tail_tilt), 0, jnp.sin(u.tail_tilt)], [0, 1, 0], [-jnp.sin(u.tail_tilt), 0., jnp.cos(u.tail_tilt)]])
    
    F1 = mavrik_aero.Cx(u, wing_transform, tail_transform)
    F1_array = jnp.array([F1.Fx, F1.Fy, F1.Fz])
    CX_outputs_values_close = jnp.allclose(F1_array, expected_CX_outputs_values, atol=0.0001)
    print("CX_outputs_values_close???", CX_outputs_values_close)
    if not CX_outputs_values_close:
        print(f"\n  Expected: {expected_CX_outputs_values}\n  Got: {F1_array}")
        max_diff_index_CX_outputs_values = jnp.argmax(jnp.abs(F1_array - expected_CX_outputs_values))
        print(f"\n  Max difference in CX_outputs_values at index {max_diff_index_CX_outputs_values}: Expected {expected_CX_outputs_values[max_diff_index_CX_outputs_values]}, Got {F1_array[max_diff_index_CX_outputs_values]}")

    
    CX_Scale = 0.5744 * u.Q
    CX_Scale_r = 0.5744 * 2.8270 * 1.225 * 0.25 * u.U * u.r
    CX_Scale_p = 0.5744 * 2.8270 * 1.225 * 0.25 * u.U * u.p
    CX_Scale_q = 0.5744 * 0.2032 * 1.225 * 0.25 * u.U * u.q
 
    print("CX_Scale_close???", jnp.allclose(CX_Scale, expected_CX_Scale_values, atol=0.0001))
    if not jnp.allclose(CX_Scale, expected_CX_Scale_values, atol=0.0001):
        print(f"\n  Expected: {expected_CX_Scale_values}\n  Got: {CX_Scale}")
    print("CX_Scale_p_close???", jnp.allclose(CX_Scale_p, expected_CX_Scale_p_values, atol=0.0001))
    if not jnp.allclose(CX_Scale_p, expected_CX_Scale_p_values, atol=0.0001):
        print(f"\n  Expected: {expected_CX_Scale_p_values}\n  Got: {CX_Scale_p}") 
    print("CX_Scale_q_close???", jnp.allclose(CX_Scale_q, expected_CX_Scale_q_values, atol=0.0001))
    if not jnp.allclose(CX_Scale_q, expected_CX_Scale_q_values, atol=0.0001):
        print(f"\n  Expected: {expected_CX_Scale_q_values}\n  Got: {CX_Scale_q}")
    print("CX_Scale_r_close???", jnp.allclose(CX_Scale_r, expected_CX_Scale_r_values, atol=0.0001))
    if not jnp.allclose(CX_Scale_r, expected_CX_Scale_r_values, atol=0.0001):
        print(f"\n  Expected: {expected_CX_Scale_r_values}\n  Got: {CX_Scale_r}") 
   
    '''
    print("wing_transform_close???", jnp.allclose(wing_transform, expected_wind_transform, atol=0.0001))
    if not jnp.allclose(wing_transform, expected_wind_transform, atol=0.0001):
        print(f"\n  Expected: {expected_wind_transform}\n  Got: {wing_transform}")
        max_diff_index_wing_transform = jnp.argmax(jnp.abs(wing_transform - expected_wind_transform))
        print(f"\n  Max difference in wing_transform at index {max_diff_index_wing_transform}: Expected {expected_wind_transform[max_diff_index_wing_transform]}, Got {wing_transform[max_diff_index_wing_transform]}")

    print("tail_transform_close???", jnp.allclose(tail_transform, expected_tail_transform, atol=0.0001))
    if not jnp.allclose(tail_transform, expected_tail_transform, atol=0.0001):
        print(f"\n  Expected: {expected_tail_transform}\n  Got: {tail_transform}")
        max_diff_index_tail_transform = jnp.argmax(jnp.abs(tail_transform - expected_tail_transform))
        print(f"\n  Max difference in tail_transform at index {max_diff_index_tail_transform}: Expected {expected_tail_transform[max_diff_index_tail_transform]}, Got {tail_transform[max_diff_index_tail_transform]}")
    '''

    CX_aileron_wing = mavrik_aero.CX_aileron_wing_lookup_table(jnp.array([
        u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron
    ]))
    CX_aileron_wing_close =  jnp.allclose(CX_aileron_wing, expected_CX_alieron_wing_values, atol=0.0001)
    CX_aileron_wing_padded = jnp.array([CX_aileron_wing, 0.0, 0.0])
    CX_aileron_wing_padded_transformed = jnp.dot(wing_transform, CX_aileron_wing_padded * CX_Scale)    
    
    CX_elevator_tail = mavrik_aero.CX_elevator_tail_lookup_table(jnp.array([
        u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator
    ]))
    CX_elevator_tail_padded = jnp.array([CX_elevator_tail, 0.0, 0.0])
    CX_elevator_tail_padded_transformed = jnp.dot(tail_transform, CX_elevator_tail_padded * CX_Scale)

    CX_flap_wing = mavrik_aero.CX_flap_wing_lookup_table(jnp.array([
        u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap
    ]))
    CX_flap_wing_padded = jnp.array([CX_flap_wing, 0.0, 0.0])
    CX_flap_wing_padded_transformed = jnp.dot(wing_transform, CX_flap_wing_padded * CX_Scale)

    CX_rudder_tail = mavrik_aero.CX_rudder_tail_lookup_table(jnp.array([
        u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder
    ]))
    CX_rudder_tail_padded = jnp.array([CX_rudder_tail, 0.0, 0.0])
    CX_rudder_tail_padded_transformed = jnp.dot(tail_transform, CX_rudder_tail_padded * CX_Scale)

    # Tail
    CX_tail = mavrik_aero.CX_tail_lookup_table(jnp.array([
        u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
    ]))
    CX_tail_padded = jnp.array([CX_tail, 0.0, 0.0])
    CX_tail_padded_transformed = jnp.dot(tail_transform, CX_tail_padded * CX_Scale)

    # Tail Damp p
    CX_tail_damp_p = mavrik_aero.CX_tail_damp_p_lookup_table(jnp.array([
        u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
    ])) 
    CX_tail_damp_p_padded = jnp.array([CX_tail_damp_p, 0.0, 0.0])
    CX_tail_damp_p_padded_transformed = jnp.dot(tail_transform, CX_tail_damp_p_padded * CX_Scale_p)

    # Tail Damp q
    CX_tail_damp_q = mavrik_aero.CX_tail_damp_q_lookup_table(jnp.array([
        u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
    ]))
    CX_tail_damp_q_padded = jnp.array([CX_tail_damp_q, 0.0, 0.0])
    CX_tail_damp_q_padded_transformed = jnp.dot(tail_transform, CX_tail_damp_q_padded * CX_Scale_q)

    # Tail Damp r
    CX_tail_damp_r = mavrik_aero.CX_tail_damp_r_lookup_table(jnp.array([
        u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
    ]))
    CX_tail_damp_r_padded = jnp.array([CX_tail_damp_r, 0.0, 0.0])
    CX_tail_damp_r_padded_transformed = jnp.dot(tail_transform, CX_tail_damp_r_padded * CX_Scale_r)

    # Wing
    CX_wing = mavrik_aero.CX_wing_lookup_table(jnp.array([
        u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
    ]))
    CX_wing_padded = jnp.array([CX_wing, 0.0, 0.0])
    CX_wing_padded_transformed = jnp.dot(wing_transform, CX_wing_padded * CX_Scale)

    # Wing Damp p
    CX_wing_damp_p = mavrik_aero.CX_wing_damp_p_lookup_table(jnp.array([
        u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
    ]))
    CX_wing_damp_p_padded = jnp.array([CX_wing_damp_p, 0.0, 0.0])
    CX_wing_damp_p_padded_transformed = jnp.dot(wing_transform, CX_wing_damp_p_padded * CX_Scale_p)

    # Wing Damp q
    CX_wing_damp_q = mavrik_aero.CX_wing_damp_q_lookup_table(jnp.array([
        u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
    ]))
    CX_wing_damp_q_padded = jnp.array([CX_wing_damp_q, 0.0, 0.0])
    CX_wing_damp_q_padded_transformed = jnp.dot(wing_transform, CX_wing_damp_q_padded * CX_Scale_q)

    # Wing Damp r
    CX_wing_damp_r = mavrik_aero.CX_wing_damp_r_lookup_table(jnp.array([
        u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
    ]))
    CX_wing_damp_r_padded = jnp.array([CX_wing_damp_r, 0.0, 0.0])
    CX_wing_damp_r_padded_transformed = jnp.dot(wing_transform, CX_wing_damp_r_padded * CX_Scale_r)

    # Hover Fuse
    CX_hover_fuse = mavrik_aero.CX_hover_fuse_lookup_table(jnp.array([
        u.U, u.alpha, u.beta
    ]))
    CX_hover_fuse_padded = jnp.array([CX_hover_fuse * CX_Scale, 0.0, 0.0])
   
    
    CX_aileron_wing_close = jnp.allclose(CX_aileron_wing, expected_CX_alieron_wing_values, atol=0.0001)
    print("CX_aileron_wing_close???", CX_aileron_wing_close)
    if not CX_aileron_wing_close:
        print(f"\n  Expected: {expected_CX_alieron_wing_values}\n  Got: {CX_aileron_wing}")
        max_diff_index_CX_aileron_wing = jnp.argmax(jnp.abs(CX_aileron_wing - expected_CX_alieron_wing_values))
        print(f"\n  Max difference in CX_aileron_wing at index {max_diff_index_CX_aileron_wing}: Expected {expected_CX_alieron_wing_values[max_diff_index_CX_aileron_wing]}, Got {CX_aileron_wing[max_diff_index_CX_aileron_wing]}")
    CX_aileron_wing_padded_transformed_values_close = jnp.allclose(CX_aileron_wing_padded_transformed, expected_CX_aileron_wing_padded_transformed_values, atol=0.0001)
    print("CX_aileron_wing_padded_transformed_values_close???", CX_aileron_wing_padded_transformed_values_close)
    
    CX_elevator_tail_close = jnp.allclose(CX_elevator_tail, expected_CX_elevator_tail_values, atol=0.0001)
    print("CX_elevator_tail_close???", CX_elevator_tail_close)
    if not CX_elevator_tail_close:
        print(f"\n  Expected: {expected_CX_elevator_tail_values}\n  Got: {CX_elevator_tail}")
        max_diff_index_CX_elevator_tail = jnp.argmax(jnp.abs(CX_elevator_tail - expected_CX_elevator_tail_values))
        print(f"\n  Max difference in CX_elevator_tail at index {max_diff_index_CX_elevator_tail}: Expected {expected_CX_elevator_tail_values[max_diff_index_CX_elevator_tail]}, Got {CX_elevator_tail[max_diff_index_CX_elevator_tail]}")
    CX_elevator_tail_padded_transformed_values_close = jnp.allclose(CX_elevator_tail_padded_transformed, expected_CX_elevator_tail_padded_transformed_values, atol=0.0001)
    print("CX_elevator_tail_padded_transformed_values_close???", CX_elevator_tail_padded_transformed_values_close)
     
    CX_flap_wing_close = jnp.allclose(CX_flap_wing, expected_CX_flap_wing_values, atol=0.0001)
    print("CX_flap_wing_close???", CX_flap_wing_close)
    if not CX_flap_wing_close:
        print(f"\n  Expected: {expected_CX_flap_wing_values}\n  Got: {CX_flap_wing}")
        max_diff_index_CX_flap_wing = jnp.argmax(jnp.abs(CX_flap_wing - expected_CX_flap_wing_values))
        print(f"\n  Max difference in CX_flap_wing at index {max_diff_index_CX_flap_wing}: Expected {expected_CX_flap_wing_values[max_diff_index_CX_flap_wing]}, Got {CX_flap_wing[max_diff_index_CX_flap_wing]}")
    CX_flap_wing_padded_transformed_values_close = jnp.allclose(CX_flap_wing_padded_transformed, expected_CX_flap_wing_padded_transformed_values, atol=0.0001)  
    print("CX_flap_wing_padded_transformed_values_close???", CX_flap_wing_padded_transformed_values_close)
    
    
    CX_rudder_tail_close = jnp.allclose(CX_rudder_tail, expected_CX_ruder_tail_values, atol=0.0001)
    print("CX_rudder_tail_close???", CX_rudder_tail_close)
    if not CX_rudder_tail_close:
        print(f"\n  Expected: {expected_CX_ruder_tail_values}\n  Got: {CX_rudder_tail}")
        max_diff_index_CX_rudder_tail = jnp.argmax(jnp.abs(CX_rudder_tail - expected_CX_ruder_tail_values))
        print(f"\n  Max difference in CX_rudder_tail at index {max_diff_index_CX_rudder_tail}: Expected {expected_CX_ruder_tail_values[max_diff_index_CX_rudder_tail]}, Got {CX_rudder_tail[max_diff_index_CX_rudder_tail]}")
    CX_rudder_tail_padded_transformed_values_close = jnp.allclose(CX_rudder_tail_padded_transformed, expected_CX_ruder_tail_padded_transformed_values, atol=0.0001)
    print("CX_rudder_tail_padded_transformed_values_close???", CX_rudder_tail_padded_transformed_values_close)
     
    CX_tail_close = jnp.allclose(CX_tail, expected_CX_tail_values, atol=0.0001)
    print("CX_tail_close???", CX_tail_close)
    if not CX_tail_close:
        print(f"\n  Expected: {expected_CX_tail_values}\n  Got: {CX_tail}")
        max_diff_index_CX_tail = jnp.argmax(jnp.abs(CX_tail - expected_CX_tail_values))
        print(f"\n  Max difference in CX_tail at index {max_diff_index_CX_tail}: Expected {expected_CX_tail_values[max_diff_index_CX_tail]}, Got {CX_tail[max_diff_index_CX_tail]}")
    CX_tail_padded_transformed_values_close = jnp.allclose(CX_tail_padded_transformed, expected_CX_tail_padded_transformed_values, atol=0.0001)
    print("CX_tail_padded_transformed_values_close???", CX_tail_padded_transformed_values_close)
    if not CX_tail_padded_transformed_values_close:
        print(f"\n  Expected: {expected_CX_tail_padded_transformed_values}\n  Got: {CX_tail_padded_transformed}")


    CX_tail_damp_p_close = jnp.allclose(CX_tail_damp_p, expected_CX_tail_damp_p_values, atol=0.0001)
    print("CX_tail_damp_p_close???", CX_tail_damp_p_close)
    if not CX_tail_damp_p_close:
        print(f"\n  Expected: {expected_CX_tail_damp_p_values}\n  Got: {CX_tail_damp_p}")
        max_diff_index_CX_tail_damp_p = jnp.argmax(jnp.abs(CX_tail_damp_p - expected_CX_tail_damp_p_values))
        print(f"\n  Max difference in CX_tail_damp_p at index {max_diff_index_CX_tail_damp_p}: Expected {expected_CX_tail_damp_p_values[max_diff_index_CX_tail_damp_p]}, Got {CX_tail_damp_p[max_diff_index_CX_tail_damp_p]}")
    CX_tail_damp_p_padded_transformed_values_close = jnp.allclose(CX_tail_damp_p_padded_transformed, expected_CX_tail_damp_p_padded_transformed_values, atol=0.0001)
    print("CX_tail_damp_p_padded_transformed_values_close???", CX_tail_damp_p_padded_transformed_values_close)
     
    CX_tail_damp_q_close = jnp.allclose(CX_tail_damp_q, expected_CX_tail_damp_q_values, atol=0.0001)
    print("CX_tail_damp_q_close???", CX_tail_damp_q_close)
    if not CX_tail_damp_q_close:
        print(f"\n  Expected: {expected_CX_tail_damp_q_values}\n  Got: {CX_tail_damp_q}")
        max_diff_index_CX_tail_damp_q = jnp.argmax(jnp.abs(CX_tail_damp_q - expected_CX_tail_damp_q_values))
        print(f"\n  Max difference in CX_tail_damp_q at index {max_diff_index_CX_tail_damp_q}: Expected {expected_CX_tail_damp_q_values[max_diff_index_CX_tail_damp_q]}, Got {CX_tail_damp_q[max_diff_index_CX_tail_damp_q]}")
    CX_tail_damp_q_padded_transformed_values_close = jnp.allclose(CX_tail_damp_q_padded_transformed, expected_CX_tail_damp_q_padded_transformed_values, atol=0.0001)
    print("CX_tail_damp_q_padded_transformed_values_close???", CX_tail_damp_q_padded_transformed_values_close)
    if not CX_tail_damp_q_padded_transformed_values_close:
        print(f"\n  Expected: {expected_CX_tail_damp_q_padded_transformed_values}\n  Got: {CX_tail_damp_q_padded_transformed}")
     
    CX_tail_damp_r_close = jnp.allclose(CX_tail_damp_r, expected_CX_tail_damp_r_values, atol=0.0001)
    print("CX_tail_damp_r_close???", CX_tail_damp_r_close)
    if not CX_tail_damp_r_close:
        print(f"\n  Expected: {expected_CX_tail_damp_r_values}\n  Got: {CX_tail_damp_r}")
        max_diff_index_CX_tail_damp_r = jnp.argmax(jnp.abs(CX_tail_damp_r - expected_CX_tail_damp_r_values))
        print(f"\n  Max difference in CX_tail_damp_r at index {max_diff_index_CX_tail_damp_r}: Expected {expected_CX_tail_damp_r_values[max_diff_index_CX_tail_damp_r]}, Got {CX_tail_damp_r[max_diff_index_CX_tail_damp_r]}")
    CX_tail_damp_r_padded_transformed_values_close = jnp.allclose(CX_tail_damp_r_padded_transformed, expected_CX_tail_damp_r_padded_transformed_values, atol=0.0001)
    print("CX_tail_damp_r_padded_transformed_values_close???", CX_tail_damp_r_padded_transformed_values_close)
     
    CX_wing_close = jnp.allclose(CX_wing, expected_CX_wing_values, atol=0.0001)
    print("CX_wing_close???", CX_wing_close)
    if not CX_wing_close:
        print(f"\n  Expected: {expected_CX_wing_values}\n  Got: {CX_wing}")
        max_diff_index_CX_wing = jnp.argmax(jnp.abs(CX_wing - expected_CX_wing_values))
        print(f"\n  Max difference in CX_wing at index {max_diff_index_CX_wing}: Expected {expected_CX_wing_values[max_diff_index_CX_wing]}, Got {CX_wing[max_diff_index_CX_wing]}")
    CX_wing_padded_transformed_values_close = jnp.allclose(CX_wing_padded_transformed, expected_CX_wing_padded_transformed_values, atol=0.0001)
    print("CX_wing_padded_transformed_values_close???", CX_wing_padded_transformed_values_close)
    
    CX_wing_damp_p_close = jnp.allclose(CX_wing_damp_p, expected_CX_wing_damp_p_values, atol=0.0001)
    print("CX_wing_damp_p_close???", CX_wing_damp_p_close)
    if not CX_wing_damp_p_close:
        print(f"\n  Expected: {expected_CX_wing_damp_p_values}\n  Got: {CX_wing_damp_p}")
        max_diff_index_CX_wing_damp_p = jnp.argmax(jnp.abs(CX_wing_damp_p - expected_CX_wing_damp_p_values))
        print(f"\n  Max difference in CX_wing_damp_p at index {max_diff_index_CX_wing_damp_p}: Expected {expected_CX_wing_damp_p_values[max_diff_index_CX_wing_damp_p]}, Got {CX_wing_damp_p[max_diff_index_CX_wing_damp_p]}")
    CX_wing_damp_p_padded_transformed_values_close = jnp.allclose(CX_wing_damp_p_padded_transformed, expected_CX_wing_damp_p_padded_transformed_values, atol=0.0001)
    print("CX_wing_damp_p_padded_transformed_values_close???", CX_wing_damp_p_padded_transformed_values_close)
     
    CX_wing_damp_q_close = jnp.allclose(CX_wing_damp_q, expected_CX_wing_damp_q_values, atol=0.0001)
    print("CX_wing_damp_q_close???", CX_wing_damp_q_close)
    if not CX_wing_damp_q_close:
        print(f"\n  Expected: {expected_CX_wing_damp_q_values}\n  Got: {CX_wing_damp_q}")
        max_diff_index_CX_wing_damp_q = jnp.argmax(jnp.abs(CX_wing_damp_q - expected_CX_wing_damp_q_values))
        print(f"\n  Max difference in CX_wing_damp_q at index {max_diff_index_CX_wing_damp_q}: Expected {expected_CX_wing_damp_q_values[max_diff_index_CX_wing_damp_q]}, Got {CX_wing_damp_q[max_diff_index_CX_wing_damp_q]}")
    CX_wing_damp_q_padded_transformed_values_close = jnp.allclose(CX_wing_damp_q_padded_transformed, expected_CX_wing_damp_q_padded_transformed_values, atol=0.0001)
    print("CX_wing_damp_q_padded_transformed_values_close???", CX_wing_damp_q_padded_transformed_values_close)
    if not CX_wing_damp_q_padded_transformed_values_close:
        print(f"\n  Expected: {expected_CX_wing_damp_q_padded_transformed_values}\n  Got: {CX_wing_damp_q_padded_transformed}")
     
    CX_wing_damp_r_close = jnp.allclose(CX_wing_damp_r, expected_CX_wing_damp_r_values, atol=0.0001)
    print("CX_wing_damp_r_close???", CX_wing_damp_r_close)
    if not CX_wing_damp_r_close:
        print(f"\n  Expected: {expected_CX_wing_damp_r_values}\n  Got: {CX_wing_damp_r}")
        max_diff_index_CX_wing_damp_r = jnp.argmax(jnp.abs(CX_wing_damp_r - expected_CX_wing_damp_r_values))
        print(f"\n  Max difference in CX_wing_damp_r at index {max_diff_index_CX_wing_damp_r}: Expected {expected_CX_wing_damp_r_values[max_diff_index_CX_wing_damp_r]}, Got {CX_wing_damp_r[max_diff_index_CX_wing_damp_r]}")
    CX_wing_damp_r_padded_transformed_values_close = jnp.allclose(CX_wing_damp_r_padded_transformed, expected_CX_wing_damp_r_padded_transformed_values, atol=0.0001)
    print("CX_wing_damp_r_padded_transformed_values_close???", CX_wing_damp_r_padded_transformed_values_close)
    if not CX_wing_damp_r_padded_transformed_values_close:
        print(f"\n  Expected: {expected_CX_wing_damp_r_padded_transformed_values}\n  Got: {CX_wing_damp_r_padded_transformed}")

    CX_hover_fuse_close = jnp.allclose(CX_hover_fuse, expected_CX_hover_fuse_values, atol=0.0001)
    print("CX_hover_fuse_close???", CX_hover_fuse_close)
    if not CX_hover_fuse_close:
        print(f"\n  Expected: {expected_CX_hover_fuse_values}\n  Got: {CX_hover_fuse}")
        max_diff_index_CX_hover_fuse = jnp.argmax(jnp.abs(CX_hover_fuse - expected_CX_hover_fuse_values))
        print(f"\n  Max difference in CX_hover_fuse at index {max_diff_index_CX_hover_fuse}: Expected {expected_CX_hover_fuse_values[max_diff_index_CX_hover_fuse]}, Got {CX_hover_fuse[max_diff_index_CX_hover_fuse]}")
    CX_hover_fuse_padded_transformed_values_close = jnp.allclose(CX_hover_fuse_padded, expected_CX_hover_fuse_padded_values, atol=0.0001)
    print("CX_hover_fuse_padded_transformed_values_close???", CX_hover_fuse_padded_transformed_values_close)
    if not CX_hover_fuse_padded_transformed_values_close:
        print(f"\n  Expected: {expected_CX_hover_fuse_padded_values}\n  Got: {CX_hover_fuse_padded}")


    F1_array = jnp.array([
            CX_aileron_wing_padded_transformed[0] + CX_elevator_tail_padded_transformed[0] + CX_flap_wing_padded_transformed[0] + CX_rudder_tail_padded_transformed[0] +
            CX_tail_padded_transformed[0] + CX_tail_damp_p_padded_transformed[0] + CX_tail_damp_q_padded_transformed[0] + CX_tail_damp_r_padded_transformed[0] +
            CX_wing_padded_transformed[0] + CX_wing_damp_p_padded_transformed[0] + CX_wing_damp_q_padded_transformed[0] + CX_wing_damp_r_padded_transformed[0] +
            CX_hover_fuse_padded[0],
            CX_aileron_wing_padded_transformed[1] + CX_elevator_tail_padded_transformed[1] + CX_flap_wing_padded_transformed[1] + CX_rudder_tail_padded_transformed[1] +
            CX_tail_padded_transformed[1] + CX_tail_damp_p_padded_transformed[1] + CX_tail_damp_q_padded_transformed[1] + CX_tail_damp_r_padded_transformed[1] +
            CX_wing_padded_transformed[1]  + CX_wing_damp_p_padded_transformed[1] + CX_wing_damp_q_padded_transformed[1] + CX_wing_damp_r_padded_transformed[1] +
            CX_hover_fuse_padded[1],
            CX_aileron_wing_padded_transformed[2] + CX_elevator_tail_padded_transformed[2] + CX_flap_wing_padded_transformed[2] + CX_rudder_tail_padded_transformed[2] +
            CX_tail_padded_transformed[2] + CX_tail_damp_p_padded_transformed[2] + CX_tail_damp_q_padded_transformed[2] + CX_tail_damp_r_padded_transformed[2] +
            CX_wing_padded_transformed[2] + CX_wing_damp_p_padded_transformed[2] + CX_wing_damp_q_padded_transformed[2] + CX_wing_damp_r_padded_transformed[2] +
            CX_hover_fuse_padded[2]
    ])


    CX_outputs_values_close = jnp.allclose(F1_array, expected_CX_outputs_values, atol=0.0001)
    print("CX_outputs_values_close???", CX_outputs_values_close)
    if not CX_outputs_values_close:
        print(f"\n  Expected: {expected_CX_outputs_values}\n  Got: {F1_array}")
        max_diff_index_CX_outputs_values = jnp.argmax(jnp.abs(F1_array - expected_CX_outputs_values))
        print(f"\n  Max difference in CX_outputs_values at index {max_diff_index_CX_outputs_values}: Expected {expected_CX_outputs_values[max_diff_index_CX_outputs_values]}, Got {F1_array[max_diff_index_CX_outputs_values]}")