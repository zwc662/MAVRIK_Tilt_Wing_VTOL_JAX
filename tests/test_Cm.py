
import pytest

import os
import sys

from jax_mavrik.src.mavrik_aero import MavrikAero, M, interpolate_nd, CM_LOOKUP_TABLES

from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.mavrik_types import StateVariables, ControlInputs, Forces
from jax_mavrik.src.actuator import ActuatorOutput 

from jax_mavrik.src.utils.mat_tools import euler_to_dcm

import jax.numpy as jnp

from .test_mavrik_aero import mavrik_aero, expected_actuator_outputs_values as actuator_outputs_values, expected_Cm_outputs_values as expected_Cm_outputs_values


expected_Cm_alieron_wing_values = jnp.zeros([11])
expected_Cm_elevator_tail_values = jnp.zeros([11])
expected_Cm_flap_wing_values = jnp.zeros([11])
expected_Cm_ruder_tail_values = jnp.zeros([11])
expected_Cm_tail_values = jnp.array([
    -0.468077794679075, -0.446941846020429, -0.419343954303686, -0.386067544798538,
    -0.347910075487998, -0.305672313882372, -0.260148322225623, -0.212116221940555,
    -0.162329798719885, -0.111510998258496, -0.0606417610377775
])
expected_Cm_tail_damp_p_values = jnp.array([
    2.55138631943819e-08, 0.000209415781460261, 0.000834642616666511, 0.00187119223657698,
    0.00331485399470258, 0.00516196639731621, 0.00740968249346905, 0.0100562558530501,
    0.0131013450675675, 0.0165463320446324, 0.0203950015435096
])
expected_Cm_tail_damp_q_values = jnp.array([
    -74.046774596636, -74.0247991563987, -73.9961642674293, -73.9616785605141,
    -73.9221648632473, -73.8784491482376, -73.8313502229573, -73.7816702319254,
    -73.7301860321679, -73.6776414911951, -73.6244657779175
])
expected_Cm_tail_damp_r_values = jnp.array([
    -1.94882411007032e-05, 0.000164886487953645, 0.000715303207994446, 0.00162775186613299,
    0.00289847999067661, 0.00452421163024052, 0.00650238326485106, 0.0088313977199017,
    0.0115108942277843, 0.0145420304404523, 0.0179281497000751
])
expected_Cm_wing_values = jnp.array([
    -1.04615642167448, -1.01978373130211, -0.985247891778407, -0.943535733918178,
    -0.895652272643353, -0.842607180004782, -0.785402128828548, -0.725019100650267,
    -0.662409739916717, -0.598485822026172, -0.534290737334493
])
expected_Cm_wing_damp_p_values = jnp.array([
    0.000645327884073458, 0.000738720199543885, 0.00106809180561898, 0.00163488774938569,
    0.00244162767970418, 0.00349181913920494, 0.00478988833204214, 0.00634113420362827,
    0.00815170941225356, 0.0102286295688548, 0.0125793708992475
])
expected_Cm_wing_damp_q_values = jnp.array([
    -44.8647891387391, -44.869296413032, -44.8752448074649, -44.8824613816276,
    -44.8907698471957, -44.8999929359811, -44.9099546236731, -44.9204821917074,
    -44.9314081116331, -44.9425717387887, -44.9537277158208
])
expected_Cm_wing_damp_r_values = jnp.array([
    0.00920632237564356, 0.00930365538788288, 0.00939262881483325, 0.00947327628502858,
    0.00954582314942758, 0.00961067678655423, 0.00966840931662565, 0.00971973439139478,
    0.00976547977032944, 0.00980655735816595, 0.00991978867521118
])
expected_Cm_hover_fuse_values = jnp.array([
    6.7762635780344e-21, 1.67817489169376e-09, 5.73757414749547e-09, 1.03134554928003e-08,
    1.3114607481053e-08, 1.15509869067377e-08, 2.85243860336423e-09, -1.58223221039936e-08,
    -4.72871487054632e-08, -9.42409215227188e-08, -1.59197240442002e-07
])

expected_Cm_Scale_values = jnp.array([
    64.3408416, 64.1563124177873, 63.9765426199311, 63.8025319115778,
    63.6350062252756, 63.4744181215697, 63.3209596838023, 63.1745854810838,
    63.0350431256754, 62.9019090012365, 62.7747401662053
])

expected_Cm_Scale_p_values = jnp.array([
    0, -0.0232593096761994, -0.0466145109882745, -0.0701390362675629,
    -0.0939031640895908, -0.117975429545505, -0.142424121252733, -0.167318806522902,
    -0.192731833755154, -0.218739768539169, -0.245502793636654
])
 
expected_Cm_Scale_q_values = jnp.array([
    0, -0.0337239706859323, -0.0646055168633447, -0.0924802602785598,
    -0.117234267839899, -0.138800691151852, -0.157156236047169, -0.172317558545883,
    -0.184337652174034, -0.193302265344587, -0.199331845140007
])

expected_Cm_Scale_r_values = jnp.array([
    0.0, 0.000559179098988309, 0.00124005986830336, 0.0021298389383568, 0.00331321931096358,
    0.00487268817949557, 0.00688883018897743, 0.00944069464605581, 0.0126062274849626,
    0.0164627715567274, 0.0210897031804209
])
 
expected_wind_transform = jnp.repeat(jnp.diag(jnp.array([1., 1., 1.]))[None, :, :], 11, axis=0)
expected_tail_transform = jnp.repeat(jnp.diag(jnp.array([1., 1., 1.]))[None, :, :], 11, axis=0) 

expected_Cm_aileron_wing_padded_transformed_values = jnp.zeros([11, 3])
expected_Cm_elevator_tail_padded_transformed_values = jnp.zeros([11, 3])
expected_Cm_flap_wing_padded_transformed_values = jnp.zeros([11, 3])
expected_Cm_ruder_tail_padded_transformed_values = jnp.zeros([11, 3])
expected_Cm_tail_padded_transformed_values = jnp.array([
    [0, -30.1165192439237, 0], [0, -28.6741407058692, 0], [0, -26.8281763649202, 0], [0, -24.6320868470332, 0],
    [0, -22.1392598195149, 0], [0, -19.4023722595574, 0], [0, -16.4728414234575, 0], [0, -13.4003543949081, 0],
    [0, -10.2324658628901, 0], [0, -7.01425466509293, 0], [0, -3.8067707923676, 0]
])
expected_Cm_tail_damp_p_padded_transformed_values = jnp.array([
    [0., 0., 0.], [0, -4.87086651206751e-06, 0], [0, -3.89064574258833e-05, 0], [0, -0.000131243620144855, 0], 
    [0, -0.000311275278597592, 0], [0, -0.000608985203022841, 0], [0, -0.00105531751789409, 0], [0, -0.00168260072742129, 0], 
    [0, -0.00252504625953133, 0], [0, -0.00361934084161512, 0], [0, -0.00500702985515547, 0]
])
expected_Cm_tail_damp_q_padded_transformed_values = jnp.array([
    [0., 0., 0.], [0, 2.49641015678242, 0], [0, 4.78056043840223, 0], [0, 6.83999528391552, 0], 
    [0, 8.6662108748831, 0], [0, 10.2543798030023, 0], [0, 11.6030571033202, 0], [0, 12.7138772798029, 0], 
    [0, 13.5912493875246, 0], [0, 14.2420550054944, 0], [0, 14.6757006109596, 0]
])
expected_Cm_tail_damp_r_padded_transformed_values = jnp.array([
    [0., 0., 0.], [0, 9.2201077769266e-08, 0], [0, 8.87018801902561e-07, 0], [0, 3.466849306473e-06, 0],
    [0, 9.60329987755129e-06, 0], [0, 2.20450725322093e-05, 0], [0, 4.47938141352076e-05, 0], [0, 8.33745291714655e-05, 0], 
    [0, 0.000145108951190792, 0], [0, 0.000239402125112143, 0], [0, 0.000378099355748735, 0]
])
expected_Cm_wing_padded_transformed_values = jnp.array([
    [0, -67.3105846157804, 0], [0, -65.425563663995, 0], [0, -63.0327537395585, 0], [0, -60.1999687730285, 0],
    [0, -56.994837945342, 0], [0, -53.4840004558603, 0], [0, -49.7324165351249, 0], [0, -45.8027811494488, 0],
    [0, -41.7550265225177, 0], [0, -37.6459007156204, 0], [0, -33.5399622093831, 0]
])
expected_Cm_wing_damp_p_padded_transformed_values = jnp.array([
    [0., 0., 0.], [0., -1.71821218852551e-05, 0], [0, -4.97885772095121e-05, 0], [0, -0.000114669451147557, 0],
    [0, -0.000229276564652948, 0], [0, -0.000411948862842917, 0], [0, -0.000682195636589822, 0], [0, -0.00106099100695264, 0], 
    [0, -0.00157109390326278, 0], [0, -0.00223740806436418, 0], [0, -0.00308827069795688, 0]
]) 
expected_Cm_wing_damp_q_padded_transformed_values = jnp.array([
    [0., 0., 0.], [0, 1.5131708369315, 0], [0, 2.8991883851554, 0], [0, 4.15074171051533, 0],
    [0, 5.26273653580541, 0], [0, 6.23215005222745, 0], [0, 7.05787942970561, 0], [0, 7.74058781997885, 0],
    [0, 8.28255028017178, 0], [0, 8.68750092751948, 0], [0, 8.96070949151601, 0]
])  
expected_Cm_wing_damp_r_padded_transformed_values = jnp.array([
    [0., 0., 0.], [0., 5.20240963709407e-06, 0.], [0., 1.16474220511444e-05, 0.], [0., 2.01765527056659e-05, 0.],
    [0., 3.16274055977267e-05, 0.], [0., 4.68298311747953e-05, 0.], [0., 6.66040299797614e-05, 0.], [0., 9.17610444299253e-05, 0.],
    [0., 0.000123105859484573, 0.], [0., 0.00016144311354543, 0.], [0., 0.000209205398772705, 0.]
]) 
 

@pytest.mark.parametrize(
    "id, actuator_outputs_values, \
        expected_Cm_outputs_values, expected_Cm_alieron_wing_values, expected_Cm_elevator_tail_values, expected_Cm_flap_wing_values, expected_Cm_ruder_tail_values, expected_Cm_tail_values, expected_Cm_tail_damp_p_values, expected_Cm_tail_damp_q_values, expected_Cm_tail_damp_r_values, expected_Cm_wing_values, expected_Cm_wing_damp_p_values, expected_Cm_wing_damp_q_values, expected_Cm_wing_damp_r_values, expected_Cm_hover_fuse_values, \
            expected_Cm_Scale_values, expected_Cm_Scale_p_values, expected_Cm_Scale_q_values, expected_Cm_Scale_r_values, \
                expected_wind_transform, expected_tail_transform, \
                expected_Cm_tail_padded_transformed_values, expected_Cm_tail_damp_p_padded_transformed_values, expected_Cm_tail_damp_q_padded_transformed_values, expected_Cm_tail_damp_r_padded_transformed_values, expected_Cm_wing_padded_transformed_values, expected_Cm_wing_damp_p_padded_transformed_values, expected_Cm_wing_damp_q_padded_transformed_values, expected_Cm_wing_damp_r_padded_transformed_values",
    zip(
        list(range(11)), actuator_outputs_values, \
            expected_Cm_outputs_values, expected_Cm_alieron_wing_values, expected_Cm_elevator_tail_values, expected_Cm_flap_wing_values, expected_Cm_ruder_tail_values, expected_Cm_tail_values, expected_Cm_tail_damp_p_values, expected_Cm_tail_damp_q_values, expected_Cm_tail_damp_r_values, expected_Cm_wing_values, expected_Cm_wing_damp_p_values, expected_Cm_wing_damp_q_values, expected_Cm_wing_damp_r_values, expected_Cm_hover_fuse_values, \
            expected_Cm_Scale_values, expected_Cm_Scale_p_values, expected_Cm_Scale_q_values, expected_Cm_Scale_r_values,
            expected_wind_transform, expected_tail_transform, \
            expected_Cm_tail_padded_transformed_values, expected_Cm_tail_damp_p_padded_transformed_values, expected_Cm_tail_damp_q_padded_transformed_values, expected_Cm_tail_damp_r_padded_transformed_values, expected_Cm_wing_padded_transformed_values, expected_Cm_wing_damp_p_padded_transformed_values, expected_Cm_wing_damp_q_padded_transformed_values, expected_Cm_wing_damp_r_padded_transformed_values
    )
)
 
def test_mavrik_aero(id, mavrik_aero, actuator_outputs_values, \
                     expected_Cm_outputs_values, expected_Cm_alieron_wing_values, expected_Cm_elevator_tail_values, expected_Cm_flap_wing_values, expected_Cm_ruder_tail_values, expected_Cm_tail_values, expected_Cm_tail_damp_p_values, expected_Cm_tail_damp_q_values, expected_Cm_tail_damp_r_values, expected_Cm_wing_values, expected_Cm_wing_damp_p_values, expected_Cm_wing_damp_q_values, expected_Cm_wing_damp_r_values, expected_Cm_hover_fuse_values, \
                     expected_Cm_Scale_values, expected_Cm_Scale_p_values, expected_Cm_Scale_q_values, expected_Cm_Scale_r_values,
                     expected_wind_transform, expected_tail_transform, \
                        expected_Cm_tail_padded_transformed_values, expected_Cm_tail_damp_p_padded_transformed_values, expected_Cm_tail_damp_q_padded_transformed_values, expected_Cm_tail_damp_r_padded_transformed_values, expected_Cm_wing_padded_transformed_values, expected_Cm_wing_damp_p_padded_transformed_values, expected_Cm_wing_damp_q_padded_transformed_values, expected_Cm_wing_damp_r_padded_transformed_values):
    u = ActuatorOutput(*actuator_outputs_values)
        
    print(f">>>>>>>>>>>>>>>>>>>> Test ID: {id} <<<<<<<<<<<<<<<<<<<<<<")
    
    wing_transform = jnp.array([[jnp.cos(u.wing_tilt), 0, jnp.sin(u.wing_tilt)], [0, 1, 0], [-jnp.sin(u.wing_tilt), 0., jnp.cos(u.wing_tilt)]])
    tail_transform = jnp.array([[jnp.cos(u.tail_tilt), 0, jnp.sin(u.tail_tilt)], [0, 1, 0], [-jnp.sin(u.tail_tilt), 0., jnp.cos(u.tail_tilt)]])


    M2 = mavrik_aero.M(u, wing_transform, tail_transform)
    M2_array = jnp.array([M2.L, M2.M, M2.N])
    Cm_outputs_values_close = jnp.allclose(M2_array, expected_Cm_outputs_values, atol=0.0001)
    print("Cm_outputs_values_close???", Cm_outputs_values_close)
    if not Cm_outputs_values_close:
        print(f"\n  Expected: {expected_Cm_outputs_values}\n  Got: {M2_array}")
        max_diff_index_Cm_outputs_values = jnp.argmax(jnp.abs(M2_array - expected_Cm_outputs_values))
        print(f"\n  Max difference in Cm_outputs_values at index {max_diff_index_Cm_outputs_values}: Expected {expected_Cm_outputs_values[max_diff_index_Cm_outputs_values]}, Got {M2_array[max_diff_index_Cm_outputs_values]}")
 

    Cm_Scale = 0.5744 * 0.2032 * u.Q
    Cm_Scale_p = 0.5744 * 0.2032 * 2.8270 * 1.225 * 0.25 * u.U * u.p
    Cm_Scale_q = 0.5744 * 0.2032**2 * 1.225 * 0.25 * u.U * u.q
    Cm_Scale_r = 0.5744 * 0.2032 * 2.8270 * 1.225 * 0.25 * u.U * u.r

    
    print("Cm_Scale_close???", jnp.allclose(Cm_Scale, expected_Cm_Scale_values, atol=0.0001))
    if not jnp.allclose(Cm_Scale, expected_Cm_Scale_values, atol=0.0001):
        print(f"\n  Expected: {expected_Cm_Scale_values}\n  Got: {Cm_Scale}")
    print("Cm_Scale_p_close???", jnp.allclose(Cm_Scale_p, expected_Cm_Scale_p_values, atol=0.0001))
    if not jnp.allclose(Cm_Scale_p, expected_Cm_Scale_p_values, atol=0.0001):
        print(f"\n  Expected: {expected_Cm_Scale_p_values}\n  Got: {Cm_Scale_p}") 
    print("Cm_Scale_q_close???", jnp.allclose(Cm_Scale_q, expected_Cm_Scale_q_values, atol=0.0001))
    if not jnp.allclose(Cm_Scale_q, expected_Cm_Scale_q_values, atol=0.0001):
        print(f"\n  Expected: {expected_Cm_Scale_q_values}\n  Got: {Cm_Scale_q}")
    print("Cm_Scale_r_close???", jnp.allclose(Cm_Scale_r, expected_Cm_Scale_r_values, atol=0.0001))
    if not jnp.allclose(Cm_Scale_r, expected_Cm_Scale_r_values, atol=0.0001):
        print(f"\n  Expected: {expected_Cm_Scale_r_values}\n  Got: {Cm_Scale_r}") 
   
    
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
   
        
    Cm_Scale = 0.5744 * 0.2032 * u.Q
    Cm_Scale_p = 0.5744 * 0.2032 * 2.8270 * 1.225 * 0.25 * u.U * u.p
    Cm_Scale_q = 0.5744 * 0.2032**2 * 1.225 * 0.25 * u.U * u.q
    Cm_Scale_r = 0.5744 * 0.2032 * 2.8270 * 1.225 * 0.25 * u.U * u.r
    Cm_lookup_tables = mavrik_aero.Cm_lookup_tables

    Cm_aileron_wing = interpolate_nd(
        jnp.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron]),
        breakpoints=Cm_lookup_tables.Cm_aileron_wing_lookup_table.breakpoints,
        values=Cm_lookup_tables.Cm_aileron_wing_lookup_table.values
    )
    Cm_aileron_wing_padded = jnp.array([0.0, Cm_aileron_wing, 0.0])
    Cm_aileron_wing_padded_transformed = jnp.dot(wing_transform, Cm_aileron_wing_padded * Cm_Scale)

    Cm_elevator_tail = interpolate_nd(
        jnp.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator]),
        breakpoints=Cm_lookup_tables.Cm_elevator_tail_lookup_table.breakpoints,
        values=Cm_lookup_tables.Cm_elevator_tail_lookup_table.values
    )
    Cm_elevator_tail_padded = jnp.array([0.0, Cm_elevator_tail, 0.0])
    Cm_elevator_tail_padded_transformed = jnp.dot(tail_transform, Cm_elevator_tail_padded * Cm_Scale)

    Cm_flap_wing = interpolate_nd(
        jnp.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap]),
        breakpoints=Cm_lookup_tables.Cm_flap_wing_lookup_table.breakpoints,
        values=Cm_lookup_tables.Cm_flap_wing_lookup_table.values
    )
    Cm_flap_wing_padded = jnp.array([0.0, Cm_flap_wing, 0.0])
    Cm_flap_wing_padded_transformed = jnp.dot(wing_transform, Cm_flap_wing_padded * Cm_Scale)

    Cm_rudder_tail = interpolate_nd(
        jnp.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder]),
        breakpoints=Cm_lookup_tables.Cm_rudder_tail_lookup_table.breakpoints,
        values=Cm_lookup_tables.Cm_rudder_tail_lookup_table.values
    )
    Cm_rudder_tail_padded = jnp.array([0.0, Cm_rudder_tail, 0.0])
    Cm_rudder_tail_padded_transformed = jnp.dot(tail_transform, Cm_rudder_tail_padded * Cm_Scale)

    # Tail
    Cm_tail = interpolate_nd(
        jnp.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
        breakpoints=Cm_lookup_tables.Cm_tail_lookup_table.breakpoints,
        values=Cm_lookup_tables.Cm_tail_lookup_table.values
    )
    Cm_tail_padded = jnp.array([0.0, Cm_tail, 0.0])
    Cm_tail_padded_transformed = jnp.dot(tail_transform, Cm_tail_padded * Cm_Scale)

    # Tail Damp p
    Cm_tail_damp_p = interpolate_nd(
        jnp.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
        breakpoints=Cm_lookup_tables.Cm_tail_damp_p_lookup_table.breakpoints,
        values=Cm_lookup_tables.Cm_tail_damp_p_lookup_table.values
    )
    Cm_tail_damp_p_padded = jnp.array([0.0, Cm_tail_damp_p, 0.0])
    Cm_tail_damp_p_padded_transformed = jnp.dot(tail_transform, Cm_tail_damp_p_padded * Cm_Scale_p)

    # Tail Damp q
    Cm_tail_damp_q = interpolate_nd(
        jnp.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
        breakpoints=Cm_lookup_tables.Cm_tail_damp_q_lookup_table.breakpoints,
        values=Cm_lookup_tables.Cm_tail_damp_q_lookup_table.values
    )
    Cm_tail_damp_q_padded = jnp.array([0.0, Cm_tail_damp_q, 0.0])
    Cm_tail_damp_q_padded_transformed = jnp.dot(tail_transform, Cm_tail_damp_q_padded * Cm_Scale_q)

    # Tail Damp r
    Cm_tail_damp_r = interpolate_nd(
        jnp.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
        breakpoints=Cm_lookup_tables.Cm_tail_damp_r_lookup_table.breakpoints,
        values=Cm_lookup_tables.Cm_tail_damp_r_lookup_table.values
    )
    Cm_tail_damp_r_padded = jnp.array([0.0, Cm_tail_damp_r, 0.0])
    Cm_tail_damp_r_padded_transformed = jnp.dot(tail_transform, Cm_tail_damp_r_padded * Cm_Scale_r)

    # Wing
    Cm_wing = interpolate_nd(
        jnp.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
        breakpoints=Cm_lookup_tables.Cm_wing_lookup_table.breakpoints,
        values=Cm_lookup_tables.Cm_wing_lookup_table.values
    )
    Cm_wing_padded = jnp.array([0.0, Cm_wing, 0.0])
    Cm_wing_padded_transformed = jnp.dot(wing_transform, Cm_wing_padded * Cm_Scale)

    # Wing Damp p
    Cm_wing_damp_p = interpolate_nd(
        jnp.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
        breakpoints=Cm_lookup_tables.Cm_wing_damp_p_lookup_table.breakpoints,
        values=Cm_lookup_tables.Cm_wing_damp_p_lookup_table.values
    )
    Cm_wing_damp_p_padded = jnp.array([0.0, Cm_wing_damp_p, 0.0])
    Cm_wing_damp_p_padded_transformed = jnp.dot(wing_transform, Cm_wing_damp_p_padded * Cm_Scale_p)

    # Wing Damp q
    Cm_wing_damp_q = interpolate_nd(
        jnp.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
        breakpoints=Cm_lookup_tables.Cm_wing_damp_q_lookup_table.breakpoints,
        values=Cm_lookup_tables.Cm_wing_damp_q_lookup_table.values
    )
    Cm_wing_damp_q_padded = jnp.array([0.0, Cm_wing_damp_q, 0.0])
    Cm_wing_damp_q_padded_transformed = jnp.dot(wing_transform, Cm_wing_damp_q_padded * Cm_Scale_q)

    # Wing Damp r
    Cm_wing_damp_r = interpolate_nd(
        jnp.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
        breakpoints=Cm_lookup_tables.Cm_wing_damp_r_lookup_table.breakpoints,
        values=Cm_lookup_tables.Cm_wing_damp_r_lookup_table.values
    )
    Cm_wing_damp_r_padded = jnp.array([0.0, Cm_wing_damp_r, 0.0])
    Cm_wing_damp_r_padded_transformed = jnp.dot(wing_transform, Cm_wing_damp_r_padded * Cm_Scale_r)

    # Hover Fuse
    Cm_hover_fuse = interpolate_nd(
        jnp.array([u.U, u.alpha, u.beta]),
        breakpoints=Cm_lookup_tables.Cm_hover_fuse_lookup_table.breakpoints,
        values=Cm_lookup_tables.Cm_hover_fuse_lookup_table.values
    )
    Cm_hover_fuse_padded = jnp.array([0.0, Cm_hover_fuse * Cm_Scale, 0.0])
 


    Cm_aileron_wing_close = jnp.allclose(Cm_aileron_wing, expected_Cm_alieron_wing_values, atol=0.0001)
    print("Cm_aileron_wing_close???", Cm_aileron_wing_close)
    if not Cm_aileron_wing_close:
        print(f"\n  Expected: {expected_Cm_alieron_wing_values}\n  Got: {Cm_aileron_wing}")
        max_diff_index_Cm_aileron_wing = jnp.argmax(jnp.abs(Cm_aileron_wing - expected_Cm_alieron_wing_values))
        print(f"\n  Max difference in Cm_aileron_wing at index {max_diff_index_Cm_aileron_wing}: Expected {expected_Cm_alieron_wing_values[max_diff_index_Cm_aileron_wing]}, Got {Cm_aileron_wing[max_diff_index_Cm_aileron_wing]}")
    #Cm_aileron_wing_padded_transformed_values_close = jnp.allclose(Cm_aileron_wing_padded_transformed, expected_Cm_aileron_wing_padded_transformed_values, atol=0.0001)
    #print("Cm_aileron_wing_padded_transformed_values_close???", Cm_aileron_wing_padded_transformed_values_close)
    
    Cm_elevator_tail_close = jnp.allclose(Cm_elevator_tail, expected_Cm_elevator_tail_values, atol=0.0001)
    print("Cm_elevator_tail_close???", Cm_elevator_tail_close)
    if not Cm_elevator_tail_close:
        print(f"\n  Expected: {expected_Cm_elevator_tail_values}\n  Got: {Cm_elevator_tail}")
        max_diff_index_Cm_elevator_tail = jnp.argmax(jnp.abs(Cm_elevator_tail - expected_Cm_elevator_tail_values))
        print(f"\n  Max difference in Cm_elevator_tail at index {max_diff_index_Cm_elevator_tail}: Expected {expected_Cm_elevator_tail_values[max_diff_index_Cm_elevator_tail]}, Got {Cm_elevator_tail[max_diff_index_Cm_elevator_tail]}")
    #Cm_elevator_tail_padded_transformed_values_close = jnp.allclose(Cm_elevator_tail_padded_transformed, expected_Cm_elevator_tail_padded_transformed_values, atol=0.0001)
    #print("Cm_elevator_tail_padded_transformed_values_close???", Cm_elevator_tail_padded_transformed_values_close)
     
    Cm_flap_wing_close = jnp.allclose(Cm_flap_wing, expected_Cm_flap_wing_values, atol=0.0001)
    print("Cm_flap_wing_close???", Cm_flap_wing_close)
    if not Cm_flap_wing_close:
        print(f"\n  Expected: {expected_Cm_flap_wing_values}\n  Got: {Cm_flap_wing}")
        max_diff_index_Cm_flap_wing = jnp.argmax(jnp.abs(Cm_flap_wing - expected_Cm_flap_wing_values))
        print(f"\n  Max difference in Cm_flap_wing at index {max_diff_index_Cm_flap_wing}: Expected {expected_Cm_flap_wing_values[max_diff_index_Cm_flap_wing]}, Got {Cm_flap_wing[max_diff_index_Cm_flap_wing]}")
    #Cm_flap_wing_padded_transformed_values_close = jnp.allclose(Cm_flap_wing_padded_transformed, expected_Cm_flap_wing_padded_transformed_values, atol=0.0001)  
    #print("Cm_flap_wing_padded_transformed_values_close???", Cm_flap_wing_padded_transformed_values_close)
    
    
    Cm_rudder_tail_close = jnp.allclose(Cm_rudder_tail, expected_Cm_ruder_tail_values, atol=0.0001)
    print("Cm_rudder_tail_close???", Cm_rudder_tail_close)
    if not Cm_rudder_tail_close:
        print(f"\n  Expected: {expected_Cm_ruder_tail_values}\n  Got: {Cm_rudder_tail}")
        max_diff_index_Cm_rudder_tail = jnp.argmax(jnp.abs(Cm_rudder_tail - expected_Cm_ruder_tail_values))
        print(f"\n  Max difference in Cm_rudder_tail at index {max_diff_index_Cm_rudder_tail}: Expected {expected_Cm_ruder_tail_values[max_diff_index_Cm_rudder_tail]}, Got {Cm_rudder_tail[max_diff_index_Cm_rudder_tail]}")
    #Cm_rudder_tail_padded_transformed_values_close = jnp.allclose(Cm_rudder_tail_padded_transformed, expected_Cm_ruder_tail_padded_transformed_values, atol=0.0001)
    #print("Cm_rudder_tail_padded_transformed_values_close???", Cm_rudder_tail_padded_transformed_values_close)
     
    Cm_tail_close = jnp.allclose(Cm_tail, expected_Cm_tail_values, atol=0.0001)
    print("Cm_tail_close???", Cm_tail_close)
    if not Cm_tail_close:
        print(f"\n  Expected: {expected_Cm_tail_values}\n  Got: {Cm_tail}")
        max_diff_index_Cm_tail = jnp.argmax(jnp.abs(Cm_tail - expected_Cm_tail_values))
        print(f"\n  Max difference in Cm_tail at index {max_diff_index_Cm_tail}: Expected {expected_Cm_tail_values[max_diff_index_Cm_tail]}, Got {Cm_tail[max_diff_index_Cm_tail]}")
    Cm_tail_padded_transformed_values_close = jnp.allclose(Cm_tail_padded_transformed, expected_Cm_tail_padded_transformed_values, atol=0.0001)
    print("Cm_tail_padded_transformed_values_close???", Cm_tail_padded_transformed_values_close)
    if not Cm_tail_padded_transformed_values_close:
        print(f"\n  Expected: {expected_Cm_tail_padded_transformed_values}\n  Got: {Cm_tail_padded_transformed}")


    Cm_tail_damp_p_close = jnp.allclose(Cm_tail_damp_p, expected_Cm_tail_damp_p_values, atol=0.0001)
    print("Cm_tail_damp_p_close???", Cm_tail_damp_p_close)
    if not Cm_tail_damp_p_close:
        print(f"\n  Expected: {expected_Cm_tail_damp_p_values}\n  Got: {Cm_tail_damp_p}")
        max_diff_index_Cm_tail_damp_p = jnp.argmax(jnp.abs(Cm_tail_damp_p - expected_Cm_tail_damp_p_values))
        print(f"\n  Max difference in Cm_tail_damp_p at index {max_diff_index_Cm_tail_damp_p}: Expected {expected_Cm_tail_damp_p_values[max_diff_index_Cm_tail_damp_p]}, Got {Cm_tail_damp_p[max_diff_index_Cm_tail_damp_p]}")
    Cm_tail_damp_p_padded_transformed_values_close = jnp.allclose(Cm_tail_damp_p_padded_transformed, expected_Cm_tail_damp_p_padded_transformed_values, atol=0.0001)
    print("Cm_tail_damp_p_padded_transformed_values_close???", Cm_tail_damp_p_padded_transformed_values_close)
     
    Cm_tail_damp_q_close = jnp.allclose(Cm_tail_damp_q, expected_Cm_tail_damp_q_values, atol=0.0001)
    print("Cm_tail_damp_q_close???", Cm_tail_damp_q_close)
    if not Cm_tail_damp_q_close:
        print(f"\n  Expected: {expected_Cm_tail_damp_q_values}\n  Got: {Cm_tail_damp_q}")
        max_diff_index_Cm_tail_damp_q = jnp.argmax(jnp.abs(Cm_tail_damp_q - expected_Cm_tail_damp_q_values))
        print(f"\n  Max difference in Cm_tail_damp_q at index {max_diff_index_Cm_tail_damp_q}: Expected {expected_Cm_tail_damp_q_values[max_diff_index_Cm_tail_damp_q]}, Got {Cm_tail_damp_q[max_diff_index_Cm_tail_damp_q]}")
    Cm_tail_damp_q_padded_transformed_values_close = jnp.allclose(Cm_tail_damp_q_padded_transformed, expected_Cm_tail_damp_q_padded_transformed_values, atol=0.0001)
    print("Cm_tail_damp_q_padded_transformed_values_close???", Cm_tail_damp_q_padded_transformed_values_close)
    if not Cm_tail_damp_q_padded_transformed_values_close:
        print(f"\n  Expected: {expected_Cm_tail_damp_q_padded_transformed_values}\n  Got: {Cm_tail_damp_q_padded_transformed}")
     
    Cm_tail_damp_r_close = jnp.allclose(Cm_tail_damp_r, expected_Cm_tail_damp_r_values, atol=0.0001)
    print("Cm_tail_damp_r_close???", Cm_tail_damp_r_close)
    if not Cm_tail_damp_r_close:
        print(f"\n  Expected: {expected_Cm_tail_damp_r_values}\n  Got: {Cm_tail_damp_r}")
        max_diff_index_Cm_tail_damp_r = jnp.argmax(jnp.abs(Cm_tail_damp_r - expected_Cm_tail_damp_r_values))
        print(f"\n  Max difference in Cm_tail_damp_r at index {max_diff_index_Cm_tail_damp_r}: Expected {expected_Cm_tail_damp_r_values[max_diff_index_Cm_tail_damp_r]}, Got {Cm_tail_damp_r[max_diff_index_Cm_tail_damp_r]}")
    Cm_tail_damp_r_padded_transformed_values_close = jnp.allclose(Cm_tail_damp_r_padded_transformed, expected_Cm_tail_damp_r_padded_transformed_values, atol=0.0001)
    print("Cm_tail_damp_r_padded_transformed_values_close???", Cm_tail_damp_r_padded_transformed_values_close)
     
    Cm_wing_close = jnp.allclose(Cm_wing, expected_Cm_wing_values, atol=0.0001)
    print("Cm_wing_close???", Cm_wing_close)
    if not Cm_wing_close:
        print(f"\n  Expected: {expected_Cm_wing_values}\n  Got: {Cm_wing}")
        max_diff_index_Cm_wing = jnp.argmax(jnp.abs(Cm_wing - expected_Cm_wing_values))
        print(f"\n  Max difference in Cm_wing at index {max_diff_index_Cm_wing}: Expected {expected_Cm_wing_values[max_diff_index_Cm_wing]}, Got {Cm_wing[max_diff_index_Cm_wing]}")
    Cm_wing_padded_transformed_values_close = jnp.allclose(Cm_wing_padded_transformed, expected_Cm_wing_padded_transformed_values, atol=0.0001)
    print("Cm_wing_padded_transformed_values_close???", Cm_wing_padded_transformed_values_close)
    
    Cm_wing_damp_p_close = jnp.allclose(Cm_wing_damp_p, expected_Cm_wing_damp_p_values, atol=0.0001)
    print("Cm_wing_damp_p_close???", Cm_wing_damp_p_close)
    if not Cm_wing_damp_p_close:
        print(f"\n  Expected: {expected_Cm_wing_damp_p_values}\n  Got: {Cm_wing_damp_p}")
        max_diff_index_Cm_wing_damp_p = jnp.argmax(jnp.abs(Cm_wing_damp_p - expected_Cm_wing_damp_p_values))
        print(f"\n  Max difference in Cm_wing_damp_p at index {max_diff_index_Cm_wing_damp_p}: Expected {expected_Cm_wing_damp_p_values[max_diff_index_Cm_wing_damp_p]}, Got {Cm_wing_damp_p[max_diff_index_Cm_wing_damp_p]}")
    Cm_wing_damp_p_padded_transformed_values_close = jnp.allclose(Cm_wing_damp_p_padded_transformed, expected_Cm_wing_damp_p_padded_transformed_values, atol=0.0001)
    print("Cm_wing_damp_p_padded_transformed_values_close???", Cm_wing_damp_p_padded_transformed_values_close)
     
    Cm_wing_damp_q_close = jnp.allclose(Cm_wing_damp_q, expected_Cm_wing_damp_q_values, atol=0.0001)
    print("Cm_wing_damp_q_close???", Cm_wing_damp_q_close)
    if not Cm_wing_damp_q_close:
        print(f"\n  Expected: {expected_Cm_wing_damp_q_values}\n  Got: {Cm_wing_damp_q}")
        max_diff_index_Cm_wing_damp_q = jnp.argmax(jnp.abs(Cm_wing_damp_q - expected_Cm_wing_damp_q_values))
        print(f"\n  Max difference in Cm_wing_damp_q at index {max_diff_index_Cm_wing_damp_q}: Expected {expected_Cm_wing_damp_q_values[max_diff_index_Cm_wing_damp_q]}, Got {Cm_wing_damp_q[max_diff_index_Cm_wing_damp_q]}")
    Cm_wing_damp_q_padded_transformed_values_close = jnp.allclose(Cm_wing_damp_q_padded_transformed, expected_Cm_wing_damp_q_padded_transformed_values, atol=0.0001)
    print("Cm_wing_damp_q_padded_transformed_values_close???", Cm_wing_damp_q_padded_transformed_values_close)
    if not Cm_wing_damp_q_padded_transformed_values_close:
        print(f"\n  Expected: {expected_Cm_wing_damp_q_padded_transformed_values}\n  Got: {Cm_wing_damp_q_padded_transformed}")
     
    Cm_wing_damp_r_close = jnp.allclose(Cm_wing_damp_r, expected_Cm_wing_damp_r_values, atol=0.0001)
    print("Cm_wing_damp_r_close???", Cm_wing_damp_r_close)
    if not Cm_wing_damp_r_close:
        print(f"\n  Expected: {expected_Cm_wing_damp_r_values}\n  Got: {Cm_wing_damp_r}")
        max_diff_index_Cm_wing_damp_r = jnp.argmax(jnp.abs(Cm_wing_damp_r - expected_Cm_wing_damp_r_values))
        print(f"\n  Max difference in Cm_wing_damp_r at index {max_diff_index_Cm_wing_damp_r}: Expected {expected_Cm_wing_damp_r_values[max_diff_index_Cm_wing_damp_r]}, Got {Cm_wing_damp_r[max_diff_index_Cm_wing_damp_r]}")
    Cm_wing_damp_r_padded_transformed_values_close = jnp.allclose(Cm_wing_damp_r_padded_transformed, expected_Cm_wing_damp_r_padded_transformed_values, atol=0.0001)
    print("Cm_wing_damp_r_padded_transformed_values_close???", Cm_wing_damp_r_padded_transformed_values_close)
    if not Cm_wing_damp_r_padded_transformed_values_close:
        print(f"\n  Expected: {expected_Cm_wing_damp_r_padded_transformed_values}\n  Got: {Cm_wing_damp_r_padded_transformed}")

    Cm_hover_fuse_close = jnp.allclose(Cm_hover_fuse, expected_Cm_hover_fuse_values, atol=0.0001)
    print("Cm_hover_fuse_close???", Cm_hover_fuse_close)
    if not Cm_hover_fuse_close:
        print(f"\n  Expected: {expected_Cm_hover_fuse_values}\n  Got: {Cm_hover_fuse}")
        max_diff_index_Cm_hover_fuse = jnp.argmax(jnp.abs(Cm_hover_fuse - expected_Cm_hover_fuse_values))
        print(f"\n  Max difference in Cm_hover_fuse at index {max_diff_index_Cm_hover_fuse}: Expected {expected_Cm_hover_fuse_values[max_diff_index_Cm_hover_fuse]}, Got {Cm_hover_fuse[max_diff_index_Cm_hover_fuse]}")


    
    M2_array = jnp.array([
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
    ])


    Cm_outputs_values_close = jnp.allclose(M2_array, expected_Cm_outputs_values, atol=0.0001)
    print("Cm_outputs_values_close???", Cm_outputs_values_close)
    if not Cm_outputs_values_close:
        print(f"\n  Expected: {expected_Cm_outputs_values}\n  Got: {M2_array}")
        max_diff_index_Cm_outputs_values = jnp.argmax(jnp.abs(M2_array - expected_Cm_outputs_values))
        print(f"\n  Max difference in Cm_outputs_values at index {max_diff_index_Cm_outputs_values}: Expected {expected_Cm_outputs_values[max_diff_index_Cm_outputs_values]}, Got {M2_array[max_diff_index_Cm_outputs_values]}")
    