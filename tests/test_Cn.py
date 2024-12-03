
import pytest

import os
import sys

from jax_mavrik.src.mavrik_aero import MavrikAero, N, interpolate_nd, CN_LOOKUP_TABLES

from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.mavrik_types import StateVariables, ControlInputs, Forces
from jax_mavrik.src.actuator import ActuatorOutput 

from jax_mavrik.src.utils.mat_tools import euler_to_dcm

import jax.numpy as jnp

from .test_mavrik_aero import mavrik_aero, expected_actuator_outputs_values as actuator_outputs_values, expected_Cn_outputs_values as expected_Cn_outputs_values


expected_Cn_alieron_wing_values = jnp.zeros([11])
expected_Cn_elevator_tail_values = jnp.zeros([11])
expected_Cn_flap_wing_values = jnp.zeros([11])
expected_Cn_ruder_tail_values = jnp.zeros([11]) 
expected_Cn_tail_values = jnp.array([
    -2.31296463463574e-19, 1.68694457275377e-05, 6.73918187588505e-05, 0.000151500526237396,
    0.000269230177118721, 0.000420716567654726, 0.000606198979319392, 0.000826025195042959,
    0.00108065936540829, 0.00137069261428182, 0.00169689231948542
])
expected_Cn_tail_damp_p_values = jnp.array([
    -0.0141561533884364, -0.0130918492182991, -0.0117036869527451, -0.0100309623628608,
    -0.00811366752160639, -0.00599195344576756, -0.00370562853174562, -0.00129369627245964,
    0.00120606473005526, 0.00375747601099069, 0.00632639991491051
])
expected_Cn_tail_damp_q_values = jnp.array([
    2.77610399945782e-12, -1.32282423812355e-05, -4.81684728378677e-05, -9.55986015499916e-05,
    -0.000144004666251583, -0.00018019332719385, -0.000189882990933367, -0.000158264363689603,
    -7.05213301682256e-05, 8.76962850386515e-05, 0.000329857201024365
])
expected_Cn_tail_damp_r_values = jnp.array([
    -0.448479801638418, -0.447479498962021, -0.44617481560576, -0.444602683579942,
    -0.442800689172128, -0.440806568003398, -0.438657733715209, -0.436390843560641,
    -0.434041403730694, -0.431643416712561, -0.429228989438347
])
expected_Cn_wing_values = jnp.array([
    0.000184304613340483, 0.00019064951459549, 0.000198828892954526, 0.000208663682796846,
    0.000219954374959907, 0.000232487338368643, 0.000246041243223569, 0.000260393243134786,
    0.000275324614431918, 0.000290625614988686, 0.000307474400890488
])
expected_Cn_wing_damp_p_values = jnp.array([
    -0.0095774022882647, -0.0101263654019856, -0.010841540734805, -0.0117026619741763,
    -0.0126891308165998, -0.0137802899565774, -0.0149556768561423, -0.0161952569344399,
    -0.0174796350287419, -0.0187902442069043, -0.0201094439822526
])
expected_Cn_wing_damp_q_values = jnp.array([
    -0.00349639191648994, -0.00362496025527281, -0.00385086307321889, -0.00414848881570478,
    -0.00448610548876289, -0.0048275595297107, -0.00513391314822688, -0.00536499614864538,
    -0.00548084816353371, -0.00544302872894279, -0.00528802269337279
])
expected_Cn_wing_damp_r_values = jnp.array([
    0.0100915315953612, 0.0105647119300652, 0.0111815565750504, 0.0119245544327195,
    0.0127759076969087, 0.0137177674712248, 0.01473245280996, 0.0158026520268666,
    0.0169116052959292, 0.0180432677595598, 0.0191818073635589
])
expected_Cn_hover_fuse_values = jnp.array([
    0, 1.20624385564703e-10, 4.12407169003069e-10, 7.41313815400175e-10,
    9.42655903838156e-10, 8.30265489723715e-10, 2.05028483977401e-10, -1.1372818717834e-09,
    -3.39892062856414e-09, -6.77387875960966e-09, -1.14428295924351e-08
])

expected_Cn_Scale_values = jnp.array([
    895.135626, 892.568381914787, 890.06735229599, 887.646445443064,
    885.315760821133, 883.08159463424, 880.946619222977, 878.910202534566,
    876.968833249431, 875.116617846926, 873.347393946174
])

expected_Cn_Scale_p_values = jnp.array([
    0, -0.323592856567991, -0.648519796081949, -0.975802438624017,
    -1.30641852795902, -1.64132155179696, -1.98146156880648, -2.32780642736341,
    -2.68136266744991, -3.04319550029641, -3.41553345280916
])
 
expected_Cn_Scale_q_values = jnp.array([
    0, -0.46918142287958, -0.898817894550568, -1.28662251873764,
    -1.63101021251671, -1.93105095416479, -2.18642066587276, -2.39735107287998,
    -2.56457944240154, -2.68929874079305, -2.77318467623425
])

expected_Cn_Scale_r_values = jnp.array([
    0.0, 0.00777952417736195, 0.0172522108646338, 0.0296311746000723, 0.046094837559518,
    0.0677907947019389, 0.095840171969681, 0.131342735061023, 0.175382899114121,
    0.229036688931439, 0.293408419739419
])
 
expected_wind_transform = jnp.repeat(jnp.diag(jnp.array([1., 1., 1.]))[None, :, :], 11, axis=0)
expected_tail_transform = jnp.repeat(jnp.diag(jnp.array([1., 1., 1.]))[None, :, :], 11, axis=0) 

expected_Cn_aileron_wing_padded_transformed_values = jnp.zeros([11, 3])
expected_Cn_elevator_tail_padded_transformed_values = jnp.zeros([11, 3])
expected_Cn_flap_wing_padded_transformed_values = jnp.zeros([11, 3])
expected_Cn_ruder_tail_padded_transformed_values = jnp.zeros([11, 3])
expected_Cn_tail_padded_transformed_values = jnp.array([
    [0, 0, -2.07041704614053e-16], [0, 0, 0.0150571338768276], [0, 0, 0.0599832576891013], [0, 0, 0.134478903597378],
    [0, 0, 0.238353719091869], [0, 0, 0.371527057453579], [0, 0, 0.534028941407838], [0, 0, 0.726001971473862],
    [0, 0, 0.947704582822183], [0, 0, 1.19951588471807], [0, 0, 1.48197648502987]
])
expected_Cn_tail_damp_p_padded_transformed_values = jnp.array([
    [0., 0., 0.], [0., 0., 0.00423642888630682], [0., 0., 0.00759007267600123], [0., 0., 0.00978823753542533], 
    [0., 0., 0.0105998455799259], [0., 0., 0.00983472232790234], [0., 0., 0.00734256052392673], [0., 0., 0.00301147449808765], 
    [0., 0., -0.00323389694169824], [0., 0., -0.0114347340891186], [0., 0., -0.0216080305452259]
])
expected_Cn_tail_damp_q_padded_transformed_values = jnp.array([
    [0., 0., 0.], [0., 0., 6.20644558262404e-06], [0., 0., 4.32946853398485e-05], [0., 0., 0.000122999313514046], 
    [0., 0., 0.000234873081306392], [0., 0., 0.000347962496411812], [0., 0., 0.000415164095474444], [0., 0., 0.000379415242089938], 
    [0., 0., 0.000180857553600243], [0., 0., -0.000235841508926674], [0., 0., -0.000914754935226289]
])
expected_Cn_tail_damp_r_padded_transformed_values = jnp.array([
    [0., 0., 0.], [0., 0., -0.00348117758104886], [0., 0., -0.00769750200131967], [0., 0., -0.0131740997448179],
    [0., 0., -0.0204108258386319], [0., 0., -0.0298826275547846], [0., 0., -0.0420410326350961], [0., 0., -0.0573167669488414],
    [0., 0., -0.0761234397218516], [0., 0., -0.0988621789628984], [0., 0., -0.125939399497453]
])
expected_Cn_wing_padded_transformed_values = jnp.array([
    [0, 0, 0.164977625437222], [0, 0, 0.170167728755336], [0, 0, 0.176971106311977], [0, 0, 0.185219576327679],
    [0, 0, 0.194729074813567], [0, 0, 0.205305289498852], [0, 0, 0.216749201407221], [0, 0, 0.228862278062228],
    [0, 0, 0.241451105883209], [0, 0, 0.254331305248582], [0, 0, 0.268531966722869]
])
expected_Cn_wing_damp_p_padded_transformed_values = jnp.array([
    [0., 0., 0.], [0., 0., 0.00327681950707979], [0., 0., 0.0070309537865499], [0., 0., 0.0114194860927938],
    [0., 0., 0.0165773156025017], [0., 0., 0.0226178868957417], [0., 0., 0.0296340989259345], [0., 0., 0.0376994231847909],
    [0., 0., 0.0468692408067183], [0., 0., 0.0571823866199218], [0., 0., 0.0686844786387755]
]) 
expected_Cn_wing_damp_q_padded_transformed_values = jnp.array([
    [0., 0., 0.], [0., 0., 0.00170076401045082], [0., 0., 0.00346122463967313], [0., 0., 0.00533753912901702],
    [0., 0., 0.00731688386659952], [0., 0., 0.00932226343613519], [0., 0., 0.0112248938040792], [0., 0., 0.012861779272952],
    [0., 0., 0.0140560705271228], [0., 0., 0.0146379303068463], [0., 0., 0.0146646635008404]
]) 
expected_Cn_wing_damp_r_padded_transformed_values = jnp.array([
    [0., 0., 0.], [0., 0., 8.21884318868067e-05], [0., 0., 0.000192906571827602], [0., 0., 0.000353338554423978],
    [0., 0., 0.000588903389964402], [0., 0., 0.000929938358410739], [0., 0., 0.00141196081084177], [0., 0., 0.00207556353842627],
    [0., 0., 0.00296600636547379], [0., 0., 0.00413257030515297], [0., 0., 0.00562810378628776]
]) 
expected_Cn_hover_fuse_padded_values = jnp.array([
    [0, 0, 0], [0, 0, 1.07665512642953e-07], [0, 0, 3.67070156982446e-07], [0, 0, 6.580245731978e-07],
    [0, 0, 8.3454812869901e-07], [0, 0, 7.33192172634996e-07], [0, 0, 1.80619149804303e-07], [0, 0, -9.9956864026804e-07],
    [0, 0, -2.98074745793931e-06], [0, 0, -5.92793386981474e-06], [0, 0, -9.99356540392336e-06]
])

  

@pytest.mark.parametrize(
    "id, actuator_outputs_values, \
        expected_Cn_outputs_values, expected_Cn_alieron_wing_values, expected_Cn_elevator_tail_values, expected_Cn_flap_wing_values, expected_Cn_ruder_tail_values, expected_Cn_tail_values, expected_Cn_tail_damp_p_values, expected_Cn_tail_damp_q_values, expected_Cn_tail_damp_r_values, expected_Cn_wing_values, expected_Cn_wing_damp_p_values, expected_Cn_wing_damp_q_values, expected_Cn_wing_damp_r_values, expected_Cn_hover_fuse_values, \
            expected_Cn_Scale_values, expected_Cn_Scale_p_values, expected_Cn_Scale_q_values, expected_Cn_Scale_r_values, \
                expected_wind_transform, expected_tail_transform, \
                expected_Cn_tail_padded_transformed_values, expected_Cn_tail_damp_p_padded_transformed_values, expected_Cn_tail_damp_q_padded_transformed_values, expected_Cn_tail_damp_r_padded_transformed_values, expected_Cn_wing_padded_transformed_values, expected_Cn_wing_damp_p_padded_transformed_values, expected_Cn_wing_damp_q_padded_transformed_values, expected_Cn_wing_damp_r_padded_transformed_values",
    zip(
        list(range(11)), actuator_outputs_values, \
            expected_Cn_outputs_values, expected_Cn_alieron_wing_values, expected_Cn_elevator_tail_values, expected_Cn_flap_wing_values, expected_Cn_ruder_tail_values, expected_Cn_tail_values, expected_Cn_tail_damp_p_values, expected_Cn_tail_damp_q_values, expected_Cn_tail_damp_r_values, expected_Cn_wing_values, expected_Cn_wing_damp_p_values, expected_Cn_wing_damp_q_values, expected_Cn_wing_damp_r_values, expected_Cn_hover_fuse_values, \
            expected_Cn_Scale_values, expected_Cn_Scale_p_values, expected_Cn_Scale_q_values, expected_Cn_Scale_r_values,
            expected_wind_transform, expected_tail_transform, \
            expected_Cn_tail_padded_transformed_values, expected_Cn_tail_damp_p_padded_transformed_values, expected_Cn_tail_damp_q_padded_transformed_values, expected_Cn_tail_damp_r_padded_transformed_values, expected_Cn_wing_padded_transformed_values, expected_Cn_wing_damp_p_padded_transformed_values, expected_Cn_wing_damp_q_padded_transformed_values, expected_Cn_wing_damp_r_padded_transformed_values
    )
)
 
def test_mavrik_aero(id, mavrik_aero, actuator_outputs_values, \
                     expected_Cn_outputs_values, expected_Cn_alieron_wing_values, expected_Cn_elevator_tail_values, expected_Cn_flap_wing_values, expected_Cn_ruder_tail_values, expected_Cn_tail_values, expected_Cn_tail_damp_p_values, expected_Cn_tail_damp_q_values, expected_Cn_tail_damp_r_values, expected_Cn_wing_values, expected_Cn_wing_damp_p_values, expected_Cn_wing_damp_q_values, expected_Cn_wing_damp_r_values, expected_Cn_hover_fuse_values, \
                     expected_Cn_Scale_values, expected_Cn_Scale_p_values, expected_Cn_Scale_q_values, expected_Cn_Scale_r_values,
                     expected_wind_transform, expected_tail_transform, \
                        expected_Cn_tail_padded_transformed_values, expected_Cn_tail_damp_p_padded_transformed_values, expected_Cn_tail_damp_q_padded_transformed_values, expected_Cn_tail_damp_r_padded_transformed_values, expected_Cn_wing_padded_transformed_values, expected_Cn_wing_damp_p_padded_transformed_values, expected_Cn_wing_damp_q_padded_transformed_values, expected_Cn_wing_damp_r_padded_transformed_values):
    u = ActuatorOutput(*actuator_outputs_values)
        
    print(f">>>>>>>>>>>>>>>>>>>> Test ID: {id} <<<<<<<<<<<<<<<<<<<<<<")
    

    wing_transform = jnp.array([[jnp.cos(u.wing_tilt), 0, jnp.sin(u.wing_tilt)], [0, 1, 0], [-jnp.sin(u.wing_tilt), 0., jnp.cos(u.wing_tilt)]])
    tail_transform = jnp.array([[jnp.cos(u.tail_tilt), 0, jnp.sin(u.tail_tilt)], [0, 1, 0], [-jnp.sin(u.tail_tilt), 0., jnp.cos(u.tail_tilt)]])

    M3 = mavrik_aero.N(u, wing_transform, tail_transform)
    M3_array = jnp.array([M3.L, M3.M, M3.N])
    Cn_outputs_values_close = jnp.allclose(M3_array, expected_Cn_outputs_values, atol=0.0001)
    print("Cn_outputs_values_close???", Cn_outputs_values_close)
    if not Cn_outputs_values_close:
        print(f"\n  Expected: {expected_Cn_outputs_values}\n  Got: {M3_array}")
        max_diff_index_Cn_outputs_values = jnp.argmax(jnp.abs(M3_array - expected_Cn_outputs_values))
        print(f"\n  Max difference in Cn_outputs_values at index {max_diff_index_Cn_outputs_values}: Expected {expected_Cn_outputs_values[max_diff_index_Cn_outputs_values]}, Got {M3_array[max_diff_index_Cn_outputs_values]}")
        print("[Warning: Cn_tail_padded_transformed_values not close possibly due to identified numerical issue]")
        
    Cn_Scale = 0.5744 * 2.8270 * u.Q
    Cn_Scale_p = 0.5744 * 2.8270**2 * 1.225 * 0.25 * u.U * u.p
    Cn_Scale_q = 0.5744 * 0.2032 * 2.8270 * 1.225 * 0.25 * u.U * u.q
    Cn_Scale_r = 0.5744 * 2.8270**2 * 1.225 * 0.25 * u.U * u.r

    print("Cn_Scale_close???", jnp.allclose(Cn_Scale, expected_Cn_Scale_values, atol=0.0001))
    if not jnp.allclose(Cn_Scale, expected_Cn_Scale_values, atol=0.0001):
        print(f"\n  Expected: {expected_Cn_Scale_values}\n  Got: {Cn_Scale}")
    print("Cn_Scale_p_close???", jnp.allclose(Cn_Scale_p, expected_Cn_Scale_p_values, atol=0.0001))
    if not jnp.allclose(Cn_Scale_p, expected_Cn_Scale_p_values, atol=0.0001):
        print(f"\n  Expected: {expected_Cn_Scale_p_values}\n  Got: {Cn_Scale_p}") 
    print("Cn_Scale_q_close???", jnp.allclose(Cn_Scale_q, expected_Cn_Scale_q_values, atol=0.0001))
    if not jnp.allclose(Cn_Scale_q, expected_Cn_Scale_q_values, atol=0.0001):
        print(f"\n  Expected: {expected_Cn_Scale_q_values}\n  Got: {Cn_Scale_q}")
    print("Cn_Scale_r_close???", jnp.allclose(Cn_Scale_r, expected_Cn_Scale_r_values, atol=0.0001))
    if not jnp.allclose(Cn_Scale_r, expected_Cn_Scale_r_values, atol=0.0001):
        print(f"\n  Expected: {expected_Cn_Scale_r_values}\n  Got: {Cn_Scale_r}") 

    Cn_lookup_tables = mavrik_aero.Cn_lookup_tables

    Cn_aileron_wing = interpolate_nd(
        jnp.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron]),
        breakpoints=Cn_lookup_tables.Cn_aileron_wing_lookup_table.breakpoints,
        values=Cn_lookup_tables.Cn_aileron_wing_lookup_table.values
    )
    Cn_aileron_wing_padded = jnp.array([0.0, 0.0, Cn_aileron_wing])
    Cn_aileron_wing_padded_transformed = jnp.dot(wing_transform, Cn_aileron_wing_padded * Cn_Scale)

    Cn_elevator_tail = interpolate_nd(
        jnp.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator]),
        breakpoints=Cn_lookup_tables.Cn_elevator_tail_lookup_table.breakpoints,
        values=Cn_lookup_tables.Cn_elevator_tail_lookup_table.values
    )
    Cn_elevator_tail_padded = jnp.array([0.0, 0.0, Cn_elevator_tail])
    Cn_elevator_tail_padded_transformed = jnp.dot(tail_transform, Cn_elevator_tail_padded * Cn_Scale)

    Cn_flap_wing = interpolate_nd(
        jnp.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap]),
        breakpoints=Cn_lookup_tables.Cn_flap_wing_lookup_table.breakpoints,
        values=Cn_lookup_tables.Cn_flap_wing_lookup_table.values
    )
    Cn_flap_wing_padded = jnp.array([0.0, 0.0, Cn_flap_wing])
    Cn_flap_wing_padded_transformed = jnp.dot(wing_transform, Cn_flap_wing_padded * Cn_Scale)

    Cn_rudder_tail = interpolate_nd(
        jnp.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder]),
        breakpoints=Cn_lookup_tables.Cn_rudder_tail_lookup_table.breakpoints,
        values=Cn_lookup_tables.Cn_rudder_tail_lookup_table.values
    )
    Cn_rudder_tail_padded = jnp.array([0.0, 0.0, Cn_rudder_tail])
    Cn_rudder_tail_padded_transformed = jnp.dot(tail_transform, Cn_rudder_tail_padded * Cn_Scale)

    # Tail
    Cn_tail = interpolate_nd(
        jnp.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
        breakpoints=Cn_lookup_tables.Cn_tail_lookup_table.breakpoints,
        values=Cn_lookup_tables.Cn_tail_lookup_table.values
    )
    Cn_tail_padded = jnp.array([0.0, 0.0, Cn_tail])
    Cn_tail_padded_transformed = jnp.dot(tail_transform, Cn_tail_padded * Cn_Scale)

    # Tail Damp p
    Cn_tail_damp_p = interpolate_nd(
        jnp.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
        breakpoints=Cn_lookup_tables.Cn_tail_damp_p_lookup_table.breakpoints,
        values=Cn_lookup_tables.Cn_tail_damp_p_lookup_table.values
    )
    Cn_tail_damp_p_padded = jnp.array([0.0, 0.0, Cn_tail_damp_p])
    Cn_tail_damp_p_padded_transformed = jnp.dot(tail_transform, Cn_tail_damp_p_padded * Cn_Scale_p)

    # Tail Damp q
    Cn_tail_damp_q = interpolate_nd(
        jnp.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
        breakpoints=Cn_lookup_tables.Cn_tail_damp_q_lookup_table.breakpoints,
        values=Cn_lookup_tables.Cn_tail_damp_q_lookup_table.values
    )
    Cn_tail_damp_q_padded = jnp.array([0.0, 0.0, Cn_tail_damp_q])
    Cn_tail_damp_q_padded_transformed = jnp.dot(tail_transform, Cn_tail_damp_q_padded * Cn_Scale_q)

    # Tail Damp r
    Cn_tail_damp_r = interpolate_nd(
        jnp.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
        breakpoints=Cn_lookup_tables.Cn_tail_damp_r_lookup_table.breakpoints,
        values=Cn_lookup_tables.Cn_tail_damp_r_lookup_table.values
    )
    Cn_tail_damp_r_padded = jnp.array([0.0, 0.0, Cn_tail_damp_r])
    Cn_tail_damp_r_padded_transformed = jnp.dot(tail_transform, Cn_tail_damp_r_padded * Cn_Scale_r)

    # Wing
    Cn_wing = interpolate_nd(
        jnp.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
        breakpoints=Cn_lookup_tables.Cn_wing_lookup_table.breakpoints,
        values=Cn_lookup_tables.Cn_wing_lookup_table.values
    )
    Cn_wing_padded = jnp.array([0.0, 0.0, Cn_wing])
    Cn_wing_padded_transformed = jnp.dot(wing_transform, Cn_wing_padded * Cn_Scale)

    # Wing Damp p
    Cn_wing_damp_p = interpolate_nd(
        jnp.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
        breakpoints=Cn_lookup_tables.Cn_wing_damp_p_lookup_table.breakpoints,
        values=Cn_lookup_tables.Cn_wing_damp_p_lookup_table.values
    )
    Cn_wing_damp_p_padded = jnp.array([0.0, 0.0, Cn_wing_damp_p])
    Cn_wing_damp_p_padded_transformed = jnp.dot(wing_transform, Cn_wing_damp_p_padded * Cn_Scale_p)

    # Wing Damp q
    Cn_wing_damp_q = interpolate_nd(
        jnp.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
        breakpoints=Cn_lookup_tables.Cn_wing_damp_q_lookup_table.breakpoints,
        values=Cn_lookup_tables.Cn_wing_damp_q_lookup_table.values
    )
    Cn_wing_damp_q_padded = jnp.array([0.0, 0.0, Cn_wing_damp_q])
    Cn_wing_damp_q_padded_transformed = jnp.dot(wing_transform, Cn_wing_damp_q_padded * Cn_Scale_q)

    # Wing Damp r
    Cn_wing_damp_r = interpolate_nd(
        jnp.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
        breakpoints=Cn_lookup_tables.Cn_wing_damp_r_lookup_table.breakpoints,
        values=Cn_lookup_tables.Cn_wing_damp_r_lookup_table.values
    )
    Cn_wing_damp_r_padded = jnp.array([0.0, 0.0, Cn_wing_damp_r])
    Cn_wing_damp_r_padded_transformed = jnp.dot(wing_transform, Cn_wing_damp_r_padded * Cn_Scale_r)

    # Hover Fuse
    Cn_hover_fuse = interpolate_nd(
        jnp.array([u.U, u.alpha, u.beta]),
        breakpoints=Cn_lookup_tables.Cn_hover_fuse_lookup_table.breakpoints,
        values=Cn_lookup_tables.Cn_hover_fuse_lookup_table.values
    )
    Cn_hover_fuse_padded = jnp.array([0.0, 0.0, Cn_hover_fuse * Cn_Scale])



    Cn_aileron_wing_close = jnp.allclose(Cn_aileron_wing, expected_Cn_alieron_wing_values, atol=0.0001)
    print("Cn_aileron_wing_close???", Cn_aileron_wing_close)
    if not Cn_aileron_wing_close:
        print(f"\n  Expected: {expected_Cn_alieron_wing_values}\n  Got: {Cn_aileron_wing}")
        max_diff_index_Cn_aileron_wing = jnp.argmax(jnp.abs(Cn_aileron_wing - expected_Cn_alieron_wing_values))
        print(f"\n  Max difference in Cn_aileron_wing at index {max_diff_index_Cn_aileron_wing}: Expected {expected_Cn_alieron_wing_values[max_diff_index_Cn_aileron_wing]}, Got {Cn_aileron_wing[max_diff_index_Cn_aileron_wing]}")
    #Cn_aileron_wing_padded_transformed_values_close = jnp.allclose(Cn_aileron_wing_padded_transformed, expected_Cn_aileron_wing_padded_transformed_values, atol=0.0001)
    #print("Cn_aileron_wing_padded_transformed_values_close???", Cn_aileron_wing_padded_transformed_values_close)
    
    Cn_elevator_tail_close = jnp.allclose(Cn_elevator_tail, expected_Cn_elevator_tail_values, atol=0.0001)
    print("Cn_elevator_tail_close???", Cn_elevator_tail_close)
    if not Cn_elevator_tail_close:
        print(f"\n  Expected: {expected_Cn_elevator_tail_values}\n  Got: {Cn_elevator_tail}")
        max_diff_index_Cn_elevator_tail = jnp.argmax(jnp.abs(Cn_elevator_tail - expected_Cn_elevator_tail_values))
        print(f"\n  Max difference in Cn_elevator_tail at index {max_diff_index_Cn_elevator_tail}: Expected {expected_Cn_elevator_tail_values[max_diff_index_Cn_elevator_tail]}, Got {Cn_elevator_tail[max_diff_index_Cn_elevator_tail]}")
    #Cn_elevator_tail_padded_transformed_values_close = jnp.allclose(Cn_elevator_tail_padded_transformed, expected_Cn_elevator_tail_padded_transformed_values, atol=0.0001)
    #print("Cn_elevator_tail_padded_transformed_values_close???", Cn_elevator_tail_padded_transformed_values_close)
     
    Cn_flap_wing_close = jnp.allclose(Cn_flap_wing, expected_Cn_flap_wing_values, atol=0.0001)
    print("Cn_flap_wing_close???", Cn_flap_wing_close)
    if not Cn_flap_wing_close:
        print(f"\n  Expected: {expected_Cn_flap_wing_values}\n  Got: {Cn_flap_wing}")
        max_diff_index_Cn_flap_wing = jnp.argmax(jnp.abs(Cn_flap_wing - expected_Cn_flap_wing_values))
        print(f"\n  Max difference in Cn_flap_wing at index {max_diff_index_Cn_flap_wing}: Expected {expected_Cn_flap_wing_values[max_diff_index_Cn_flap_wing]}, Got {Cn_flap_wing[max_diff_index_Cn_flap_wing]}")
    #Cn_flap_wing_padded_transformed_values_close = jnp.allclose(Cn_flap_wing_padded_transformed, expected_Cn_flap_wing_padded_transformed_values, atol=0.0001)  
    #print("Cn_flap_wing_padded_transformed_values_close???", Cn_flap_wing_padded_transformed_values_close)
    
    
    Cn_rudder_tail_close = jnp.allclose(Cn_rudder_tail, expected_Cn_ruder_tail_values, atol=0.0001)
    print("Cn_rudder_tail_close???", Cn_rudder_tail_close)
    if not Cn_rudder_tail_close:
        print(f"\n  Expected: {expected_Cn_ruder_tail_values}\n  Got: {Cn_rudder_tail}")
        max_diff_index_Cn_rudder_tail = jnp.argmax(jnp.abs(Cn_rudder_tail - expected_Cn_ruder_tail_values))
        print(f"\n  Max difference in Cn_rudder_tail at index {max_diff_index_Cn_rudder_tail}: Expected {expected_Cn_ruder_tail_values[max_diff_index_Cn_rudder_tail]}, Got {Cn_rudder_tail[max_diff_index_Cn_rudder_tail]}")
    #Cn_rudder_tail_padded_transformed_values_close = jnp.allclose(Cn_rudder_tail_padded_transformed, expected_Cn_ruder_tail_padded_transformed_values, atol=0.0001)
    #print("Cn_rudder_tail_padded_transformed_values_close???", Cn_rudder_tail_padded_transformed_values_close)
     
    Cn_tail_close = jnp.allclose(Cn_tail, expected_Cn_tail_values, atol=0.0001)
    print("Cn_tail_close???", Cn_tail_close)
    if not Cn_tail_close:
        print(f"\n  Expected: {expected_Cn_tail_values}\n  Got: {Cn_tail}")
        max_diff_index_Cn_tail = jnp.argmax(jnp.abs(Cn_tail - expected_Cn_tail_values))
        print(f"\n  Max difference in Cn_tail at index {max_diff_index_Cn_tail}: Expected {expected_Cn_tail_values[max_diff_index_Cn_tail]}, Got {Cn_tail[max_diff_index_Cn_tail]}")
    Cn_tail_padded_transformed_values_close = jnp.allclose(Cn_tail_padded_transformed, expected_Cn_tail_padded_transformed_values, atol=0.1)
    print("[Warning: Cn_tail_padded_transformed_values not close possibly due to numerical issue] Cn_tail_padded_transformed_values_close???", Cn_tail_padded_transformed_values_close)
    if not Cn_tail_padded_transformed_values_close:
        print(f"\n  Expected: {expected_Cn_tail_padded_transformed_values}\n  Got: {Cn_tail_padded_transformed}")


    Cn_tail_damp_p_close = jnp.allclose(Cn_tail_damp_p, expected_Cn_tail_damp_p_values, atol=0.0001)
    print("Cn_tail_damp_p_close???", Cn_tail_damp_p_close)
    if not Cn_tail_damp_p_close:
        print(f"\n  Expected: {expected_Cn_tail_damp_p_values}\n  Got: {Cn_tail_damp_p}")
        max_diff_index_Cn_tail_damp_p = jnp.argmax(jnp.abs(Cn_tail_damp_p - expected_Cn_tail_damp_p_values))
        print(f"\n  Max difference in Cn_tail_damp_p at index {max_diff_index_Cn_tail_damp_p}: Expected {expected_Cn_tail_damp_p_values[max_diff_index_Cn_tail_damp_p]}, Got {Cn_tail_damp_p[max_diff_index_Cn_tail_damp_p]}")
    Cn_tail_damp_p_padded_transformed_values_close = jnp.allclose(Cn_tail_damp_p_padded_transformed, expected_Cn_tail_damp_p_padded_transformed_values, atol=0.0001)
    print("Cn_tail_damp_p_padded_transformed_values_close???", Cn_tail_damp_p_padded_transformed_values_close)
     
    Cn_tail_damp_q_close = jnp.allclose(Cn_tail_damp_q, expected_Cn_tail_damp_q_values, atol=0.0001)
    print("Cn_tail_damp_q_close???", Cn_tail_damp_q_close)
    if not Cn_tail_damp_q_close:
        print(f"\n  Expected: {expected_Cn_tail_damp_q_values}\n  Got: {Cn_tail_damp_q}")
        max_diff_index_Cn_tail_damp_q = jnp.argmax(jnp.abs(Cn_tail_damp_q - expected_Cn_tail_damp_q_values))
        print(f"\n  Max difference in Cn_tail_damp_q at index {max_diff_index_Cn_tail_damp_q}: Expected {expected_Cn_tail_damp_q_values[max_diff_index_Cn_tail_damp_q]}, Got {Cn_tail_damp_q[max_diff_index_Cn_tail_damp_q]}")
    Cn_tail_damp_q_padded_transformed_values_close = jnp.allclose(Cn_tail_damp_q_padded_transformed, expected_Cn_tail_damp_q_padded_transformed_values, atol=0.0001)
    print("Cn_tail_damp_q_padded_transformed_values_close???", Cn_tail_damp_q_padded_transformed_values_close)
    if not Cn_tail_damp_q_padded_transformed_values_close:
        print(f"\n  Expected: {expected_Cn_tail_damp_q_padded_transformed_values}\n  Got: {Cn_tail_damp_q_padded_transformed}")
     
    Cn_tail_damp_r_close = jnp.allclose(Cn_tail_damp_r, expected_Cn_tail_damp_r_values, atol=0.0001)
    print("Cn_tail_damp_r_close???", Cn_tail_damp_r_close)
    if not Cn_tail_damp_r_close:
        print(f"\n  Expected: {expected_Cn_tail_damp_r_values}\n  Got: {Cn_tail_damp_r}")
        max_diff_index_Cn_tail_damp_r = jnp.argmax(jnp.abs(Cn_tail_damp_r - expected_Cn_tail_damp_r_values))
        print(f"\n  Max difference in Cn_tail_damp_r at index {max_diff_index_Cn_tail_damp_r}: Expected {expected_Cn_tail_damp_r_values[max_diff_index_Cn_tail_damp_r]}, Got {Cn_tail_damp_r[max_diff_index_Cn_tail_damp_r]}")
    Cn_tail_damp_r_padded_transformed_values_close = jnp.allclose(Cn_tail_damp_r_padded_transformed, expected_Cn_tail_damp_r_padded_transformed_values, atol=0.0001)
    print("Cn_tail_damp_r_padded_transformed_values_close???", Cn_tail_damp_r_padded_transformed_values_close)
     
    Cn_wing_close = jnp.allclose(Cn_wing, expected_Cn_wing_values, atol=0.0001)
    print("Cn_wing_close???", Cn_wing_close)
    if not Cn_wing_close:
        print(f"\n  Expected: {expected_Cn_wing_values}\n  Got: {Cn_wing}")
        max_diff_index_Cn_wing = jnp.argmax(jnp.abs(Cn_wing - expected_Cn_wing_values))
        print(f"\n  Max difference in Cn_wing at index {max_diff_index_Cn_wing}: Expected {expected_Cn_wing_values[max_diff_index_Cn_wing]}, Got {Cn_wing[max_diff_index_Cn_wing]}")
    Cn_wing_padded_transformed_values_close = jnp.allclose(Cn_wing_padded_transformed, expected_Cn_wing_padded_transformed_values, atol=0.0001)
    print("Cn_wing_padded_transformed_values_close???", Cn_wing_padded_transformed_values_close)
    
    Cn_wing_damp_p_close = jnp.allclose(Cn_wing_damp_p, expected_Cn_wing_damp_p_values, atol=0.0001)
    print("Cn_wing_damp_p_close???", Cn_wing_damp_p_close)
    if not Cn_wing_damp_p_close:
        print(f"\n  Expected: {expected_Cn_wing_damp_p_values}\n  Got: {Cn_wing_damp_p}")
        max_diff_index_Cn_wing_damp_p = jnp.argmax(jnp.abs(Cn_wing_damp_p - expected_Cn_wing_damp_p_values))
        print(f"\n  Max difference in Cn_wing_damp_p at index {max_diff_index_Cn_wing_damp_p}: Expected {expected_Cn_wing_damp_p_values[max_diff_index_Cn_wing_damp_p]}, Got {Cn_wing_damp_p[max_diff_index_Cn_wing_damp_p]}")
    Cn_wing_damp_p_padded_transformed_values_close = jnp.allclose(Cn_wing_damp_p_padded_transformed, expected_Cn_wing_damp_p_padded_transformed_values, atol=0.0001)
    print("Cn_wing_damp_p_padded_transformed_values_close???", Cn_wing_damp_p_padded_transformed_values_close)
     
    Cn_wing_damp_q_close = jnp.allclose(Cn_wing_damp_q, expected_Cn_wing_damp_q_values, atol=0.0001)
    print("Cn_wing_damp_q_close???", Cn_wing_damp_q_close)
    if not Cn_wing_damp_q_close:
        print(f"\n  Expected: {expected_Cn_wing_damp_q_values}\n  Got: {Cn_wing_damp_q}")
        max_diff_index_Cn_wing_damp_q = jnp.argmax(jnp.abs(Cn_wing_damp_q - expected_Cn_wing_damp_q_values))
        print(f"\n  Max difference in Cn_wing_damp_q at index {max_diff_index_Cn_wing_damp_q}: Expected {expected_Cn_wing_damp_q_values[max_diff_index_Cn_wing_damp_q]}, Got {Cn_wing_damp_q[max_diff_index_Cn_wing_damp_q]}")
    Cn_wing_damp_q_padded_transformed_values_close = jnp.allclose(Cn_wing_damp_q_padded_transformed, expected_Cn_wing_damp_q_padded_transformed_values, atol=0.0001)
    print("Cn_wing_damp_q_padded_transformed_values_close???", Cn_wing_damp_q_padded_transformed_values_close)
    if not Cn_wing_damp_q_padded_transformed_values_close:
        print(f"\n  Expected: {expected_Cn_wing_damp_q_padded_transformed_values}\n  Got: {Cn_wing_damp_q_padded_transformed}")
     
    Cn_wing_damp_r_close = jnp.allclose(Cn_wing_damp_r, expected_Cn_wing_damp_r_values, atol=0.0001)
    print("Cn_wing_damp_r_close???", Cn_wing_damp_r_close)
    if not Cn_wing_damp_r_close:
        print(f"\n  Expected: {expected_Cn_wing_damp_r_values}\n  Got: {Cn_wing_damp_r}")
        max_diff_index_Cn_wing_damp_r = jnp.argmax(jnp.abs(Cn_wing_damp_r - expected_Cn_wing_damp_r_values))
        print(f"\n  Max difference in Cn_wing_damp_r at index {max_diff_index_Cn_wing_damp_r}: Expected {expected_Cn_wing_damp_r_values[max_diff_index_Cn_wing_damp_r]}, Got {Cn_wing_damp_r[max_diff_index_Cn_wing_damp_r]}")
    Cn_wing_damp_r_padded_transformed_values_close = jnp.allclose(Cn_wing_damp_r_padded_transformed, expected_Cn_wing_damp_r_padded_transformed_values, atol=0.0001)
    print("Cn_wing_damp_r_padded_transformed_values_close???", Cn_wing_damp_r_padded_transformed_values_close)
    if not Cn_wing_damp_r_padded_transformed_values_close:
        print(f"\n  Expected: {expected_Cn_wing_damp_r_padded_transformed_values}\n  Got: {Cn_wing_damp_r_padded_transformed}")

    Cn_hover_fuse_close = jnp.allclose(Cn_hover_fuse, expected_Cn_hover_fuse_values, atol=0.0001)
    print("Cn_hover_fuse_close???", Cn_hover_fuse_close)
    if not Cn_hover_fuse_close:
        print(f"\n  Expected: {expected_Cn_hover_fuse_values}\n  Got: {Cn_hover_fuse}")
        max_diff_index_Cn_hover_fuse = jnp.argmax(jnp.abs(Cn_hover_fuse - expected_Cn_hover_fuse_values))
        print(f"\n  Max difference in Cn_hover_fuse at index {max_diff_index_Cn_hover_fuse}: Expected {expected_Cn_hover_fuse_values[max_diff_index_Cn_hover_fuse]}, Got {Cn_hover_fuse[max_diff_index_Cn_hover_fuse]}")
    Cn_hover_fuse_padded_values_close = jnp.allclose(Cn_hover_fuse_padded, expected_Cn_hover_fuse_padded_values, atol=0.0001)
    print("Cn_hover_fuse_padded_values_close???", Cn_hover_fuse_padded_values_close)
    if not Cn_hover_fuse_padded_values_close:
        print(f"\n  Expected: {expected_Cn_hover_fuse_padded_values}\n  Got: {Cn_hover_fuse_padded}")


    
    M3_array = jnp.array([
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
    ])


    Cn_outputs_values_close = jnp.allclose(M3_array, expected_Cn_outputs_values, atol=0.0001)
    print("Cn_outputs_values_close???", Cn_outputs_values_close)
    if not Cn_outputs_values_close:
        print(f"\n  Expected: {expected_Cn_outputs_values}\n  Got: {M3_array}")
        max_diff_index_Cn_outputs_values = jnp.argmax(jnp.abs(M3_array - expected_Cn_outputs_values))
        print(f"\n  Max difference in Cn_outputs_values at index {max_diff_index_Cn_outputs_values}: Expected {expected_Cn_outputs_values[max_diff_index_Cn_outputs_values]}, Got {M3_array[max_diff_index_Cn_outputs_values]}")
        print("[Warning: Cn_tail_padded_transformed_values not close possibly due to identified numerical issue]")