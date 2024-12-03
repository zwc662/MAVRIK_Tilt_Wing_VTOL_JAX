
import pytest

import os
import sys

from jax_mavrik.src.mavrik_aero import MavrikAero

from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.mavrik_types import StateVariables, ControlInputs, Forces
from jax_mavrik.src.actuator import ActuatorOutput 

from jax_mavrik.src.utils.mat_tools import euler_to_dcm

import jax.numpy as jnp

from .test_mavrik_aero import expected_actuator_outputs_values as actuator_outputs_values, expected_Cl_outputs_values as expected_Cl_outputs_values


expected_Cl_alieron_wing_values = jnp.zeros([11])
expected_Cl_elevator_tail_values = jnp.zeros([11])
expected_Cl_flap_wing_values = jnp.zeros([11])
expected_Cl_ruder_tail_values = jnp.zeros([11])
expected_Cl_tail_values = jnp.array([
    5.26199454379631e-18, -8.21513429541714e-06, -3.28254290896977e-05, -7.38116986401611e-05,
    -0.000131207465562497, -0.000205098069114438, -0.000295620933087363, -0.000402967204698023,
    -0.000527384841891959, -0.000669183107302679, -0.000828756006321496
])
expected_Cl_tail_damp_p_values = jnp.array([
    0.00377864750296384, 0.00338298860881428, 0.00286692196883962, 0.00224505869723053,
    0.00153226927495496, 0.000743483460652908, -0.000106496458521804, -0.0010031677674813,
    -0.00193248364597144, -0.00288099279983048, -0.00383601846487451
])
expected_Cl_tail_damp_q_values = jnp.array([
    -8.99466635432648e-12, 7.10501998695196e-06, 2.62450560351976e-05, 5.31986590039317e-05,
    8.27026687299806e-05, 0.000108732325601059, 0.000124771758656816, 0.000124070656105436,
    9.98829598325593e-05, 4.56837144220599e-05, -4.46453638860413e-05
])
expected_Cl_tail_damp_r_values = jnp.array([
    0.221587621737447, 0.220947184436212, 0.220111685461931, 0.219104788867008,
    0.217950578878292, 0.216673236270277, 0.215296736258328, 0.213844570018616,
    0.212339491657081, 0.210803292107028, 0.209258349319486
])
expected_Cl_wing_values = jnp.array([
    -0.00415458508496627, -0.00419454260947585, -0.00423103006400414, -0.00426406266231232,
    -0.00429373508770546, -0.00432021738503664, -0.00434374771617764, -0.00436462267457084,
    -0.00438318587177731, -0.00439981549275778, -0.00444465999616078
])
expected_Cl_wing_damp_p_values = jnp.array([
    -0.036248706145892, -0.0334099851185107, -0.0297089222659419, -0.0252503214865677,
    -0.0201407958535564, -0.0144873417815261, -0.00839600998758306, -0.00197068177138848,
    0.00468804206402661, 0.0114838337310152, 0.0183256154801606
])
expected_Cl_wing_damp_q_values = jnp.array([
    0.00178284009887367, 0.00143739914123512, 0.0003660313515795, -0.00142894191060853,
    -0.00394712052339388, -0.00719008625917033, -0.0111615118986804, -0.0158673265158974,
    -0.021315938752837, -0.0275185147456274, -0.0344768633340457
])
expected_Cl_wing_damp_r_values = jnp.array([
    0.207015414447948, 0.201020143343644, 0.19318508448679, 0.183733060023768,
    0.172890918662929, 0.160886485111478, 0.147945709288944, 0.134290035772302,
    0.12013401126973, 0.105683144680789, 0.0911517794208286
])
expected_Cl_hover_fuse_values = jnp.array([
    -4.2351647362715e-22, -2.81355060683179e-10, -1.13508198350771e-09, -2.58419078289118e-09,
    -4.66154255377487e-09, -7.40766281627927e-09, -1.08686955656673e-08, -1.50945260782677e-08,
    -2.01370958755826e-08, -2.60489290145274e-08, -3.28827159284515e-08
])

expected_Cl_Scale_values = jnp.array([
    895.135626, 892.568381914787, 890.06735229599, 887.646445443064,
    885.315760821133, 883.08159463424, 880.946619222977, 878.910202534566,
    876.968833249431, 875.116617846926, 873.347393946174
])

expected_Cl_Scale_p_values = jnp.array([
    0, -0.323592856567991, -0.648519796081949, -0.975802438624017,
    -1.30641852795902, -1.64132155179696, -1.98146156880648, -2.32780642736341,
    -2.68136266744991, -3.04319550029641, -3.41553345280916
])
 
expected_Cl_Scale_q_values = jnp.array([
    0, -0.46918142287958, -0.898817894550568, -1.28662251873764,
    -1.63101021251671, -1.93105095416479, -2.18642066587276, -2.39735107287998,
    -2.56457944240154, -2.68929874079305, -2.77318467623425
])

expected_Cl_Scale_r_values = jnp.array([
    0.0, 0.00777952417736195, 0.0172522108646338, 0.0296311746000723, 0.046094837559518,
    0.0677907947019389, 0.095840171969681, 0.131342735061023, 0.175382899114121,
    0.229036688931439, 0.293408419739419
])
 
expected_wind_transform = jnp.repeat(jnp.diag(jnp.array([1., 1., 1.]))[None, :, :], 11, axis=0)
expected_tail_transform = jnp.repeat(jnp.diag(jnp.array([1., 1., 1.]))[None, :, :], 11, axis=0) 

expected_Cl_aileron_wing_padded_transformed_values = jnp.zeros([11, 3])
expected_Cl_elevator_tail_padded_transformed_values = jnp.zeros([11, 3])
expected_Cl_flap_wing_padded_transformed_values = jnp.zeros([11, 3])
expected_Cl_ruder_tail_padded_transformed_values = jnp.zeros([11, 3])
expected_Cl_tail_padded_transformed_values = jnp.array([
    [4.7101987799697e-15, 0, 0], [-0.00733256912527315, 0, 0], [-0.029216842757847, 0, 0], [-0.0655186919300536, 0, 0],
    [-0.116160037199875, 0, 0], [-0.181118329929982, 0, 0], [-0.260426261574854, 0, 0], [-0.354171987495928, 0, 0],
    [-0.462500069467427, 0, 0], [-0.585613257583017, 0, 0], [-0.723791898338118, 0, 0]
])
expected_Cl_tail_damp_p_padded_transformed_values = jnp.array([
    [0., 0., 0.], [-0.00109471094766319, 0, 0], [-0.00185925565061473, 0, 0], [-0.00219073375161161, 0, 0], 
    [-0.0020017849706235, 0, 0], [-0.0012202954273742, 0, 0], [0.000211018639774947, 0, 0], [0.00233518037686678, 0, 0], 
    [0.00518168950376532, 0, 0], [0.00876742432483045, 0, 0], [0.0131020493923725, 0, 0]
])
expected_Cl_tail_damp_q_padded_transformed_values = jnp.array([
    [0., 0., 0.], [-3.33354338706597e-06, 0, 0], [-2.3589526007918e-05, 0, 0], [-6.84465926411035e-05, 0, 0], 
    [-0.000134888897300984, 0, 0], [-0.000209967661100482, 0, 0], [-0.00027280355164455, 0, 0], [-0.000297440920527291, 0, 0], 
    [-0.0002561577854328, 0, 0], [-0.000122857155669995, 0, 0], [0.000123809838993672, 0, 0]
])
expected_Cl_tail_damp_r_padded_transformed_values = jnp.array([
    [0., 0., 0.], [0.00171886396324156, 0., 0.], [0.00379741321135919, 0., 0.], [0.00649233225463027, 0., 0.],
    [0.0100463965293978, 0., 0.], [0.014688450877403, 0., 0.], [0.0206340762275092, 0., 0.], [0.0280869307041934, 0., 0.],
    [0.0372407156432375, 0., 0.], [0.0482816880400407, 0., 0.], [0.0613981615911097, 0., 0.]
])
expected_Cl_wing_padded_transformed_values = jnp.array([
    [-3.71891712080154, 0, 0], [-3.74391610981249, 0, 0], [-3.7659017265529, 0, 0], [-3.78498006534802, 0, 0],
    [-3.80131134593636, 0, 0], [-3.81510445754472, 0, 0], [-3.82660986532422, 0, 0], [-3.83611139889402, 0, 0],
    [-3.84391739988794, 0, 0], [-3.8503516531727, 0, 0], [-3.88173222462383, 0, 0]
])
expected_Cl_wing_damp_p_padded_transformed_values = jnp.array([
    [0., 0., 0.], [0.0108112325223929, 0., 0.], [0.0192668242097231, 0., 0.], [0.0246393252826331, 0., 0.],
    [0.0263123088709264, 0., 0.], [0.0237783862942674, 0., 0.], [0.0166363711217112, 0., 0.], [0.00458736569372602, 0., 0.],
    [-0.0125703409739158, 0., 0.], [-0.0349475511363777, 0., 0.], [-0.0625917527158059, 0., 0.]
]) 
expected_Cl_wing_damp_q_padded_transformed_values = jnp.array([
    [0., 0., 0.], [-0.000674400974330578, 0., 0.], [-0.000328995528766185, 0., 0.], [0.00183850884015693, 0., 0.],
    [0.00643779388368971, 0., 0.], [0.013884422931298, 0., 0.], [0.0244037602776597, 0., 0.], [0.0380395522466237, 0., 0.],
    [0.0546664183210161, 0., 0.], [0.0740055070539108, 0., 0.], [0.095610709082598, 0., 0.]
]) 
expected_Cl_wing_damp_r_padded_transformed_values = jnp.array([
    [0., 0., 0.], [0.00156384106527864, 0., 0.], [0.0033328698134682, 0., 0.], [0.00544422638136982, 0., 0.],
    [0.00796937881128356, 0., 0.], [0.0109066226825088, 0., 0.], [0.0141791422204288, 0., 0.], [0.0176380205897767, 0., 0.],
    [0.0210694511786937, 0., 0.], [0.0242053175335502, 0., 0.], [0.0267446995563014, 0., 0.]
]) 

@pytest.fixture
def mavrik_aero():
    mavrik_setup = MavrikSetup(file_path=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "jax_mavrik/aero_export.mat")
    )
    return MavrikAero(mavrik_setup=mavrik_setup)


@pytest.mark.parametrize(
    "id, actuator_outputs_values, \
        expected_Cl_outputs_values, expected_Cl_alieron_wing_values, expected_Cl_elevator_tail_values, expected_Cl_flap_wing_values, expected_Cl_ruder_tail_values, expected_Cl_tail_values, expected_Cl_tail_damp_p_values, expected_Cl_tail_damp_q_values, expected_Cl_tail_damp_r_values, expected_Cl_wing_values, expected_Cl_wing_damp_p_values, expected_Cl_wing_damp_q_values, expected_Cl_wing_damp_r_values, expected_Cl_hover_fuse_values, \
            expected_Cl_Scale_values, expected_Cl_Scale_p_values, expected_Cl_Scale_q_values, expected_Cl_Scale_r_values, \
                expected_wind_transform, expected_tail_transform, \
                expected_Cl_tail_padded_transformed_values, expected_Cl_tail_damp_p_padded_transformed_values, expected_Cl_tail_damp_q_padded_transformed_values, expected_Cl_tail_damp_r_padded_transformed_values, expected_Cl_wing_padded_transformed_values, expected_Cl_wing_damp_p_padded_transformed_values, expected_Cl_wing_damp_q_padded_transformed_values, expected_Cl_wing_damp_r_padded_transformed_values",
    zip(
        list(range(11)), actuator_outputs_values, \
            expected_Cl_outputs_values, expected_Cl_alieron_wing_values, expected_Cl_elevator_tail_values, expected_Cl_flap_wing_values, expected_Cl_ruder_tail_values, expected_Cl_tail_values, expected_Cl_tail_damp_p_values, expected_Cl_tail_damp_q_values, expected_Cl_tail_damp_r_values, expected_Cl_wing_values, expected_Cl_wing_damp_p_values, expected_Cl_wing_damp_q_values, expected_Cl_wing_damp_r_values, expected_Cl_hover_fuse_values, \
            expected_Cl_Scale_values, expected_Cl_Scale_p_values, expected_Cl_Scale_q_values, expected_Cl_Scale_r_values,
            expected_wind_transform, expected_tail_transform, \
            expected_Cl_tail_padded_transformed_values, expected_Cl_tail_damp_p_padded_transformed_values, expected_Cl_tail_damp_q_padded_transformed_values, expected_Cl_tail_damp_r_padded_transformed_values, expected_Cl_wing_padded_transformed_values, expected_Cl_wing_damp_p_padded_transformed_values, expected_Cl_wing_damp_q_padded_transformed_values, expected_Cl_wing_damp_r_padded_transformed_values
    )
)
 
def test_mavrik_aero(id, mavrik_aero, actuator_outputs_values, \
                     expected_Cl_outputs_values, expected_Cl_alieron_wing_values, expected_Cl_elevator_tail_values, expected_Cl_flap_wing_values, expected_Cl_ruder_tail_values, expected_Cl_tail_values, expected_Cl_tail_damp_p_values, expected_Cl_tail_damp_q_values, expected_Cl_tail_damp_r_values, expected_Cl_wing_values, expected_Cl_wing_damp_p_values, expected_Cl_wing_damp_q_values, expected_Cl_wing_damp_r_values, expected_Cl_hover_fuse_values, \
                     expected_Cl_Scale_values, expected_Cl_Scale_p_values, expected_Cl_Scale_q_values, expected_Cl_Scale_r_values,
                     expected_wind_transform, expected_tail_transform, \
                        expected_Cl_tail_padded_transformed_values, expected_Cl_tail_damp_p_padded_transformed_values, expected_Cl_tail_damp_q_padded_transformed_values, expected_Cl_tail_damp_r_padded_transformed_values, expected_Cl_wing_padded_transformed_values, expected_Cl_wing_damp_p_padded_transformed_values, expected_Cl_wing_damp_q_padded_transformed_values, expected_Cl_wing_damp_r_padded_transformed_values):
    u = ActuatorOutput(*actuator_outputs_values)
        
    print(f">>>>>>>>>>>>>>>>>>>> Test ID: {id} <<<<<<<<<<<<<<<<<<<<<<")
    
    wing_transform = jnp.array([[jnp.cos(u.wing_tilt), 0, jnp.sin(u.wing_tilt)], [0, 1, 0], [-jnp.sin(u.wing_tilt), 0., jnp.cos(u.wing_tilt)]])
    tail_transform = jnp.array([[jnp.cos(u.tail_tilt), 0, jnp.sin(u.tail_tilt)], [0, 1, 0], [-jnp.sin(u.tail_tilt), 0., jnp.cos(u.tail_tilt)]])

    M1 = mavrik_aero.L(u, wing_transform, tail_transform)
    M1_array = jnp.array([M1.L, M1.M, M1.N])
    Cl_outputs_values_close = jnp.allclose(M1_array, expected_Cl_outputs_values, atol=0.0001)
    print("Cl_outputs_values_close???", Cl_outputs_values_close)
    if not Cl_outputs_values_close:
        print(f"\n  Expected: {expected_Cl_outputs_values}\n  Got: {M1_array}")
        max_diff_index_Cl_outputs_values = jnp.argmax(jnp.abs(M1_array - expected_Cl_outputs_values))
        print(f"\n  Max difference in Cl_outputs_values at index {max_diff_index_Cl_outputs_values}: Expected {expected_Cl_outputs_values[max_diff_index_Cl_outputs_values]}, Got {M1_array[max_diff_index_Cl_outputs_values]}")
 

    Cl_Scale = 0.5744 * 2.8270 * u.Q
    Cl_Scale_p = 0.5744 * 2.8270**2 * 1.225 * 0.25 * u.U * u.p
    Cl_Scale_q = 0.5744 * 2.8270 * 0.2032 * 1.225 * 0.25 * u.U * u.q
    Cl_Scale_r = 0.5744 * 2.8270**2 * 1.225 * 0.25 * u.U * u.r

    
    print("Cl_Scale_close???", jnp.allclose(Cl_Scale, expected_Cl_Scale_values, atol=0.0001))
    if not jnp.allclose(Cl_Scale, expected_Cl_Scale_values, atol=0.0001):
        print(f"\n  Expected: {expected_Cl_Scale_values}\n  Got: {Cl_Scale}")
    print("Cl_Scale_p_close???", jnp.allclose(Cl_Scale_p, expected_Cl_Scale_p_values, atol=0.0001))
    if not jnp.allclose(Cl_Scale_p, expected_Cl_Scale_p_values, atol=0.0001):
        print(f"\n  Expected: {expected_Cl_Scale_p_values}\n  Got: {Cl_Scale_p}") 
    print("Cl_Scale_q_close???", jnp.allclose(Cl_Scale_q, expected_Cl_Scale_q_values, atol=0.0001))
    if not jnp.allclose(Cl_Scale_q, expected_Cl_Scale_q_values, atol=0.0001):
        print(f"\n  Expected: {expected_Cl_Scale_q_values}\n  Got: {Cl_Scale_q}")
    print("Cl_Scale_r_close???", jnp.allclose(Cl_Scale_r, expected_Cl_Scale_r_values, atol=0.0001))
    if not jnp.allclose(Cl_Scale_r, expected_Cl_Scale_r_values, atol=0.0001):
        print(f"\n  Expected: {expected_Cl_Scale_r_values}\n  Got: {Cl_Scale_r}") 
   
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
   
        
    Cl_aileron_wing = mavrik_aero.Cl_aileron_wing_lookup_table(jnp.array([
        u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron
    ]))
    Cl_aileron_wing_padded = jnp.array([Cl_aileron_wing, 0.0, 0.0])
    Cl_aileron_wing_padded_transformed = jnp.dot(wing_transform, Cl_aileron_wing_padded * Cl_Scale)

    
    Cl_elevator_tail = mavrik_aero.Cl_elevator_tail_lookup_table(jnp.array([
        u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator
    ]))
    Cl_elevator_tail_padded = jnp.array([Cl_elevator_tail, 0.0, 0.0])
    Cl_elevator_tail_padded_transformed = jnp.dot(tail_transform, Cl_elevator_tail_padded * Cl_Scale)

    Cl_flap_wing = mavrik_aero.Cl_flap_wing_lookup_table(jnp.array([
        u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap
    ]))
    Cl_flap_wing_padded = jnp.array([Cl_flap_wing, 0.0, 0.0])
    Cl_flap_wing_padded_transformed = jnp.dot(wing_transform, Cl_flap_wing_padded * Cl_Scale)

    Cl_rudder_tail = mavrik_aero.Cl_rudder_tail_lookup_table(jnp.array([
        u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder
    ]))
    Cl_rudder_tail_padded = jnp.array([Cl_rudder_tail, 0.0, 0.0])
    Cl_rudder_tail_padded_transformed = jnp.dot(tail_transform, Cl_rudder_tail_padded * Cl_Scale)

    # Tail
    Cl_tail = mavrik_aero.Cl_tail_lookup_table(jnp.array([
        u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
    ]))
    Cl_tail_padded = jnp.array([Cl_tail, 0.0, 0.0])
    Cl_tail_padded_transformed = jnp.dot(tail_transform, Cl_tail_padded * Cl_Scale)

    # Tail Damp p
    Cl_tail_damp_p = mavrik_aero.Cl_tail_damp_p_lookup_table(jnp.array([
        u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
    ]))
    Cl_tail_damp_p_padded = jnp.array([Cl_tail_damp_p, 0.0, 0.0])
    Cl_tail_damp_p_padded_transformed = jnp.dot(tail_transform, Cl_tail_damp_p_padded * Cl_Scale_p)

    # Tail Damp q
    Cl_tail_damp_q = mavrik_aero.Cl_tail_damp_q_lookup_table(jnp.array([
        u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
    ]))
    Cl_tail_damp_q_padded = jnp.array([Cl_tail_damp_q, 0.0, 0.0])
    Cl_tail_damp_q_padded_transformed = jnp.dot(tail_transform, Cl_tail_damp_q_padded * Cl_Scale_q)

    # Tail Damp r
    Cl_tail_damp_r = mavrik_aero.Cl_tail_damp_r_lookup_table(jnp.array([
        u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta
    ]))
    Cl_tail_damp_r_padded = jnp.array([Cl_tail_damp_r, 0.0, 0.0])
    Cl_tail_damp_r_padded_transformed = jnp.dot(tail_transform, Cl_tail_damp_r_padded * Cl_Scale_r)

    # Wing
    Cl_wing = mavrik_aero.Cl_wing_lookup_table(jnp.array([
        u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
    ]))
    Cl_wing_padded = jnp.array([Cl_wing, 0.0, 0.0])
    Cl_wing_padded_transformed = jnp.dot(wing_transform, Cl_wing_padded * Cl_Scale)

    # Wing Damp p
    Cl_wing_damp_p = mavrik_aero.Cl_wing_damp_p_lookup_table(jnp.array([
        u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
    ]))
    Cl_wing_damp_p_padded = jnp.array([Cl_wing_damp_p, 0.0, 0.0])
    Cl_wing_damp_p_padded_transformed = jnp.dot(wing_transform, Cl_wing_damp_p_padded * Cl_Scale_p)

    # Wing Damp q
    Cl_wing_damp_q = mavrik_aero.Cl_wing_damp_q_lookup_table(jnp.array([
        u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
    ]))
    Cl_wing_damp_q_padded = jnp.array([Cl_wing_damp_q, 0.0, 0.0])
    Cl_wing_damp_q_padded_transformed = jnp.dot(wing_transform, Cl_wing_damp_q_padded * Cl_Scale_q)

    # Wing Damp r
    Cl_wing_damp_r = mavrik_aero.Cl_wing_damp_r_lookup_table(jnp.array([
        u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta
    ]))
    Cl_wing_damp_r_padded = jnp.array([Cl_wing_damp_r, 0.0, 0.0])
    Cl_wing_damp_r_padded_transformed = jnp.dot(wing_transform, Cl_wing_damp_r_padded * Cl_Scale_r)

    # Hover Fuse
    Cl_hover_fuse = mavrik_aero.Cl_hover_fuse_lookup_table(jnp.array([
        u.U, u.alpha, u.beta
    ]))
    Cl_hover_fuse_padded = jnp.array([Cl_hover_fuse * Cl_Scale, 0.0, 0.0])



    Cl_aileron_wing_close = jnp.allclose(Cl_aileron_wing, expected_Cl_alieron_wing_values, atol=0.0001)
    print("Cl_aileron_wing_close???", Cl_aileron_wing_close)
    if not Cl_aileron_wing_close:
        print(f"\n  Expected: {expected_Cl_alieron_wing_values}\n  Got: {Cl_aileron_wing}")
        max_diff_index_Cl_aileron_wing = jnp.argmax(jnp.abs(Cl_aileron_wing - expected_Cl_alieron_wing_values))
        print(f"\n  Max difference in Cl_aileron_wing at index {max_diff_index_Cl_aileron_wing}: Expected {expected_Cl_alieron_wing_values[max_diff_index_Cl_aileron_wing]}, Got {Cl_aileron_wing[max_diff_index_Cl_aileron_wing]}")
    #Cl_aileron_wing_padded_transformed_values_close = jnp.allclose(Cl_aileron_wing_padded_transformed, expected_Cl_aileron_wing_padded_transformed_values, atol=0.0001)
    #print("Cl_aileron_wing_padded_transformed_values_close???", Cl_aileron_wing_padded_transformed_values_close)
    
    Cl_elevator_tail_close = jnp.allclose(Cl_elevator_tail, expected_Cl_elevator_tail_values, atol=0.0001)
    print("Cl_elevator_tail_close???", Cl_elevator_tail_close)
    if not Cl_elevator_tail_close:
        print(f"\n  Expected: {expected_Cl_elevator_tail_values}\n  Got: {Cl_elevator_tail}")
        max_diff_index_Cl_elevator_tail = jnp.argmax(jnp.abs(Cl_elevator_tail - expected_Cl_elevator_tail_values))
        print(f"\n  Max difference in Cl_elevator_tail at index {max_diff_index_Cl_elevator_tail}: Expected {expected_Cl_elevator_tail_values[max_diff_index_Cl_elevator_tail]}, Got {Cl_elevator_tail[max_diff_index_Cl_elevator_tail]}")
    #Cl_elevator_tail_padded_transformed_values_close = jnp.allclose(Cl_elevator_tail_padded_transformed, expected_Cl_elevator_tail_padded_transformed_values, atol=0.0001)
    #print("Cl_elevator_tail_padded_transformed_values_close???", Cl_elevator_tail_padded_transformed_values_close)
     
    Cl_flap_wing_close = jnp.allclose(Cl_flap_wing, expected_Cl_flap_wing_values, atol=0.0001)
    print("Cl_flap_wing_close???", Cl_flap_wing_close)
    if not Cl_flap_wing_close:
        print(f"\n  Expected: {expected_Cl_flap_wing_values}\n  Got: {Cl_flap_wing}")
        max_diff_index_Cl_flap_wing = jnp.argmax(jnp.abs(Cl_flap_wing - expected_Cl_flap_wing_values))
        print(f"\n  Max difference in Cl_flap_wing at index {max_diff_index_Cl_flap_wing}: Expected {expected_Cl_flap_wing_values[max_diff_index_Cl_flap_wing]}, Got {Cl_flap_wing[max_diff_index_Cl_flap_wing]}")
    #Cl_flap_wing_padded_transformed_values_close = jnp.allclose(Cl_flap_wing_padded_transformed, expected_Cl_flap_wing_padded_transformed_values, atol=0.0001)  
    #print("Cl_flap_wing_padded_transformed_values_close???", Cl_flap_wing_padded_transformed_values_close)
    
    
    Cl_rudder_tail_close = jnp.allclose(Cl_rudder_tail, expected_Cl_ruder_tail_values, atol=0.0001)
    print("Cl_rudder_tail_close???", Cl_rudder_tail_close)
    if not Cl_rudder_tail_close:
        print(f"\n  Expected: {expected_Cl_ruder_tail_values}\n  Got: {Cl_rudder_tail}")
        max_diff_index_Cl_rudder_tail = jnp.argmax(jnp.abs(Cl_rudder_tail - expected_Cl_ruder_tail_values))
        print(f"\n  Max difference in Cl_rudder_tail at index {max_diff_index_Cl_rudder_tail}: Expected {expected_Cl_ruder_tail_values[max_diff_index_Cl_rudder_tail]}, Got {Cl_rudder_tail[max_diff_index_Cl_rudder_tail]}")
    #Cl_rudder_tail_padded_transformed_values_close = jnp.allclose(Cl_rudder_tail_padded_transformed, expected_Cl_ruder_tail_padded_transformed_values, atol=0.0001)
    #print("Cl_rudder_tail_padded_transformed_values_close???", Cl_rudder_tail_padded_transformed_values_close)
     
    Cl_tail_close = jnp.allclose(Cl_tail, expected_Cl_tail_values, atol=0.0001)
    print("Cl_tail_close???", Cl_tail_close)
    if not Cl_tail_close:
        print(f"\n  Expected: {expected_Cl_tail_values}\n  Got: {Cl_tail}")
        max_diff_index_Cl_tail = jnp.argmax(jnp.abs(Cl_tail - expected_Cl_tail_values))
        print(f"\n  Max difference in Cl_tail at index {max_diff_index_Cl_tail}: Expected {expected_Cl_tail_values[max_diff_index_Cl_tail]}, Got {Cl_tail[max_diff_index_Cl_tail]}")
    Cl_tail_padded_transformed_values_close = jnp.allclose(Cl_tail_padded_transformed, expected_Cl_tail_padded_transformed_values, atol=0.0001)
    print("Cl_tail_padded_transformed_values_close???", Cl_tail_padded_transformed_values_close)
    if not Cl_tail_padded_transformed_values_close:
        print(f"\n  Expected: {expected_Cl_tail_padded_transformed_values}\n  Got: {Cl_tail_padded_transformed}")


    Cl_tail_damp_p_close = jnp.allclose(Cl_tail_damp_p, expected_Cl_tail_damp_p_values, atol=0.0001)
    print("Cl_tail_damp_p_close???", Cl_tail_damp_p_close)
    if not Cl_tail_damp_p_close:
        print(f"\n  Expected: {expected_Cl_tail_damp_p_values}\n  Got: {Cl_tail_damp_p}")
        max_diff_index_Cl_tail_damp_p = jnp.argmax(jnp.abs(Cl_tail_damp_p - expected_Cl_tail_damp_p_values))
        print(f"\n  Max difference in Cl_tail_damp_p at index {max_diff_index_Cl_tail_damp_p}: Expected {expected_Cl_tail_damp_p_values[max_diff_index_Cl_tail_damp_p]}, Got {Cl_tail_damp_p[max_diff_index_Cl_tail_damp_p]}")
    Cl_tail_damp_p_padded_transformed_values_close = jnp.allclose(Cl_tail_damp_p_padded_transformed, expected_Cl_tail_damp_p_padded_transformed_values, atol=0.0001)
    print("Cl_tail_damp_p_padded_transformed_values_close???", Cl_tail_damp_p_padded_transformed_values_close)
     
    Cl_tail_damp_q_close = jnp.allclose(Cl_tail_damp_q, expected_Cl_tail_damp_q_values, atol=0.0001)
    print("Cl_tail_damp_q_close???", Cl_tail_damp_q_close)
    if not Cl_tail_damp_q_close:
        print(f"\n  Expected: {expected_Cl_tail_damp_q_values}\n  Got: {Cl_tail_damp_q}")
        max_diff_index_Cl_tail_damp_q = jnp.argmax(jnp.abs(Cl_tail_damp_q - expected_Cl_tail_damp_q_values))
        print(f"\n  Max difference in Cl_tail_damp_q at index {max_diff_index_Cl_tail_damp_q}: Expected {expected_Cl_tail_damp_q_values[max_diff_index_Cl_tail_damp_q]}, Got {Cl_tail_damp_q[max_diff_index_Cl_tail_damp_q]}")
    Cl_tail_damp_q_padded_transformed_values_close = jnp.allclose(Cl_tail_damp_q_padded_transformed, expected_Cl_tail_damp_q_padded_transformed_values, atol=0.0001)
    print("Cl_tail_damp_q_padded_transformed_values_close???", Cl_tail_damp_q_padded_transformed_values_close)
    if not Cl_tail_damp_q_padded_transformed_values_close:
        print(f"\n  Expected: {expected_Cl_tail_damp_q_padded_transformed_values}\n  Got: {Cl_tail_damp_q_padded_transformed}")
     
    Cl_tail_damp_r_close = jnp.allclose(Cl_tail_damp_r, expected_Cl_tail_damp_r_values, atol=0.0001)
    print("Cl_tail_damp_r_close???", Cl_tail_damp_r_close)
    if not Cl_tail_damp_r_close:
        print(f"\n  Expected: {expected_Cl_tail_damp_r_values}\n  Got: {Cl_tail_damp_r}")
        max_diff_index_Cl_tail_damp_r = jnp.argmax(jnp.abs(Cl_tail_damp_r - expected_Cl_tail_damp_r_values))
        print(f"\n  Max difference in Cl_tail_damp_r at index {max_diff_index_Cl_tail_damp_r}: Expected {expected_Cl_tail_damp_r_values[max_diff_index_Cl_tail_damp_r]}, Got {Cl_tail_damp_r[max_diff_index_Cl_tail_damp_r]}")
    Cl_tail_damp_r_padded_transformed_values_close = jnp.allclose(Cl_tail_damp_r_padded_transformed, expected_Cl_tail_damp_r_padded_transformed_values, atol=0.0001)
    print("Cl_tail_damp_r_padded_transformed_values_close???", Cl_tail_damp_r_padded_transformed_values_close)
     
    Cl_wing_close = jnp.allclose(Cl_wing, expected_Cl_wing_values, atol=0.0001)
    print("Cl_wing_close???", Cl_wing_close)
    if not Cl_wing_close:
        print(f"\n  Expected: {expected_Cl_wing_values}\n  Got: {Cl_wing}")
        max_diff_index_Cl_wing = jnp.argmax(jnp.abs(Cl_wing - expected_Cl_wing_values))
        print(f"\n  Max difference in Cl_wing at index {max_diff_index_Cl_wing}: Expected {expected_Cl_wing_values[max_diff_index_Cl_wing]}, Got {Cl_wing[max_diff_index_Cl_wing]}")
    Cl_wing_padded_transformed_values_close = jnp.allclose(Cl_wing_padded_transformed, expected_Cl_wing_padded_transformed_values, atol=0.0001)
    print("Cl_wing_padded_transformed_values_close???", Cl_wing_padded_transformed_values_close)
    
    Cl_wing_damp_p_close = jnp.allclose(Cl_wing_damp_p, expected_Cl_wing_damp_p_values, atol=0.0001)
    print("Cl_wing_damp_p_close???", Cl_wing_damp_p_close)
    if not Cl_wing_damp_p_close:
        print(f"\n  Expected: {expected_Cl_wing_damp_p_values}\n  Got: {Cl_wing_damp_p}")
        max_diff_index_Cl_wing_damp_p = jnp.argmax(jnp.abs(Cl_wing_damp_p - expected_Cl_wing_damp_p_values))
        print(f"\n  Max difference in Cl_wing_damp_p at index {max_diff_index_Cl_wing_damp_p}: Expected {expected_Cl_wing_damp_p_values[max_diff_index_Cl_wing_damp_p]}, Got {Cl_wing_damp_p[max_diff_index_Cl_wing_damp_p]}")
    Cl_wing_damp_p_padded_transformed_values_close = jnp.allclose(Cl_wing_damp_p_padded_transformed, expected_Cl_wing_damp_p_padded_transformed_values, atol=0.0001)
    print("Cl_wing_damp_p_padded_transformed_values_close???", Cl_wing_damp_p_padded_transformed_values_close)
     
    Cl_wing_damp_q_close = jnp.allclose(Cl_wing_damp_q, expected_Cl_wing_damp_q_values, atol=0.0001)
    print("Cl_wing_damp_q_close???", Cl_wing_damp_q_close)
    if not Cl_wing_damp_q_close:
        print(f"\n  Expected: {expected_Cl_wing_damp_q_values}\n  Got: {Cl_wing_damp_q}")
        max_diff_index_Cl_wing_damp_q = jnp.argmax(jnp.abs(Cl_wing_damp_q - expected_Cl_wing_damp_q_values))
        print(f"\n  Max difference in Cl_wing_damp_q at index {max_diff_index_Cl_wing_damp_q}: Expected {expected_Cl_wing_damp_q_values[max_diff_index_Cl_wing_damp_q]}, Got {Cl_wing_damp_q[max_diff_index_Cl_wing_damp_q]}")
    Cl_wing_damp_q_padded_transformed_values_close = jnp.allclose(Cl_wing_damp_q_padded_transformed, expected_Cl_wing_damp_q_padded_transformed_values, atol=0.0001)
    print("Cl_wing_damp_q_padded_transformed_values_close???", Cl_wing_damp_q_padded_transformed_values_close)
    if not Cl_wing_damp_q_padded_transformed_values_close:
        print(f"\n  Expected: {expected_Cl_wing_damp_q_padded_transformed_values}\n  Got: {Cl_wing_damp_q_padded_transformed}")
     
    Cl_wing_damp_r_close = jnp.allclose(Cl_wing_damp_r, expected_Cl_wing_damp_r_values, atol=0.0001)
    print("Cl_wing_damp_r_close???", Cl_wing_damp_r_close)
    if not Cl_wing_damp_r_close:
        print(f"\n  Expected: {expected_Cl_wing_damp_r_values}\n  Got: {Cl_wing_damp_r}")
        max_diff_index_Cl_wing_damp_r = jnp.argmax(jnp.abs(Cl_wing_damp_r - expected_Cl_wing_damp_r_values))
        print(f"\n  Max difference in Cl_wing_damp_r at index {max_diff_index_Cl_wing_damp_r}: Expected {expected_Cl_wing_damp_r_values[max_diff_index_Cl_wing_damp_r]}, Got {Cl_wing_damp_r[max_diff_index_Cl_wing_damp_r]}")
    Cl_wing_damp_r_padded_transformed_values_close = jnp.allclose(Cl_wing_damp_r_padded_transformed, expected_Cl_wing_damp_r_padded_transformed_values, atol=0.0001)
    print("Cl_wing_damp_r_padded_transformed_values_close???", Cl_wing_damp_r_padded_transformed_values_close)
    if not Cl_wing_damp_r_padded_transformed_values_close:
        print(f"\n  Expected: {expected_Cl_wing_damp_r_padded_transformed_values}\n  Got: {Cl_wing_damp_r_padded_transformed}")

    Cl_hover_fuse_close = jnp.allclose(Cl_hover_fuse, expected_Cl_hover_fuse_values, atol=0.0001)
    print("Cl_hover_fuse_close???", Cl_hover_fuse_close)
    if not Cl_hover_fuse_close:
        print(f"\n  Expected: {expected_Cl_hover_fuse_values}\n  Got: {Cl_hover_fuse}")
        max_diff_index_Cl_hover_fuse = jnp.argmax(jnp.abs(Cl_hover_fuse - expected_Cl_hover_fuse_values))
        print(f"\n  Max difference in Cl_hover_fuse at index {max_diff_index_Cl_hover_fuse}: Expected {expected_Cl_hover_fuse_values[max_diff_index_Cl_hover_fuse]}, Got {Cl_hover_fuse[max_diff_index_Cl_hover_fuse]}")


    
    M1_array = jnp.array([
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
    ])

    Cl_outputs_values_close = jnp.allclose(M1_array, expected_Cl_outputs_values, atol=0.0001)
    print("Cl_outputs_values_close???", Cl_outputs_values_close)
    if not Cl_outputs_values_close:
        print(f"\n  Expected: {expected_Cl_outputs_values}\n  Got: {M1_array}")
        max_diff_index_Cl_outputs_values = jnp.argmax(jnp.abs(M1_array - expected_Cl_outputs_values))
        print(f"\n  Max difference in Cl_outputs_values at index {max_diff_index_Cl_outputs_values}: Expected {expected_Cl_outputs_values[max_diff_index_Cl_outputs_values]}, Got {M1_array[max_diff_index_Cl_outputs_values]}")
 