
import pytest

import os
import sys

from jax_mavrik.src.mavrik_aero import MavrikAero

from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.mavrik_types import StateVariables, ControlInputs
 
from .test_mavrik import( 
    vned_values, xned_values, euler_values, vb_values, pqr_values, 
    forces_values as expected_forces_values, 
    moments_values as expected_moments_values
) 

import jax.numpy as jnp

expected_actuator_outputs_values = jnp.array([
    [30.000000, 0.069813, 0.000000, 0.000000, 0.000000, 0.000000, 0.069813, 0.000000, 7500.000000, 0.069813, 0.069813, 0.000000, 0.000000, 0.011636, 0.000000, 0.069813, 0.000000, 7500.000000, 0.069813, 0.069813, 0.000000, 0.000000, 0.069813, 0.000000, 551.250000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
    [29.956949, 0.065698, 0.000038, -0.007684, -0.154990, 0.000185, 0.065698, 0.000038, 7500.000000, 0.065698, 0.065698, 0.000038, 0.000038, 0.010950, 0.000006, 0.065698, 0.000038, 7500.000000, 0.065698, 0.065698, 0.000038, 0.000038, 0.065698, 0.000038, 549.669018, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000], 
    [29.914949, 0.060331, 0.000153, -0.015420, -0.297333, 0.000410, 0.060331, 0.000153, 7500.000000, 0.060331, 0.060331, 0.000153, 0.000153, 0.010055, 0.000025, 0.060331, 0.000153, 7500.000000, 0.060331, 0.060331, 0.000153, 0.000153, 0.060331, 0.000153, 548.128813, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
    [29.874238, 0.053863, 0.000344, -0.023234, -0.426201, 0.000706, 0.053863, 0.000344, 7500.000000, 0.053863, 0.053863, 0.000344, 0.000344, 0.008977, 0.000057, 0.053863, 0.000344, 7500.000000, 0.053863, 0.053863, 0.000344, 0.000344, 0.053863, 0.000344, 546.637949, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
    [29.834992, 0.046450, 0.000612, -0.031147, -0.540992, 0.001099, 0.046450, 0.000612, 7500.000000, 0.046450, 0.046450, 0.000612, 0.000612, 0.007742, 0.000102, 0.046450, 0.000612, 7500.000000, 0.046450, 0.046450, 0.000612, 0.000612, 0.046450, 0.000612, 545.202647, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
    [29.797323, 0.038246, 0.000957, -0.039181, -0.641322, 0.001618, 0.038246, 0.000957, 7500.000000, 0.038246, 0.038246, 0.000957, 0.000957, 0.006374, 0.000159, 0.038246, 0.000957, 7500.000000, 0.038246, 0.038246, 0.000957, 0.000957, 0.038246, 0.000957, 543.826784, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
    [29.761282, 0.029406, 0.001380, -0.047358, -0.727013, 0.002291, 0.029406, 0.001380, 7500.000000, 0.029406, 0.029406, 0.001380, 0.001380, 0.004901, 0.000230, 0.029406, 0.001380, 7500.000000, 0.029406, 0.029406, 0.001380, 0.001380, 0.029406, 0.001380, 542.512006, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
    [29.726863, 0.020080, 0.001883, -0.055700, -0.798073, 0.003143, 0.020080, 0.001883, 7500.000000, 0.020080, 0.020080, 0.001883, 0.001883, 0.003347, 0.000314, 0.020080, 0.001883, 7500.000000, 0.020080, 0.020080, 0.001883, 0.001883, 0.020080, 0.001883, 541.257922, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
    [29.694014, 0.010414, 0.002466, -0.064231, -0.854687, 0.004201, 0.010414, 0.002466, 7500.000000, 0.010414, 0.010414, 0.002466, 0.002466, 0.001736, 0.000411, 0.010414, 0.002466, 7500.000000, 0.010414, 0.010414, 0.002466, 0.002466, 0.010414, 0.002466, 540.062372, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
    [29.662640, 0.000549, 0.003131, -0.072976, -0.897200, 0.005492, 0.000549, 0.003131, 7500.000000, 0.000549, 0.000549, 0.003131, 0.003131, 0.000091, 0.000522, 0.000549, 0.003131, 7500.000000, 0.000549, 0.000549, 0.003131, 0.003131, 0.000549, 0.003131, 538.921725, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
    [29.632640, -0.009384, 0.003881, -0.081987, -0.926122, 0.007043, -0.009384, 0.003881, 7500.000000, -0.009384, -0.009384, 0.003881, 0.003881, -0.001564, 0.000647, -0.009384, 0.003881, 7500.000000, -0.009384, -0.009384, 0.003881, 0.003881, -0.009384, 0.003881, 537.832186, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000, 7500.000000],
])
 
expected_Cx_outputs_values = jnp.array([
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

expected_Cy_outputs_values = jnp.array([
        [0, 1.54656473174905e-15, 0],
        [0, -0.0160390391484072, 0],
        [0, -0.0611958036837794, 0],
        [0, -0.134389887735588, 0],
        [0, -0.234701364158825, 0],
        [0, -0.361370119624134, 0],
        [0, -0.513797036084389, 0],
        [0, -0.691546661577133, 0],
        [0, -0.894350936998841, 0],
        [0, -1.12211352186119, 0],
        [0, -1.3749283426252, 0]
    ]
)

expected_Cz_outputs_values = jnp.array([
        [0, 0, -264.893059183493],
        [0, 0, -247.330317583035],
        [0, 0, -227.927496579797],
        [0, 0, -207.063069903171],
        [0, 0, -185.102372839729],
        [0, 0, -162.394413922597],
        [0, 0, -139.269299081639],
        [0, 0, -116.036178507391],
        [0, 0, -92.9816327558558],
        [0, 0, -70.3684267025321],
        [0, 0, -48.4869242664429]
    ]
)

expected_Cl_outputs_values = jnp.array([
    [-3.71891712080154, 0, 0],
    [-3.73892743798086, 0, 0],
    [-3.770934313081, 0, 0],
    [-3.81434583871129, 0, 0],
    [-3.86884630584595, 0, 0],
    [-3.93440170934839, 0, 0],
    [-4.01125413670425, 0, 0],
    [-4.09990704443226, 0, 0],
    [-4.20110335307348, 0, 0],
    [-4.31579817794609, 0, 0],
    [-4.47116516425064, 0, 0]
    ]
)

expected_Cm_outputs_values = jnp.array([
        [0, -97.4271038597041, 0],
        [0, -90.0901400268625, 0],
        [0, -82.1812570744447, 0],
        [0, -73.8415402372756, 0],
        [0, -65.205648840758, 0],
        [0, -56.4007941861578, 0],
        [0, -47.5459473602478, 0],
        [0, -38.7512399003046, 0],
        [0, -30.1175236238111, 0],
        [0, -21.7360612793007, 0],
        [0, -13.7178408886391, 0]
    ]
)

expected_Cn_outputs_values = jnp.array([
        [0, 0, 0.164977625437221],
        [0, 0, 0.191046199997934],
        [0, 0, 0.247575681429308],
        [0, 0, 0.333546638829987],
        [0, 0, 0.44799062413523],
        [0, 0, 0.59000322610442],
        [0, 0, 0.758765968959369],
        [0, 0, 0.953574138754955],
        [0, 0, 1.1738675465473],
        [0, 0, 1.41926139470376],
        [0, 0, 1.69101351913533]
        ])

expected_Ct_outputs_values = jnp.array([
    [23.8478523411019, 0, 0, 0, 0.210084930483265, 1.77635683940025e-15],
    [24.0653722230781, 0, 0, 0, 0.212208064150277, 0],
    [24.267708349065, 0, 0, 0, 0.214335340663162, 4.44089209850063e-16],
    [24.4546794297957, 0, 0, 0, 0.216449502182345, 1.33226762955019e-15],
    [24.6264819613241, 0, 0, 0, 0.218536062825545, -4.44089209850063e-16],
    [24.7836771443068, 0, 0, 0, 0.220583408644189, -8.88178419700125e-16],
    [24.9271621051189, 0, 0, 0, 0.222582756561505, 0],
    [25.058128498218, 0, 0, 0, 0.22452799637347, 4.44089209850063e-16],
    [25.1780116995042, 0, 0, 0, 0.226415441147283, -4.44089209850063e-16],
    [25.2884337985374, 0, 0, 0, 0.228243511444974, -4.44089209850063e-16],
    [25.5345143902457, 0, 0, 0, 0.229192307180639, 0]
])

expected_Kq_outputs_values = jnp.zeros((11, 3))



assert expected_actuator_outputs_values.shape == (11, 45)
 
@pytest.fixture
def mavrik_aero():
    mavrik_setup = MavrikSetup(file_path=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "jax_mavrik/aero_export.mat")
    )
    return MavrikAero(mavrik_setup=mavrik_setup)


@pytest.fixture
def control_inputs():
    return ControlInputs(
        wing_tilt=0.0, tail_tilt=0.0, aileron=0.0,
        elevator=0.0, flap=0.0, rudder=0.0,
        RPM_tailLeft=7500, RPM_tailRight=7500,
        RPM_leftOut1=7500, RPM_left2=7500,
        RPM_left3=7500, RPM_left4=7500,
        RPM_left5=7500, RPM_left6In=7500,
        RPM_right7In=7500, RPM_right8=7500,
        RPM_right9=7500, RPM_right10=7500,
        RPM_right11=7500, RPM_right12Out=7500
    )

@pytest.mark.parametrize(
    "id, vned, xned, euler, vb, pqr, expected_forces, expected_moments, expected_actuator_outputs_values, expected_Cx_outputs_values, expected_Cy_outputs_values, expected_Cz_outputs_values, expected_Cl_outputs_values, expected_Cm_outputs_values, expected_Cn_outputs_values, expected_Ct_outputs_values, expected_Kq_outputs_values",
    zip(
        list(range(11)),
        vned_values, xned_values, euler_values, vb_values, pqr_values, expected_forces_values, expected_moments_values,
        expected_actuator_outputs_values, expected_Cx_outputs_values, expected_Cy_outputs_values, expected_Cz_outputs_values,
        expected_Cl_outputs_values, expected_Cm_outputs_values, expected_Cn_outputs_values, expected_Ct_outputs_values, expected_Kq_outputs_values
    )
)

def test_mavrik_aero(id, mavrik_aero, control_inputs, vned, xned, euler, vb, pqr, expected_forces, expected_moments, expected_actuator_outputs_values, expected_Cx_outputs_values, expected_Cy_outputs_values, expected_Cz_outputs_values, expected_Cl_outputs_values, expected_Cm_outputs_values, expected_Cn_outputs_values, expected_Ct_outputs_values, expected_Kq_outputs_values):
    state = StateVariables(
        u=vb[0], v=vb[1], w=vb[2],
        Xe=xned[0], Ye=xned[1], Ze=xned[2],
        roll=euler[0], pitch=euler[1], yaw=euler[2],
        VXe=vned[0], VYe=vned[1], VZe=vned[2],
        p=pqr[0], q=pqr[1], r=pqr[2], 
        Fx = 0, Fy = 0, Fz = 0,
        L = 0, M = 0, N = 0
    )

    forces, moments, actuator_outputs = mavrik_aero(state, control_inputs)
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

    print(f">>>>>>>>>>>>>>>>>>>> Test ID: {id} <<<<<<<<<<<<<<<<<<<<<<")
    actuator_close = jnp.allclose(actuator_outputs_array, expected_actuator_outputs_values, atol=0.001)
    print('Actuator Outputs close???', actuator_close)
    if not actuator_close:
        print(f"\n  Expected: {expected_actuator_outputs_values}\n  Got: {actuator_outputs_array}")
        max_diff_index = jnp.argmax(jnp.abs(actuator_outputs_array - expected_actuator_outputs_values))
        print(f"\n  Max difference at index {max_diff_index}: Expected {expected_actuator_outputs_values[max_diff_index]}, Got {actuator_outputs_array[max_diff_index]}\n\n")
     
    wing_transform = jnp.array([[jnp.cos(actuator_outputs.wing_tilt), 0, jnp.sin(actuator_outputs.wing_tilt)], [0, 1, 0], [-jnp.sin(actuator_outputs.wing_tilt), 0., jnp.cos(actuator_outputs.wing_tilt)]]);
    tail_transform = jnp.array([[jnp.cos(actuator_outputs.tail_tilt), 0, jnp.sin(actuator_outputs.tail_tilt)], [0, 1, 0], [-jnp.sin(actuator_outputs.tail_tilt), 0., jnp.cos(actuator_outputs.tail_tilt)]])

    
    F0, M0 = mavrik_aero.Ct(actuator_outputs, wing_transform, tail_transform)
    Ct_array = jnp.array([F0.Fx, F0.Fy, F0.Fz, M0.L, M0.M, M0.N])
    F1 = mavrik_aero.Cx(actuator_outputs, wing_transform, tail_transform)
    Cx_array = jnp.array([F1.Fx, F1.Fy, F1.Fz])
    F2 = mavrik_aero.Cy(actuator_outputs, wing_transform, tail_transform)
    Cy_array = jnp.array([F2.Fx, F2.Fy, F2.Fz])
    F3 = mavrik_aero.Cz(actuator_outputs, wing_transform, tail_transform)
    Cz_array = jnp.array([F3.Fx, F3.Fy, F3.Fz])
    M1 = mavrik_aero.L(actuator_outputs, wing_transform, tail_transform)
    Cl_array = jnp.array([M1.L, M1.M, M1.N])
    M2 = mavrik_aero.M(actuator_outputs, wing_transform, tail_transform)
    Cm_array = jnp.array([M2.L, M2.M, M2.N])
    M3 = mavrik_aero.N(actuator_outputs, wing_transform, tail_transform)
    Cn_array = jnp.array([M3.L, M3.M, M3.N])
    M5 = mavrik_aero.Kq(actuator_outputs, wing_transform, tail_transform)
    Kq_array = jnp.array([M5.L, M5.M, M5.N])
    

    Ct_close = jnp.allclose(Ct_array, expected_Ct_outputs_values, atol=0.0001)
    print("Ct Outputs close???", Ct_close)
    if not Ct_close:
        print(f"\n  Expected: {expected_Ct_outputs_values}\n  Got: {Ct_array}")
        max_diff_index_Ct = jnp.argmax(jnp.abs(Ct_array - expected_Ct_outputs_values))
        print(f"\n  Max difference in Ct at index {max_diff_index_Ct}: Expected {expected_Ct_outputs_values[max_diff_index_Ct]}, Got {Ct_array[max_diff_index_Ct]}")

    Cn_close = jnp.allclose(Cn_array, expected_Cn_outputs_values, atol=0.0001)
    print("Cn Outputs close???", Cn_close)
    if not Cn_close:
        print(f"\n  Expected: {expected_Cn_outputs_values}\n  Got: {Cn_array}")
        max_diff_index_Cn = jnp.argmax(jnp.abs(Cn_array - expected_Cn_outputs_values))
        print(f"\n  Max difference in Cn at index {max_diff_index_Cn}: Expected {expected_Cn_outputs_values[max_diff_index_Cn]}, Got {Cn_array[max_diff_index_Cn]}")

    Cx_close = jnp.allclose(Cx_array, expected_Cx_outputs_values, atol=0.0001)
    print("Cx Outputs close???", Cx_close)
    if not Cx_close:
        print(f"\n  Expected: {expected_Cx_outputs_values}\n  Got: {Cx_array}")
        max_diff_index_Cx = jnp.argmax(jnp.abs(Cx_array - expected_Cx_outputs_values))
        print(f"\n  Max difference in Cx at index {max_diff_index_Cx}: Expected {expected_Cx_outputs_values[max_diff_index_Cx]}, Got {Cx_array[max_diff_index_Cx]}")

    Cy_close = jnp.allclose(Cy_array, expected_Cy_outputs_values, atol=0.0001)
    print("Cy Outputs close???", Cy_close)
    if not Cy_close:
        print(f"\n ActuatorOuputs As Expected??? {(actuator_outputs_array==expected_actuator_outputs_values)}")
        print(f"{jnp.allclose(actuator_outputs_array, expected_actuator_outputs_values, atol=0.0001)}")
        print(f"\n  Expected: {expected_Cy_outputs_values}\n  Got: {Cy_array}")
        max_diff_index_Cy = jnp.argmax(jnp.abs(Cy_array - expected_Cy_outputs_values))
        print(f"\n  Max difference in Cy at index {max_diff_index_Cy}: Expected {expected_Cy_outputs_values[max_diff_index_Cy]}, Got {Cy_array[max_diff_index_Cy]}")

    Cz_close = jnp.allclose(Cz_array, expected_Cz_outputs_values, atol=0.0001)
    print("Cz Outputs close???", Cz_close)
    if not Cz_close:
        print(f"\n  Expected: {expected_Cz_outputs_values}\n  Got: {Cz_array}")
        max_diff_index_Cz = jnp.argmax(jnp.abs(Cz_array - expected_Cz_outputs_values))
        print(f"\n  Max difference in Cz at index {max_diff_index_Cz}: Expected {expected_Cz_outputs_values[max_diff_index_Cz]}, Got {Cz_array[max_diff_index_Cz]}")

    Cl_close = jnp.allclose(Cl_array, expected_Cl_outputs_values, atol=0.0001)
    print("Cl Outputs close???", Cl_close)
    if not Cl_close:
        print(f"\n  Expected: {expected_Cl_outputs_values}\n  Got: {Cl_array}")
        max_diff_index_Cl = jnp.argmax(jnp.abs(Cl_array - expected_Cl_outputs_values))
        print(f"\n  Max difference in Cl at index {max_diff_index_Cl}: Expected {expected_Cl_outputs_values[max_diff_index_Cl]}, Got {Cl_array[max_diff_index_Cl]}")

    Cm_close = jnp.allclose(Cm_array, expected_Cm_outputs_values, atol=0.0001)
    print("Cm Outputs close???", Cm_close)
    if not Cm_close:
        print(f"\n  Expected: {expected_Cm_outputs_values}\n  Got: {Cm_array}")
        max_diff_index_Cm = jnp.argmax(jnp.abs(Cm_array - expected_Cm_outputs_values))
        print(f"\n  Max difference in Cm at index {max_diff_index_Cm}: Expected {expected_Cm_outputs_values[max_diff_index_Cm]}, Got {Cm_array[max_diff_index_Cm]}")

    Kq_close = jnp.allclose(Kq_array, expected_Kq_outputs_values, atol=0.0001)
    print("Kq Outputs close???", Kq_close)
    if not Kq_close:
        print(f"\n  Expected: {expected_Kq_outputs_values}\n  Got: {Kq_array}")
        max_diff_index_Kq = jnp.argmax(jnp.abs(Kq_array - expected_Kq_outputs_values))
        print(f"\n  Max difference in Kq at index {max_diff_index_Kq}: Expected {expected_Kq_outputs_values[max_diff_index_Kq]}, Got {Kq_array[max_diff_index_Kq]}")

    forces_array = jnp.array([forces.Fx, forces.Fy, forces.Fz])
    moments_array = jnp.array([moments.L, moments.M, moments.N])

    forces_close = jnp.allclose(forces_array, expected_forces, atol=0.0001)
    print("Forces close???", forces_close)
    if not forces_close:
        print( f"\n  Expected: {expected_forces}\n  Got: {forces_array}")
        max_diff_index_forces = jnp.argmax(jnp.abs(forces_array - expected_forces))
        print(f"\n  Max difference in forces at index {max_diff_index_forces}: Expected {expected_forces[max_diff_index_forces]}, Got {forces_array[max_diff_index_forces]}")

    moments_close =  jnp.allclose(moments_array, expected_moments, atol=0.0001)
    print("Moments close???", moments_close)
    if not moments_close:
        print(f"\n  Expected: {expected_moments}\n  Got: {moments_array}")
        max_diff_index_moments = jnp.argmax(jnp.abs(moments_array - expected_moments))
        print(f"\n  Max difference in moments at index {max_diff_index_moments}: Expected {expected_moments[max_diff_index_moments]}, Got {moments_array[max_diff_index_moments]}")
