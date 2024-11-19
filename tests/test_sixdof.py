import os
import sys
import pytest
from jax_mavrik.src.sixdof import SixDOFDynamics, RigidBody, SixDOFState
from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.mavrik_types import StateVariables, ControlInputs

import jax.numpy as jnp
 
@pytest.fixture
def rigid_body():
    mavrik_setup = MavrikSetup(file_path=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "jax_mavrik/aero_export.mat")
    )
    mass = mavrik_setup.mass
    inertia = mavrik_setup.inertia
    return RigidBody(mass=mass, inertia=inertia)

@pytest.mark.parametrize("vned, xned, euler, v, a, pqr, forces, moments, expected_vned, expected_xned, expected_euler, expected_v, expected_pqr", [
    (jnp.array([30.0000, 0, 0]), jnp.array([0, 0, 0]), jnp.array([0, 0.0698, 0]), jnp.array([29.9269, 0, 2.0927]), jnp.array([-3.6179, 0, -10.5957]), jnp.array([0, 0, 0]), jnp.array([-90.4479, 0, -264.8931]), jnp.array([-3.7189, -97.2170, 0.1650]), jnp.array([29.9568, -0.0000, -0.0997]), jnp.array([0.2998, -0.0000, -0.0005]), jnp.array([-0.0000, 0.0690, 0.0000]), jnp.array([29.8923, -0.0001, 1.9667]), jnp.array([-0.0077, -0.1550, 0.0002])),
    (jnp.array([29.9568, -0.0000, -0.0997]), jnp.array([0.2998, -0.0000, -0.0005]), jnp.array([-0.0000, 0.0690, 0.0000]), jnp.array([29.8923, -0.0001, 1.9667]), jnp.array([-3.3106, -0.0213, -14.5262]), jnp.array([-0.0077, -0.1550, 0.0002]), jnp.array([-90.3843, -0.0160, -247.3303]), jnp.array([-3.7389, -89.8779, 0.1910]), jnp.array([29.9143, -0.0000, -0.1922]), jnp.array([0.5991, -0.0000, -0.0020]), jnp.array([-0.0002, 0.0668, 0.0000]), jnp.array([29.8605, -0.0004, 1.8037]), jnp.array([-0.0154, -0.2973, 0.0004])),
    (jnp.array([29.9143, -0.0000, -0.1922]), jnp.array([0.5991, -0.0000, -0.0020]), jnp.array([-0.0002, 0.0668, 0.0000]), jnp.array([29.8605, -0.0004, 1.8037]), jnp.array([-3.0595, -0.0425, -17.9956]), jnp.array([-0.0154, -0.2973, 0.0004]), jnp.array([-89.8961, -0.0612, -227.9275]), jnp.array([-3.7709, -81.9669, 0.2476]), jnp.array([29.8730, -0.0001, -0.2767]), jnp.array([0.8981, -0.0000, -0.0043]), jnp.array([-0.0003, 0.0631, 0.0000]), jnp.array([29.8309, -0.0010, 1.6083]), jnp.array([-0.0232, -0.4262, 0.0007])),
    (jnp.array([29.8730, -0.0001, -0.2767]), jnp.array([0.8981, -0.0000, -0.0043]), jnp.array([-0.0003, 0.0631, 0.0000]), jnp.array([29.8309, -0.0010, 1.6083]), jnp.array([-2.8740, -0.0638, -20.9965]), jnp.array([-0.0232, -0.4262, 0.0007]), jnp.array([-88.9870, -0.1344, -207.0631]), jnp.array([-3.8143, -73.6251, 0.3335]), jnp.array([29.8329, -0.0002, -0.3529]), jnp.array([1.1966, -0.0000, -0.0075]), jnp.array([-0.0006, 0.0583, 0.0000]), jnp.array([29.8028, -0.0017, 1.3853]), jnp.array([-0.0311, -0.5410, 0.0011])),
    (jnp.array([29.8329, -0.0002, -0.3529]), jnp.array([1.1966, -0.0000, -0.0075]), jnp.array([-0.0006, 0.0583, 0.0000]), jnp.array([29.8028, -0.0017, 1.3853]), jnp.array([-2.7576, -0.0853, -23.5272]), jnp.array([-0.0311, -0.5410, 0.0011]), jnp.array([-87.6768, -0.2347, -185.1024]), jnp.array([-3.8688, -64.9871, 0.4480]), jnp.array([29.7944, -0.0004, -0.4203]), jnp.array([1.4947, -0.0000, -0.0114]), jnp.array([-0.0010, 0.0524, 0.0000]), jnp.array([29.7755, -0.0027, 1.1393]), jnp.array([-0.0392, -0.6413, 0.0016])),
    (jnp.array([29.7944, -0.0004, -0.4203]), jnp.array([1.4947, -0.0000, -0.0114]), jnp.array([-0.0010, 0.0524, 0.0000]), jnp.array([29.7755, -0.0027, 1.1393]), jnp.array([-2.7093, -0.1073, -25.5916]), jnp.array([-0.0392, -0.6413, 0.0016]), jnp.array([-85.9998, -0.3614, -162.3944]), jnp.array([-3.9344, -56.1802, 0.5900]), jnp.array([29.7574, -0.0006, -0.4789]), jnp.array([1.7925, -0.0000, -0.0159]), jnp.array([-0.0014, 0.0455, 0.0001]), jnp.array([29.7484, -0.0038, 0.8750]), jnp.array([-0.0474, -0.7270, 0.0023])),
    (jnp.array([29.7574, -0.0006, -0.4789]), jnp.array([1.7925, -0.0000, -0.0159]), jnp.array([-0.0014, 0.0455, 0.0001]), jnp.array([29.7484, -0.0038, 0.8750]), jnp.array([-2.7239, -0.1301, -27.1984]), jnp.array([-0.0474, -0.7270, 0.0023]), jnp.array([-84.0019, -0.5138, -139.2693]), jnp.array([-4.0113, -47.3234, 0.7588]), jnp.array([29.7222, -0.0009, -0.5286]), jnp.array([2.0899, -0.0000, -0.0209]), jnp.array([-0.0019, 0.0379, 0.0001]), jnp.array([29.7209, -0.0053, 0.5969]), jnp.array([-0.0557, -0.7981, 0.0031])),
    (jnp.array([29.7222, -0.0009, -0.5286]), jnp.array([2.0899, -0.0000, -0.0209]), jnp.array([-0.0019, 0.0379, 0.0001]), jnp.array([29.7209, -0.0053, 0.5969]), jnp.array([-2.7932, -0.1543, -28.3612]), jnp.array([-0.0557, -0.7981, 0.0031]), jnp.array([-81.7374, -0.6915, -116.0362]), jnp.array([-4.0999, -38.5267, 0.9536]), jnp.array([29.6886, -0.0014, -0.5692]), jnp.array([2.3869, -0.0000, -0.0264]), jnp.array([-0.0025, 0.0296, 0.0002]), jnp.array([29.6924, -0.0069, 0.3092]), jnp.array([-0.0642, -0.8547, 0.0042])),
    (jnp.array([29.6886, -0.0014, -0.5692]), jnp.array([2.3869, -0.0000, -0.0264]), jnp.array([-0.0025, 0.0296, 0.0002]), jnp.array([29.6924, -0.0069, 0.3092]), jnp.array([-2.9064, -0.1804, -29.0974]), jnp.array([-0.0642, -0.8547, 0.0042]), jnp.array([-79.2664, -0.8944, -92.9816]), jnp.array([-4.2011, -29.8911, 1.1739]), jnp.array([29.6565, -0.0019, -0.6011]), jnp.array([2.6837, -0.0000, -0.0323]), jnp.array([-0.0032, 0.0208, 0.0002]), jnp.array([29.6626, -0.0089, 0.0163]), jnp.array([-0.0730, -0.8972, 0.0055])),
    (jnp.array([29.6565, -0.0019, -0.6011]), jnp.array([2.6837, -0.0000, -0.0323]), jnp.array([-0.0032, 0.0208, 0.0002]), jnp.array([29.6626, -0.0089, 0.0163]), jnp.array([-3.0515, -0.2090, -29.4287]), jnp.array([-0.0730, -0.8972, 0.0055]), jnp.array([-76.6521, -1.1221, -70.3684]), jnp.array([-4.3158, -21.5078, 1.4193]), jnp.array([29.6261, -0.0024, -0.6243]), jnp.array([2.9801, -0.0001, -0.0384]), jnp.array([-0.0040, 0.0117, 0.0003]), jnp.array([29.6313, -0.0111, -0.2781]), jnp.array([-0.0820, -0.9261, 0.0070])),
    (jnp.array([29.6261, -0.0024, -0.6243]), jnp.array([2.9801, -0.0001, -0.0384]), jnp.array([-0.0040, 0.0117, 0.0003]), jnp.array([29.6313, -0.0111, -0.2781]), jnp.array([-3.2104, -0.2409, -29.3826]), jnp.array([-0.0820, -0.9261, 0.0070]), jnp.array([-73.8203, -1.3749, -48.4869]), jnp.array([-4.4712, -13.4886, 1.6910]), jnp.array([29.6261, -0.0024, -0.6243]), jnp.array([2.9801, -0.0001, -0.0384]), jnp.array([-0.0040, 0.0117, 0.0003]), jnp.array([29.6313, -0.0111, -0.2781]), jnp.array([-0.0820, -0.9261, 0.0070]))
])

def test_sixdof(rigid_body, vned, xned, euler, v, a, pqr, forces, moments, expected_vned, expected_xned, expected_euler, expected_v, expected_pqr):
    initial_state = SixDOFState(
        Ve=vned,
        Xe=xned, 
        Vb=v,
        Euler=euler,
        pqr=pqr
    )
    dynamics = SixDOFDynamics(rigid_body, method="diffrax", fixed_step_size=0.01)
    results = dynamics.run_simulation(initial_state, forces, moments, 0, 0.01)
    nxt_state = results["states"][-1]
    nxt_ned = nxt_state[:3]
    nxt_xned = nxt_state[3:6]
    nxt_v = nxt_state[6:9]
    nxt_euler = nxt_state[9:12]
    nxt_pqr = nxt_state[12:15]
    threshold = 1
    print("Test Example:")
    print(f"Ve Expected: {expected_vned}, Got: {nxt_ned}, Close: {jnp.allclose(expected_vned, nxt_ned, atol=threshold)}")
    print(f"Xe Expected: {expected_xned}, Got: {nxt_xned}, Close: {jnp.allclose(expected_xned, nxt_xned, atol=threshold)}")
    print(f"Vb Expected: {expected_v}, Got: {nxt_v}, Close: {jnp.allclose(expected_v, nxt_v, atol=threshold)}")
    print(f"Euler Expected: {expected_euler}, Got: {nxt_euler}, Close: {jnp.allclose(expected_euler, nxt_euler, atol=threshold)}")
    print(f"pqr Expected: {expected_pqr}, Got: {nxt_pqr}, Close: {jnp.allclose(expected_pqr, nxt_pqr, atol=threshold)}")
     