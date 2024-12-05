# mavrik_aero.py
import functools as ft

from jax_mavrik.src.mavrik_aero import (
    CX_LOOKUP_TABLES, CY_LOOKUP_TABLES, CZ_LOOKUP_TABLES, CL_LOOKUP_TABLES, CM_LOOKUP_TABLES, CN_LOOKUP_TABLES, KQ_LOOKUP_TABLES, CT_LOOKUP_TABLES,
    get_Cx_table, get_Cy_table, get_Cz_table, get_Cl_table, get_Cm_table, get_Cn_table, get_Kq_table, get_Ct_table,
    JaxNDInterpolator, interpolate_nd, get_index_and_weight, 
    MavrikAero,
    RPM_TRANSFORMS
)
from jax_mavrik.mavrik_types import StateVariables, ControlInputs, AeroState, Forces, Moments
from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.src.actuator import ActuatorInutState, ActuatorInput, ActuatorOutput, actuate
from jax_mavrik.src.utils.mat_tools import euler_to_dcm
from jax_mavrik.src.utils.jax_types import FloatScalar

from typing import Tuple, List, NamedTuple

import numpy as np

import multiprocessing as mp
 

def CX_interpolation(table_name, u, Cx_lookup_tables, wing_transform, tail_transform, CX_Scale, CX_Scale_r, CX_Scale_p, CX_Scale_q):
    if table_name == 'aileron_wing':
        CX_aileron_wing = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron]),
            Cx_lookup_tables.CX_aileron_wing_lookup_table
        )
        CX_aileron_wing_padded = np.array([CX_aileron_wing, 0.0, 0.0])
        CX_aileron_wing_padded_transformed = np.dot(wing_transform, CX_aileron_wing_padded * CX_Scale)
        return CX_aileron_wing_padded_transformed
    elif table_name == 'elevator_tail':
        CX_elevator_tail = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator]),
            Cx_lookup_tables.CX_elevator_tail_lookup_table 
        )
        CX_elevator_tail_padded = np.array([CX_elevator_tail, 0.0, 0.0])
        CX_elevator_tail_padded_transformed = np.dot(tail_transform, CX_elevator_tail_padded * CX_Scale)
        return CX_elevator_tail_padded_transformed
    elif table_name == 'flap_wing': 
        CX_flap_wing = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap]),
            Cx_lookup_tables.CX_flap_wing_lookup_table 
        )
        CX_flap_wing_padded = np.array([CX_flap_wing, 0.0, 0.0])
        CX_flap_wing_padded_transformed = np.dot(wing_transform, CX_flap_wing_padded * CX_Scale)
        return CX_flap_wing_padded_transformed
    elif table_name == 'rudder_tail':   
        CX_rudder_tail = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder]),
            Cx_lookup_tables.CX_rudder_tail_lookup_table 
        )
        CX_rudder_tail_padded = np.array([CX_rudder_tail, 0.0, 0.0])
        CX_rudder_tail_padded_transformed = np.dot(tail_transform, CX_rudder_tail_padded * CX_Scale)
        return CX_rudder_tail_padded_transformed
    elif table_name == 'tail':
        # Tail
        CX_tail = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cx_lookup_tables.CX_tail_lookup_table 
        )
        CX_tail_padded = np.array([CX_tail, 0.0, 0.0])
        CX_tail_padded_transformed = np.dot(tail_transform, CX_tail_padded * CX_Scale)
        return CX_tail_padded_transformed
    elif table_name == 'tail_damp_p':
        # Tail Damp p
        CX_tail_damp_p = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cx_lookup_tables.CX_tail_damp_p_lookup_table 
        )
        CX_tail_damp_p_padded = np.array([CX_tail_damp_p, 0.0, 0.0])
        CX_tail_damp_p_padded_transformed = np.dot(tail_transform, CX_tail_damp_p_padded * CX_Scale_p)
        return CX_tail_damp_p_padded_transformed
    elif table_name == 'tail_damp_q':
        # Tail Damp q
        CX_tail_damp_q = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cx_lookup_tables.CX_tail_damp_q_lookup_table 
        )
        CX_tail_damp_q_padded = np.array([CX_tail_damp_q, 0.0, 0.0])
        CX_tail_damp_q_padded_transformed = np.dot(tail_transform, CX_tail_damp_q_padded * CX_Scale_q)
        return CX_tail_damp_q_padded_transformed
    elif table_name == 'tail_damp_r':
        # Tail Damp r
        CX_tail_damp_r = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cx_lookup_tables.CX_tail_damp_r_lookup_table 
        )
        CX_tail_damp_r_padded = np.array([CX_tail_damp_r, 0.0, 0.0])
        CX_tail_damp_r_padded_transformed = np.dot(tail_transform, CX_tail_damp_r_padded * CX_Scale_r)
        return CX_tail_damp_r_padded_transformed
    elif table_name == 'wing':
        # Wing
        CX_wing = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cx_lookup_tables.CX_wing_lookup_table 
        )
        CX_wing_padded = np.array([CX_wing, 0.0, 0.0])
        CX_wing_padded_transformed = np.dot(wing_transform, CX_wing_padded * CX_Scale)
        return CX_wing_padded_transformed
    elif table_name == 'wing_damp_p':
        # Wing Damp p
        CX_wing_damp_p = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cx_lookup_tables.CX_wing_damp_p_lookup_table 
        )
        CX_wing_damp_p_padded = np.array([CX_wing_damp_p, 0.0, 0.0])
        CX_wing_damp_p_padded_transformed = np.dot(wing_transform, CX_wing_damp_p_padded * CX_Scale_p)
        return CX_wing_damp_p_padded_transformed
    elif table_name == 'wing_damp_q':
        # Wing Damp q
        CX_wing_damp_q = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cx_lookup_tables.CX_wing_damp_q_lookup_table 
        )
        CX_wing_damp_q_padded = np.array([CX_wing_damp_q, 0.0, 0.0])
        CX_wing_damp_q_padded_transformed = np.dot(wing_transform, CX_wing_damp_q_padded * CX_Scale_q)
        return CX_wing_damp_q_padded_transformed
    elif table_name == 'wing_damp_r':
        # Wing Damp r
        CX_wing_damp_r = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cx_lookup_tables.CX_wing_damp_r_lookup_table 
        )
        CX_wing_damp_r_padded = np.array([CX_wing_damp_r, 0.0, 0.0])
        CX_wing_damp_r_padded_transformed = np.dot(wing_transform, CX_wing_damp_r_padded * CX_Scale_r)
        return CX_wing_damp_r_padded_transformed
    elif table_name == 'hover_fuse':
        # Hover Fuse
        CX_hover_fuse = interpolate_nd(
            np.array([u.U, u.alpha, u.beta]),
            Cx_lookup_tables.CX_hover_fuse_lookup_table 
        )
        CX_hover_fuse_padded = np.array([CX_hover_fuse * CX_Scale, 0.0, 0.0])
        return CX_hover_fuse_padded
    else:
        raise ValueError(f"Invalid table name: {table_name}")    

#@jit
def Cx(Cx_lookup_tables: CX_LOOKUP_TABLES, u: ActuatorOutput, wing_transform: FloatScalar, tail_transform: FloatScalar) -> Forces:
    CX_Scale = 0.5744 * u.Q
    CX_Scale_r = 0.5744 * 2.8270 * 1.225 * 0.25 * u.U * u.r
    CX_Scale_p = 0.5744 * 2.8270 * 1.225 * 0.25 * u.U * u.p
    CX_Scale_q = 0.5744 * 0.2032 * 1.225 * 0.25 * u.U * u.q
    
    table_names = ['aileron_wing', 'elevator_tail', 'flap_wing', 'rudder_tail', 'tail', 'tail_damp_p', 'tail_damp_q', 'tail_damp_r', 'wing', 'wing_damp_p', 'wing_damp_q', 'wing_damp_r', 'hover_fuse']
    results = []
    '''
    with mp.Pool(13) as p:
        results = p.map(ft.partial(CX_interpolation, u=u, Cx_lookup_tables=Cx_lookup_tables, wing_transform=wing_transform, tail_transform=tail_transform, CX_Scale=CX_Scale, CX_Scale_r=CX_Scale_r, CX_Scale_p=CX_Scale_p, CX_Scale_q=CX_Scale_q),
                        table_names)
    '''
    for table_name in table_names:    
        results.append(CX_interpolation(table_name, u, Cx_lookup_tables, wing_transform, tail_transform, CX_Scale, CX_Scale_r, CX_Scale_p, CX_Scale_q))
    

    
    CX_aileron_wing_padded_transformed = results[0]
    CX_elevator_tail_padded_transformed = results[1]
    CX_flap_wing_padded_transformed = results[2]
    CX_rudder_tail_padded_transformed = results[3]
    CX_tail_padded_transformed = results[4]
    CX_tail_damp_p_padded_transformed = results[5]
    CX_tail_damp_q_padded_transformed = results[6]
    CX_tail_damp_r_padded_transformed = results[7]
    CX_wing_padded_transformed = results[8]
    CX_wing_damp_p_padded_transformed = results[9]
    CX_wing_damp_q_padded_transformed = results[10]
    CX_wing_damp_r_padded_transformed = results[11]
    CX_hover_fuse_padded = results[12]

    return Forces(
        CX_aileron_wing_padded_transformed[0] + CX_elevator_tail_padded_transformed[0] + CX_flap_wing_padded_transformed[0] + CX_rudder_tail_padded_transformed[0] +
        CX_tail_padded_transformed[0] + CX_tail_damp_p_padded_transformed[0] + CX_tail_damp_q_padded_transformed[0] + CX_tail_damp_r_padded_transformed[0] +
        CX_wing_padded_transformed[0] + CX_wing_damp_p_padded_transformed[0] + CX_wing_damp_q_padded_transformed[0] + CX_wing_damp_r_padded_transformed[0] +
        CX_hover_fuse_padded[0],
        CX_aileron_wing_padded_transformed[1] + CX_elevator_tail_padded_transformed[1] + CX_flap_wing_padded_transformed[1] + CX_rudder_tail_padded_transformed[1] +
        CX_tail_padded_transformed[1] + CX_tail_damp_p_padded_transformed[1] + CX_tail_damp_q_padded_transformed[1] + CX_tail_damp_r_padded_transformed[1] +
        CX_wing_padded_transformed[1] + CX_wing_damp_p_padded_transformed[1] + CX_wing_damp_q_padded_transformed[1] + CX_wing_damp_r_padded_transformed[1] +
        CX_hover_fuse_padded[1],
        CX_aileron_wing_padded_transformed[2] + CX_elevator_tail_padded_transformed[2] + CX_flap_wing_padded_transformed[2] + CX_rudder_tail_padded_transformed[2] +
        CX_tail_padded_transformed[2] + CX_tail_damp_p_padded_transformed[2] + CX_tail_damp_q_padded_transformed[2] + CX_tail_damp_r_padded_transformed[2] +
        CX_wing_padded_transformed[2] + CX_wing_damp_p_padded_transformed[2] + CX_wing_damp_q_padded_transformed[2] + CX_wing_damp_r_padded_transformed[2] +
        CX_hover_fuse_padded[2]
    )
   

def CY_interpolation(table_name, u, Cy_lookup_tables, wing_transform, tail_transform, CY_Scale, CY_Scale_r, CY_Scale_p, CY_Scale_q):
    if table_name == 'aileron_wing':
        CY_aileron_wing = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron]),
            Cy_lookup_tables.CY_aileron_wing_lookup_table
        )
        CY_aileron_wing_padded = np.array([CY_aileron_wing, 0.0, 0.0])
        CY_aileron_wing_padded_transformed = np.dot(wing_transform, CY_aileron_wing_padded * CY_Scale)
        return CY_aileron_wing_padded_transformed
    elif table_name == 'elevator_tail':
        CY_elevator_tail = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator]),
            Cy_lookup_tables.CY_elevator_tail_lookup_table 
        )
        CY_elevator_tail_padded = np.array([CY_elevator_tail, 0.0, 0.0])
        CY_elevator_tail_padded_transformed = np.dot(tail_transform, CY_elevator_tail_padded * CY_Scale)
        return CY_elevator_tail_padded_transformed
    elif table_name == 'flap_wing': 
        CY_flap_wing = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap]),
            Cy_lookup_tables.CY_flap_wing_lookup_table 
        )
        CY_flap_wing_padded = np.array([CY_flap_wing, 0.0, 0.0])
        CY_flap_wing_padded_transformed = np.dot(wing_transform, CY_flap_wing_padded * CY_Scale)
        return CY_flap_wing_padded_transformed
    elif table_name == 'rudder_tail':   
        CY_rudder_tail = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder]),
            Cy_lookup_tables.CY_rudder_tail_lookup_table 
        )
        CY_rudder_tail_padded = np.array([CY_rudder_tail, 0.0, 0.0])
        CY_rudder_tail_padded_transformed = np.dot(tail_transform, CY_rudder_tail_padded * CY_Scale)
        return CY_rudder_tail_padded_transformed
    elif table_name == 'tail':
        # Tail
        CY_tail = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cy_lookup_tables.CY_tail_lookup_table 
        )
        CY_tail_padded = np.array([0.0, CY_tail, 0.0])
        CY_tail_padded_transformed = np.dot(tail_transform, CY_tail_padded * CY_Scale)
        return CY_tail_padded_transformed
    elif table_name == 'tail_damp_p':

        # Tail Damp p
        CY_tail_damp_p = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cy_lookup_tables.CY_tail_damp_p_lookup_table 
        )
        CY_tail_damp_p_padded = np.array([0.0, CY_tail_damp_p, 0.0])
        CY_tail_damp_p_padded_transformed = np.dot(tail_transform, CY_tail_damp_p_padded * CY_Scale_p)
        return CY_tail_damp_p_padded_transformed
    elif table_name == 'tail_damp_q':

        # Tail Damp q
        CY_tail_damp_q = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cy_lookup_tables.CY_tail_damp_q_lookup_table 
        )
        CY_tail_damp_q_padded = np.array([0.0, CY_tail_damp_q, 0.0])
        CY_tail_damp_q_padded_transformed = np.dot(tail_transform, CY_tail_damp_q_padded * CY_Scale_q)
        return CY_tail_damp_q_padded_transformed
    elif table_name == 'tail_damp_r':

        # Tail Damp r
        CY_tail_damp_r = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cy_lookup_tables.CY_tail_damp_r_lookup_table 
        )
        CY_tail_damp_r_padded = np.array([0.0, CY_tail_damp_r, 0.0])
        CY_tail_damp_r_padded_transformed = np.dot(tail_transform, CY_tail_damp_r_padded * CY_Scale_r)
        return CY_tail_damp_r_padded_transformed
    elif table_name == 'wing':

        # Wing
        CY_wing = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cy_lookup_tables.CY_wing_lookup_table 
        )
        CY_wing_padded = np.array([0.0, CY_wing, 0.0])
        CY_wing_padded_transformed = np.dot(wing_transform, CY_wing_padded * CY_Scale)
        return CY_wing_padded_transformed
    elif table_name == 'wing_damp_p':

        # Wing Damp p
        CY_wing_damp_p = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cy_lookup_tables.CY_wing_damp_p_lookup_table 
        )
        CY_wing_damp_p_padded = np.array([0.0, CY_wing_damp_p, 0.0])
        CY_wing_damp_p_padded_transformed = np.dot(wing_transform, CY_wing_damp_p_padded * CY_Scale_p)
        return CY_wing_damp_p_padded_transformed
    elif table_name == 'wing_damp_q':

        # Wing Damp q
        CY_wing_damp_q = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cy_lookup_tables.CY_wing_damp_q_lookup_table 
        )
        CY_wing_damp_q_padded = np.array([0.0, CY_wing_damp_q, 0.0])
        CY_wing_damp_q_padded_transformed = np.dot(wing_transform, CY_wing_damp_q_padded * CY_Scale_q)
        return CY_wing_damp_q_padded_transformed
    elif table_name == 'wing_damp_r':
        # Wing Damp r
        CY_wing_damp_r = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cy_lookup_tables.CY_wing_damp_r_lookup_table 
        )
        CY_wing_damp_r_padded = np.array([0.0, CY_wing_damp_r, 0.0])
        CY_wing_damp_r_padded_transformed = np.dot(wing_transform, CY_wing_damp_r_padded * CY_Scale_r)

        return CY_wing_damp_r_padded_transformed
    elif table_name == 'hover_fuse':
        # Hover Fuse
        CY_hover_fuse = interpolate_nd(
            np.array([u.U, u.alpha, u.beta]),
            Cy_lookup_tables.CY_hover_fuse_lookup_table 
        )
        CY_hover_fuse_padded = np.array([0.0, CY_hover_fuse * CY_Scale, 0.0])
        return CY_hover_fuse_padded
    else:
        raise ValueError(f"Invalid table name: {table_name}")
    
#@jit
def Cy(Cy_lookup_tables: CY_LOOKUP_TABLES, u: ActuatorOutput, wing_transform: FloatScalar, tail_transform: FloatScalar) -> Forces:
    CY_Scale = 0.5744 * u.Q
    CY_Scale_r = 0.5744 * 2.8270 * 1.225 * 0.25 * u.U * u.r
    CY_Scale_p = 0.5744 * 2.8270 * 1.225 * 0.25 * u.U * u.p
    CY_Scale_q = 0.5744 * 0.2032 * 1.225 * 0.25 * u.U * u.q

    table_names = ['aileron_wing', 'elevator_tail', 'flap_wing', 'rudder_tail', 'tail', 'tail_damp_p', 'tail_damp_q', 'tail_damp_r', 'wing', 'wing_damp_p', 'wing_damp_q', 'wing_damp_r', 'hover_fuse']
    results = []
    '''
    with mp.Pool(13) as p:
        results = p.map(ft.partial(CY_interpolation, u=u, Cy_lookup_tables=Cy_lookup_tables, wing_transform=wing_transform, tail_transform=tail_transform, CY_Scale=CY_Scale, CY_Scale_r=CY_Scale_r, CY_Scale_p=CY_Scale_p, CY_Scale_q=CY_Scale_q), table_names)
    '''
    for table_name in table_names:    
        results.append(CY_interpolation(table_name, u, Cy_lookup_tables, wing_transform, tail_transform, CY_Scale, CY_Scale_r, CY_Scale_p, CY_Scale_q))                
    
    CY_aileron_wing_padded_transformed = results[0]
    CY_elevator_tail_padded_transformed = results[1]
    CY_flap_wing_padded_transformed = results[2]
    CY_rudder_tail_padded_transformed = results[3]
    CY_tail_padded_transformed = results[4]
    CY_tail_damp_p_padded_transformed = results[5]
    CY_tail_damp_q_padded_transformed = results[6]
    CY_tail_damp_r_padded_transformed = results[7]
    CY_wing_padded_transformed = results[8]
    CY_wing_damp_p_padded_transformed = results[9]
    CY_wing_damp_q_padded_transformed = results[10]
    CY_wing_damp_r_padded_transformed = results[11]
    CY_hover_fuse_padded = results[12]



    return Forces(
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
    )
 
def CZ_interpolation(table_name, u, Cz_lookup_tables, wing_transform, tail_transform, CZ_Scale, CZ_Scale_r, CZ_Scale_p, CZ_Scale_q):
    if table_name == 'aileron_wing':
        CZ_aileron_wing = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron]),
            Cz_lookup_tables.CZ_aileron_wing_lookup_table 
        )
        CZ_aileron_wing_padded = np.array([0.0, 0.0, CZ_aileron_wing])
        CZ_aileron_wing_padded_transformed = np.dot(wing_transform, CZ_aileron_wing_padded * CZ_Scale)
        return CZ_aileron_wing_padded_transformed
    elif table_name == 'elevator_tail':
        CZ_elevator_tail = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator]),
            Cz_lookup_tables.CZ_elevator_tail_lookup_table 
        )
        CZ_elevator_tail_padded = np.array([0.0, 0.0, CZ_elevator_tail])
        CZ_elevator_tail_padded_transformed = np.dot(tail_transform, CZ_elevator_tail_padded * CZ_Scale)
        return CZ_elevator_tail_padded_transformed
    elif table_name == 'flap_wing':
        CZ_flap_wing = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap]),
            Cz_lookup_tables.CZ_flap_wing_lookup_table 
        )
        CZ_flap_wing_padded = np.array([0.0, 0.0, CZ_flap_wing])
        CZ_flap_wing_padded_transformed = np.dot(wing_transform, CZ_flap_wing_padded * CZ_Scale)
        return CZ_flap_wing_padded_transformed
    elif table_name == 'rudder_tail':
        CZ_rudder_tail = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder]),
            Cz_lookup_tables.CZ_rudder_tail_lookup_table 
        )
        CZ_rudder_tail_padded = np.array([0.0, 0.0, CZ_rudder_tail])
        CZ_rudder_tail_padded_transformed = np.dot(tail_transform, CZ_rudder_tail_padded * CZ_Scale)
        return CZ_rudder_tail_padded_transformed
    elif table_name == 'tail':
        # Tail
        CZ_tail = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cz_lookup_tables.CZ_tail_lookup_table 
        )
        CZ_tail_padded = np.array([0.0, 0.0, CZ_tail])
        CZ_tail_padded_transformed = np.dot(tail_transform, CZ_tail_padded * CZ_Scale)
        return CZ_tail_padded_transformed
    elif table_name == 'tail_damp_p':
        # Tail Damp p
        CZ_tail_damp_p = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cz_lookup_tables.CZ_tail_damp_p_lookup_table 
        )
        CZ_tail_damp_p_padded = np.array([0.0, 0.0, CZ_tail_damp_p])
        CZ_tail_damp_p_padded_transformed = np.dot(tail_transform, CZ_tail_damp_p_padded * CZ_Scale_p)
        return CZ_tail_damp_p_padded_transformed
    elif table_name == 'tail_damp_q':
        # Tail Damp q
        CZ_tail_damp_q = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cz_lookup_tables.CZ_tail_damp_q_lookup_table 
        )
        CZ_tail_damp_q_padded = np.array([0.0, 0.0, CZ_tail_damp_q])
        CZ_tail_damp_q_padded_transformed = np.dot(tail_transform, CZ_tail_damp_q_padded * CZ_Scale_q)
        return CZ_tail_damp_q_padded_transformed
    elif table_name == 'tail_damp_r':
        # Tail Damp r
        CZ_tail_damp_r = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cz_lookup_tables.CZ_tail_damp_r_lookup_table 
        )
        CZ_tail_damp_r_padded = np.array([0.0, 0.0, CZ_tail_damp_r])
        CZ_tail_damp_r_padded_transformed = np.dot(tail_transform, CZ_tail_damp_r_padded * CZ_Scale_r)
        return CZ_tail_damp_r_padded_transformed
    elif table_name == 'wing':
        # Wing
        CZ_wing = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cz_lookup_tables.CZ_wing_lookup_table 
        )
        CZ_wing_padded = np.array([0.0, 0.0, CZ_wing])
        CZ_wing_padded_transformed = np.dot(wing_transform, CZ_wing_padded * CZ_Scale)
        return CZ_wing_padded_transformed
    elif table_name == 'wing_damp_p':
        # Wing Damp p
        CZ_wing_damp_p = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cz_lookup_tables.CZ_wing_damp_p_lookup_table 
        )
        CZ_wing_damp_p_padded = np.array([0.0, 0.0, CZ_wing_damp_p])
        CZ_wing_damp_p_padded_transformed = np.dot(wing_transform, CZ_wing_damp_p_padded * CZ_Scale_p)
        return CZ_wing_damp_p_padded_transformed
    elif table_name == 'wing_damp_q':
        # Wing Damp q
        CZ_wing_damp_q = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cz_lookup_tables.CZ_wing_damp_q_lookup_table 
        )
        CZ_wing_damp_q_padded = np.array([0.0, 0.0, CZ_wing_damp_q])
        CZ_wing_damp_q_padded_transformed = np.dot(wing_transform, CZ_wing_damp_q_padded * CZ_Scale_q)
        return CZ_wing_damp_q_padded_transformed
    elif table_name == 'wing_damp_r':
        # Wing Damp r
        CZ_wing_damp_r = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cz_lookup_tables.CZ_wing_damp_r_lookup_table 
        )
        CZ_wing_damp_r_padded = np.array([0.0, 0.0, CZ_wing_damp_r])
        CZ_wing_damp_r_padded_transformed = np.dot(wing_transform, CZ_wing_damp_r_padded * CZ_Scale_r)
        return CZ_wing_damp_r_padded_transformed
    elif table_name == 'hover_fuse':
        # Hover Fuse
        CZ_hover_fuse = interpolate_nd(
            np.array([u.U, u.alpha, u.beta]),
            Cz_lookup_tables.CZ_hover_fuse_lookup_table 
        )
        CZ_hover_fuse_padded = np.array([0.0, 0.0, CZ_hover_fuse * CZ_Scale])
        return CZ_hover_fuse_padded
    else:
        raise ValueError(f"Invalid table name: {table_name}")

#@jit
def Cz(Cz_lookup_tables: CZ_LOOKUP_TABLES, u: ActuatorOutput, wing_transform: FloatScalar, tail_transform: FloatScalar) -> Forces:
    CZ_Scale = 0.5744 * u.Q
    CZ_Scale_r = 0.5744 * 2.8270 * 1.225 * 0.25 * u.U * u.r
    CZ_Scale_p = 0.5744 * 2.8270 * 1.225 * 0.25 * u.U * u.p
    CZ_Scale_q = 0.5744 * 0.2032 * 1.225 * 0.25 * u.U * u.q
        
    #wing_transform = np.array([[np.cos(u.wing_tilt), 0, np.sin(u.wing_tilt)], [0, 1, 0], [-np.sin(u.wing_tilt), 0., np.cos(u.wing_tilt)]])
    #tail_transform = np.array([[np.cos(u.tail_tilt), 0, np.sin(u.tail_tilt)], [0, 1, 0], [-np.sin(u.tail_tilt), 0., np.cos(u.tail_tilt)]])
    # Wing Damp q
     
    table_names = ['aileron_wing', 'elevator_tail', 'flap_wing', 'rudder_tail', 'tail', 'tail_damp_p', 'tail_damp_q', 'tail_damp_r', 'wing', 'wing_damp_p', 'wing_damp_q', 'wing_damp_r', 'hover_fuse']
    results = []
    '''
    with mp.Pool(13) as p:
        results = p.map(ft.partial(CZ_interpolation, u=u, Cz_lookup_tables=Cz_lookup_tables, wing_transform=wing_transform, tail_transform=tail_transform, CZ_Scale=CZ_Scale, CZ_Scale_r=CZ_Scale_r, CZ_Scale_p=CZ_Scale_p, CZ_Scale_q=CZ_Scale_q), table_names)
    '''
    for table_name in table_names:    
        results.append(CZ_interpolation(table_name, u, Cz_lookup_tables, wing_transform, tail_transform, CZ_Scale, CZ_Scale_r, CZ_Scale_p, CZ_Scale_q))

    CZ_aileron_wing_padded_transformed = results[0]
    CZ_elevator_tail_padded_transformed = results[1]
    CZ_flap_wing_padded_transformed = results[2]
    CZ_rudder_tail_padded_transformed = results[3]
    CZ_tail_padded_transformed = results[4]
    CZ_tail_damp_p_padded_transformed = results[5]
    CZ_tail_damp_q_padded_transformed = results[6]
    CZ_tail_damp_r_padded_transformed = results[7]
    CZ_wing_padded_transformed = results[8]
    CZ_wing_damp_p_padded_transformed = results[9]
    CZ_wing_damp_q_padded_transformed = results[10]
    CZ_wing_damp_r_padded_transformed = results[11]
    CZ_hover_fuse_padded = results[12]

    return Forces(
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
    )
 
def Cl_interpolation(table_name, u, Cl_lookup_tables, wing_transform, tail_transform, Cl_Scale, Cl_Scale_r, Cl_Scale_p, Cl_Scale_q):
    if table_name == 'aileron_wing':
        Cl_aileron_wing = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron]),
            Cl_lookup_tables.Cl_aileron_wing_lookup_table
        )
        Cl_aileron_wing_padded = np.array([Cl_aileron_wing, 0.0, 0.0])
        Cl_aileron_wing_padded_transformed = np.dot(wing_transform, Cl_aileron_wing_padded * Cl_Scale)
        return Cl_aileron_wing_padded_transformed
    elif table_name == 'elevator_tail':
        Cl_elevator_tail = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator]),
            Cl_lookup_tables.Cl_elevator_tail_lookup_table
        )
        Cl_elevator_tail_padded = np.array([Cl_elevator_tail, 0.0, 0.0])
        Cl_elevator_tail_padded_transformed = np.dot(tail_transform, Cl_elevator_tail_padded * Cl_Scale)
        return Cl_elevator_tail_padded_transformed
    elif table_name == 'flap_wing':
        Cl_flap_wing = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap]),
            Cl_lookup_tables.Cl_flap_wing_lookup_table
        )
        Cl_flap_wing_padded = np.array([Cl_flap_wing, 0.0, 0.0])
        Cl_flap_wing_padded_transformed = np.dot(wing_transform, Cl_flap_wing_padded * Cl_Scale)
        return Cl_flap_wing_padded_transformed
    elif table_name == 'rudder_tail':
        Cl_rudder_tail = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder]),
            Cl_lookup_tables.Cl_rudder_tail_lookup_table
        )
        Cl_rudder_tail_padded = np.array([Cl_rudder_tail, 0.0, 0.0])
        Cl_rudder_tail_padded_transformed = np.dot(tail_transform, Cl_rudder_tail_padded * Cl_Scale)
        return Cl_rudder_tail_padded_transformed
    elif table_name == 'tail':
        # Tail
        Cl_tail = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cl_lookup_tables.Cl_tail_lookup_table
        )
        Cl_tail_padded = np.array([Cl_tail, 0.0, 0.0])
        Cl_tail_padded_transformed = np.dot(tail_transform, Cl_tail_padded * Cl_Scale)
        return Cl_tail_padded_transformed
    elif table_name == 'tail_damp_p':
        # Tail Damp p
        Cl_tail_damp_p = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cl_lookup_tables.Cl_tail_damp_p_lookup_table
        )
        Cl_tail_damp_p_padded = np.array([Cl_tail_damp_p, 0.0, 0.0])
        Cl_tail_damp_p_padded_transformed = np.dot(tail_transform, Cl_tail_damp_p_padded * Cl_Scale_p)
        return Cl_tail_damp_p_padded_transformed
    elif table_name == 'tail_damp_q':
        # Tail Damp q
        Cl_tail_damp_q = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cl_lookup_tables.Cl_tail_damp_q_lookup_table
        )
        Cl_tail_damp_q_padded = np.array([Cl_tail_damp_q, 0.0, 0.0])
        Cl_tail_damp_q_padded_transformed = np.dot(tail_transform, Cl_tail_damp_q_padded * Cl_Scale_q)
        return Cl_tail_damp_q_padded_transformed
    elif table_name == 'tail_damp_r':
        # Tail Damp r
        Cl_tail_damp_r = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cl_lookup_tables.Cl_tail_damp_r_lookup_table
        )
        Cl_tail_damp_r_padded = np.array([Cl_tail_damp_r, 0.0, 0.0])
        Cl_tail_damp_r_padded_transformed = np.dot(tail_transform, Cl_tail_damp_r_padded * Cl_Scale_r)
        return Cl_tail_damp_r_padded_transformed
    elif table_name == 'wing':
        # Wing
        Cl_wing = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cl_lookup_tables.Cl_wing_lookup_table
        )
        Cl_wing_padded = np.array([Cl_wing, 0.0, 0.0])
        Cl_wing_padded_transformed = np.dot(wing_transform, Cl_wing_padded * Cl_Scale)
        return Cl_wing_padded_transformed
    elif table_name == 'wing_damp_p':
        # Wing Damp p   
        Cl_wing_damp_p = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cl_lookup_tables.Cl_wing_damp_p_lookup_table
        )
        Cl_wing_damp_p_padded = np.array([Cl_wing_damp_p, 0.0, 0.0])
        Cl_wing_damp_p_padded_transformed = np.dot(wing_transform, Cl_wing_damp_p_padded * Cl_Scale_p)
        return Cl_wing_damp_p_padded_transformed
    elif table_name == 'wing_damp_q':
        # Wing Damp q
        Cl_wing_damp_q = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cl_lookup_tables.Cl_wing_damp_q_lookup_table
        )
        Cl_wing_damp_q_padded = np.array([Cl_wing_damp_q, 0.0, 0.0])
        Cl_wing_damp_q_padded_transformed = np.dot(wing_transform, Cl_wing_damp_q_padded * Cl_Scale_q)
        return Cl_wing_damp_q_padded_transformed
    elif table_name == 'wing_damp_r':
        # Wing Damp r
        Cl_wing_damp_r = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cl_lookup_tables.Cl_wing_damp_r_lookup_table
        )
        Cl_wing_damp_r_padded = np.array([Cl_wing_damp_r, 0.0, 0.0])
        Cl_wing_damp_r_padded_transformed = np.dot(wing_transform, Cl_wing_damp_r_padded * Cl_Scale_r)
        return Cl_wing_damp_r_padded_transformed
    elif table_name == 'hover_fuse':
        # Hover Fuse
        Cl_hover_fuse = interpolate_nd(
            np.array([u.U, u.alpha, u.beta]),
            Cl_lookup_tables.Cl_hover_fuse_lookup_table
        )
        Cl_hover_fuse_padded = np.array([Cl_hover_fuse * Cl_Scale, 0.0, 0.0])
        return Cl_hover_fuse_padded
    else:
        raise ValueError(f"Invalid table name: {table_name}")
    
#@jit
def L(Cl_lookup_tables: CL_LOOKUP_TABLES, u: ActuatorOutput, wing_transform: FloatScalar, tail_transform: FloatScalar) -> Moments:
    Cl_Scale = 0.5744 * 2.8270 * u.Q
    Cl_Scale_p = 0.5744 * 2.8270**2 * 1.225 * 0.25 * u.U * u.p
    Cl_Scale_q = 0.5744 * 2.8270 * 0.2032 * 1.225 * 0.25 * u.U * u.q
    Cl_Scale_r = 0.5744 * 2.8270**2 * 1.225 * 0.25 * u.U * u.r

    table_names = ['aileron_wing', 'elevator_tail', 'flap_wing', 'rudder_tail', 'tail', 'tail_damp_p', 'tail_damp_q', 'tail_damp_r', 'wing', 'wing_damp_p', 'wing_damp_q', 'wing_damp_r', 'hover_fuse']
    results = []
    '''
    with mp.Pool(13) as p:
        results = p.map(ft.partial(Cl_interpolation, u=u, Cl_lookup_tables=Cl_lookup_tables, wing_transform=wing_transform, tail_transform=tail_transform, Cl_Scale=Cl_Scale, Cl_Scale_r=Cl_Scale_r, Cl_Scale_p=Cl_Scale_p, Cl_Scale_q=Cl_Scale_q), table_names)
    '''
    for table_name in table_names:
        results.append(Cl_interpolation(table_name, u, Cl_lookup_tables, wing_transform, tail_transform, Cl_Scale, Cl_Scale_r, Cl_Scale_p, Cl_Scale_q))

    Cl_aileron_wing_padded_transformed = results[0]
    Cl_elevator_tail_padded_transformed = results[1]
    Cl_flap_wing_padded_transformed = results[2]
    Cl_rudder_tail_padded_transformed = results[3]
    Cl_tail_padded_transformed = results[4]
    Cl_tail_damp_p_padded_transformed = results[5]
    Cl_tail_damp_q_padded_transformed = results[6]
    Cl_tail_damp_r_padded_transformed = results[7]
    Cl_wing_padded_transformed = results[8]
    Cl_wing_damp_p_padded_transformed = results[9]
    Cl_wing_damp_q_padded_transformed = results[10]
    Cl_wing_damp_r_padded_transformed = results[11]
    Cl_hover_fuse_padded = results[12]


    return Moments(
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
    )
 

def Cm_interpolation(table_name, u, Cm_lookup_tables, wing_transform, tail_transform, Cm_Scale, Cm_Scale_r, Cm_Scale_p, Cm_Scale_q):

    if table_name == 'aileron_wing':
        Cm_aileron_wing = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron]),
            Cm_lookup_tables.Cm_aileron_wing_lookup_table
        )
        Cm_aileron_wing_padded = np.array([0.0, Cm_aileron_wing, 0.0])
        Cm_aileron_wing_padded_transformed = np.dot(wing_transform, Cm_aileron_wing_padded * Cm_Scale)
        return Cm_aileron_wing_padded_transformed
    elif table_name == 'elevator_tail':
        Cm_elevator_tail = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator]),
            Cm_lookup_tables.Cm_elevator_tail_lookup_table
        )
        Cm_elevator_tail_padded = np.array([0.0, Cm_elevator_tail, 0.0])
        Cm_elevator_tail_padded_transformed = np.dot(tail_transform, Cm_elevator_tail_padded * Cm_Scale)
        return Cm_elevator_tail_padded_transformed
    elif table_name == 'flap_wing':
        Cm_flap_wing = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap]),
            Cm_lookup_tables.Cm_flap_wing_lookup_table
        )
        Cm_flap_wing_padded = np.array([0.0, Cm_flap_wing, 0.0])
        Cm_flap_wing_padded_transformed = np.dot(wing_transform, Cm_flap_wing_padded * Cm_Scale)
        return Cm_flap_wing_padded_transformed
    elif table_name == 'rudder_tail':
        Cm_rudder_tail = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder]),
            Cm_lookup_tables.Cm_rudder_tail_lookup_table
        )
        Cm_rudder_tail_padded = np.array([0.0, Cm_rudder_tail, 0.0])
        Cm_rudder_tail_padded_transformed = np.dot(tail_transform, Cm_rudder_tail_padded * Cm_Scale)
        return Cm_rudder_tail_padded_transformed
    elif table_name == 'tail':
        Cm_tail = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cm_lookup_tables.Cm_tail_lookup_table
        )
        Cm_tail_padded = np.array([0.0, Cm_tail, 0.0])
        Cm_tail_padded_transformed = np.dot(tail_transform, Cm_tail_padded * Cm_Scale)
        return Cm_tail_padded_transformed
    elif table_name == 'tail_damp_p':
        Cm_tail_damp_p = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cm_lookup_tables.Cm_tail_damp_p_lookup_table
        )
        Cm_tail_damp_p_padded = np.array([0.0, Cm_tail_damp_p, 0.0])
        Cm_tail_damp_p_padded_transformed = np.dot(tail_transform, Cm_tail_damp_p_padded * Cm_Scale_p)
        return Cm_tail_damp_p_padded_transformed
    elif table_name == 'tail_damp_q':
        Cm_tail_damp_q = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cm_lookup_tables.Cm_tail_damp_q_lookup_table
        )
        Cm_tail_damp_q_padded = np.array([0.0, Cm_tail_damp_q, 0.0])
        Cm_tail_damp_q_padded_transformed = np.dot(tail_transform, Cm_tail_damp_q_padded * Cm_Scale_q)
        return Cm_tail_damp_q_padded_transformed
    elif table_name == 'tail_damp_r':
        Cm_tail_damp_r = interpolate_nd(
            np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Cm_lookup_tables.Cm_tail_damp_r_lookup_table
        )
        Cm_tail_damp_r_padded = np.array([0.0, Cm_tail_damp_r, 0.0])
        Cm_tail_damp_r_padded_transformed = np.dot(tail_transform, Cm_tail_damp_r_padded * Cm_Scale_r)
        return Cm_tail_damp_r_padded_transformed
    elif table_name == 'wing':
        Cm_wing = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cm_lookup_tables.Cm_wing_lookup_table
        )
        Cm_wing_padded = np.array([0.0, Cm_wing, 0.0])
        Cm_wing_padded_transformed = np.dot(wing_transform, Cm_wing_padded * Cm_Scale)
        return Cm_wing_padded_transformed
    elif table_name == 'wing_damp_p':
        Cm_wing_damp_p = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cm_lookup_tables.Cm_wing_damp_p_lookup_table
        )
        Cm_wing_damp_p_padded = np.array([0.0, Cm_wing_damp_p, 0.0])
        Cm_wing_damp_p_padded_transformed = np.dot(wing_transform, Cm_wing_damp_p_padded * Cm_Scale_p)
        return Cm_wing_damp_p_padded_transformed
    elif table_name == 'wing_damp_q':
        Cm_wing_damp_q = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cm_lookup_tables.Cm_wing_damp_q_lookup_table
        )
        Cm_wing_damp_q_padded = np.array([0.0, Cm_wing_damp_q, 0.0])
        Cm_wing_damp_q_padded_transformed = np.dot(wing_transform, Cm_wing_damp_q_padded * Cm_Scale_q)
        return Cm_wing_damp_q_padded_transformed
    elif table_name == 'wing_damp_r':
        Cm_wing_damp_r = interpolate_nd(
            np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Cm_lookup_tables.Cm_wing_damp_r_lookup_table
        )
        Cm_wing_damp_r_padded = np.array([0.0, Cm_wing_damp_r, 0.0])
        Cm_wing_damp_r_padded_transformed = np.dot(wing_transform, Cm_wing_damp_r_padded * Cm_Scale_r)
        return Cm_wing_damp_r_padded_transformed
    elif table_name == 'hover_fuse':
        Cm_hover_fuse = interpolate_nd(
            np.array([u.U, u.alpha, u.beta]),
            Cm_lookup_tables.Cm_hover_fuse_lookup_table
        )
        Cm_hover_fuse_padded = np.array([0.0, Cm_hover_fuse * Cm_Scale, 0.0])
        return Cm_hover_fuse_padded
    else:
        raise ValueError(f"Invalid table name: {table_name}")

#@jit
def M(Cm_lookup_tables: CM_LOOKUP_TABLES, u: ActuatorOutput, wing_transform: FloatScalar, tail_transform: FloatScalar) -> Moments:
    Cm_Scale = 0.5744 * 0.2032 * u.Q
    Cm_Scale_p = 0.5744 * 0.2032 * 2.8270 * 1.225 * 0.25 * u.U * u.p
    Cm_Scale_q = 0.5744 * 0.2032**2 * 1.225 * 0.25 * u.U * u.q
    Cm_Scale_r = 0.5744 * 0.2032 * 2.8270 * 1.225 * 0.25 * u.U * u.r

    table_names = ['aileron_wing', 'elevator_tail', 'flap_wing', 'rudder_tail', 'tail', 'tail_damp_p', 'tail_damp_q', 'tail_damp_r', 'wing', 'wing_damp_p', 'wing_damp_q', 'wing_damp_r', 'hover_fuse']
    results = []
    '''
    with mp.Pool(13) as p:
        results = p.map(ft.partial(Cm_interpolation, u=u, Cm_lookup_tables=Cm_lookup_tables, wing_transform=wing_transform, tail_transform=tail_transform, Cm_Scale=Cm_Scale, Cm_Scale_r=Cm_Scale_r, Cm_Scale_p=Cm_Scale_p, Cm_Scale_q=Cm_Scale_q), table_names)
    '''
    for table_name in table_names:
        results.append(Cm_interpolation(table_name, u, Cm_lookup_tables, wing_transform, tail_transform, Cm_Scale, Cm_Scale_r, Cm_Scale_p, Cm_Scale_q))
                                        

    Cm_aileron_wing_padded_transformed = results[0]
    Cm_elevator_tail_padded_transformed = results[1]
    Cm_flap_wing_padded_transformed = results[2]
    Cm_rudder_tail_padded_transformed = results[3]
    Cm_tail_padded_transformed = results[4]
    Cm_tail_damp_p_padded_transformed = results[5]
    Cm_tail_damp_q_padded_transformed = results[6]
    Cm_tail_damp_r_padded_transformed = results[7]
    Cm_wing_padded_transformed = results[8]
    Cm_wing_damp_p_padded_transformed = results[9]
    Cm_wing_damp_q_padded_transformed = results[10]
    Cm_wing_damp_r_padded_transformed = results[11]
    Cm_hover_fuse_padded = results[12]
       

    return Moments(
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
    )

class CN_LOOKUP_TABLES(NamedTuple):
    Cn_aileron_wing_lookup_table: JaxNDInterpolator
    Cn_elevator_tail_lookup_table: JaxNDInterpolator
    Cn_flap_wing_lookup_table: JaxNDInterpolator
    Cn_rudder_tail_lookup_table: JaxNDInterpolator
    Cn_tail_lookup_table: JaxNDInterpolator
    Cn_tail_damp_p_lookup_table: JaxNDInterpolator
    Cn_tail_damp_q_lookup_table: JaxNDInterpolator
    Cn_tail_damp_r_lookup_table: JaxNDInterpolator
    Cn_wing_lookup_table: JaxNDInterpolator
    Cn_wing_damp_p_lookup_table: JaxNDInterpolator
    Cn_wing_damp_q_lookup_table: JaxNDInterpolator
    Cn_wing_damp_r_lookup_table: JaxNDInterpolator
    Cn_hover_fuse_lookup_table: JaxNDInterpolator

 

def Cn_interpolation(table_name, u, Cn_lookup_tables, wing_transform, tail_transform, Cn_Scale, Cn_Scale_r, Cn_Scale_p, Cn_Scale_q):

    if table_name == 'aileron_wing':
        Cn_aileron_wing = interpolate_nd(
        np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.aileron]),
        Cn_lookup_tables.Cn_aileron_wing_lookup_table
        )
        Cn_aileron_wing_padded = np.array([0.0, 0.0, Cn_aileron_wing])
        Cn_aileron_wing_padded_transformed = np.dot(wing_transform, Cn_aileron_wing_padded * Cn_Scale)
        return Cn_aileron_wing_padded_transformed
    elif table_name == 'elevator_tail':
        Cn_elevator_tail = interpolate_nd(
        np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.elevator]),
        Cn_lookup_tables.Cn_elevator_tail_lookup_table
        )
        Cn_elevator_tail_padded = np.array([0.0, 0.0, Cn_elevator_tail])
        Cn_elevator_tail_padded_transformed = np.dot(tail_transform, Cn_elevator_tail_padded * Cn_Scale)
        return Cn_elevator_tail_padded_transformed
    elif table_name == 'flap_wing':
        Cn_flap_wing = interpolate_nd(
        np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta, u.flap]),
        Cn_lookup_tables.Cn_flap_wing_lookup_table
        )
        Cn_flap_wing_padded = np.array([0.0, 0.0, Cn_flap_wing])
        Cn_flap_wing_padded_transformed = np.dot(wing_transform, Cn_flap_wing_padded * Cn_Scale)
        return Cn_flap_wing_padded_transformed
    elif table_name == 'rudder_tail':
        Cn_rudder_tail = interpolate_nd(
        np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta, u.rudder]),
        Cn_lookup_tables.Cn_rudder_tail_lookup_table
        )
        Cn_rudder_tail_padded = np.array([0.0, 0.0, Cn_rudder_tail])
        Cn_rudder_tail_padded_transformed = np.dot(tail_transform, Cn_rudder_tail_padded * Cn_Scale)
        return Cn_rudder_tail_padded_transformed
    elif table_name == 'tail':
        Cn_tail = interpolate_nd(
        np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
        Cn_lookup_tables.Cn_tail_lookup_table
        )
        Cn_tail_padded = np.array([0.0, 0.0, Cn_tail])
        Cn_tail_padded_transformed = np.dot(tail_transform, Cn_tail_padded * Cn_Scale)
        return Cn_tail_padded_transformed
    elif table_name == 'tail_damp_p':
        Cn_tail_damp_p = interpolate_nd(
        np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
        Cn_lookup_tables.Cn_tail_damp_p_lookup_table
        )
        Cn_tail_damp_p_padded = np.array([0.0, 0.0, Cn_tail_damp_p])
        Cn_tail_damp_p_padded_transformed = np.dot(tail_transform, Cn_tail_damp_p_padded * Cn_Scale_p)
        return Cn_tail_damp_p_padded_transformed
    elif table_name == 'tail_damp_q':
        Cn_tail_damp_q = interpolate_nd(
        np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
        Cn_lookup_tables.Cn_tail_damp_q_lookup_table
        )
        Cn_tail_damp_q_padded = np.array([0.0, 0.0, Cn_tail_damp_q])
        Cn_tail_damp_q_padded_transformed = np.dot(tail_transform, Cn_tail_damp_q_padded * Cn_Scale_q)
        return Cn_tail_damp_q_padded_transformed
    elif table_name == 'tail_damp_r':
        Cn_tail_damp_r = interpolate_nd(
        np.array([u.tail_alpha, u.tail_beta, u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
        Cn_lookup_tables.Cn_tail_damp_r_lookup_table
        )
        Cn_tail_damp_r_padded = np.array([0.0, 0.0, Cn_tail_damp_r])
        Cn_tail_damp_r_padded_transformed = np.dot(tail_transform, Cn_tail_damp_r_padded * Cn_Scale_r)
        return Cn_tail_damp_r_padded_transformed
    elif table_name == 'wing':
        Cn_wing = interpolate_nd(
        np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
        Cn_lookup_tables.Cn_wing_lookup_table
        )
        Cn_wing_padded = np.array([0.0, 0.0, Cn_wing])
        Cn_wing_padded_transformed = np.dot(wing_transform, Cn_wing_padded * Cn_Scale)
        return Cn_wing_padded_transformed
    elif table_name == 'wing_damp_p':
        Cn_wing_damp_p = interpolate_nd(
        np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
        Cn_lookup_tables.Cn_wing_damp_p_lookup_table
        )
        Cn_wing_damp_p_padded = np.array([0.0, 0.0, Cn_wing_damp_p])
        Cn_wing_damp_p_padded_transformed = np.dot(wing_transform, Cn_wing_damp_p_padded * Cn_Scale_p)
        return Cn_wing_damp_p_padded_transformed
    elif table_name == 'wing_damp_q':
        Cn_wing_damp_q = interpolate_nd(
        np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
        Cn_lookup_tables.Cn_wing_damp_q_lookup_table
        )
        Cn_wing_damp_q_padded = np.array([0.0, 0.0, Cn_wing_damp_q])
        Cn_wing_damp_q_padded_transformed = np.dot(wing_transform, Cn_wing_damp_q_padded * Cn_Scale_q)
        return Cn_wing_damp_q_padded_transformed
    elif table_name == 'wing_damp_r':
        Cn_wing_damp_r = interpolate_nd(
        np.array([u.wing_alpha, u.wing_beta, u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
        Cn_lookup_tables.Cn_wing_damp_r_lookup_table
        )
        Cn_wing_damp_r_padded = np.array([0.0, 0.0, Cn_wing_damp_r])
        Cn_wing_damp_r_padded_transformed = np.dot(wing_transform, Cn_wing_damp_r_padded * Cn_Scale_r)
        return Cn_wing_damp_r_padded_transformed
    elif table_name == 'hover_fuse':
        Cn_hover_fuse = interpolate_nd(
        np.array([u.U, u.alpha, u.beta]),
        Cn_lookup_tables.Cn_hover_fuse_lookup_table
        )
        Cn_hover_fuse_padded = np.array([0.0, 0.0, Cn_hover_fuse * Cn_Scale])
        return Cn_hover_fuse_padded
    else:
        raise ValueError(f"Invalid table name: {table_name}")

        
#@jit    
def N(Cn_lookup_tables: CN_LOOKUP_TABLES, u: ActuatorOutput, wing_transform: FloatScalar, tail_transform: FloatScalar) -> Moments:
    Cn_Scale = 0.5744 * 2.8270 * u.Q
    Cn_Scale_p = 0.5744 * 2.8270**2 * 1.225 * 0.25 * u.U * u.p
    Cn_Scale_q = 0.5744 * 0.2032 * 2.8270 * 1.225 * 0.25 * u.U * u.q
    Cn_Scale_r = 0.5744 * 2.8270**2 * 1.225 * 0.25 * u.U * u.r
    
    table_names = ['aileron_wing', 'elevator_tail', 'flap_wing', 'rudder_tail', 'tail', 'tail_damp_p', 'tail_damp_q', 'tail_damp_r', 'wing', 'wing_damp_p', 'wing_damp_q', 'wing_damp_r', 'hover_fuse']
    results = []
    '''
    with mp.Pool(13) as p:
        results = p.map(ft.partial(Cn_interpolation, u=u, Cn_lookup_tables=Cn_lookup_tables, wing_transform=wing_transform, tail_transform=tail_transform, Cn_Scale=Cn_Scale, Cn_Scale_r=Cn_Scale_r, Cn_Scale_p=Cn_Scale_p, Cn_Scale_q=Cn_Scale_q), table_names)
    '''
    for table_name in table_names:
        results.append(Cn_interpolation(table_name, u, Cn_lookup_tables, wing_transform, tail_transform, Cn_Scale, Cn_Scale_r, Cn_Scale_p, Cn_Scale_q))

    Cn_aileron_wing_padded_transformed = results[0]
    Cn_elevator_tail_padded_transformed = results[1]
    Cn_flap_wing_padded_transformed = results[2]
    Cn_rudder_tail_padded_transformed = results[3]
    Cn_tail_padded_transformed = results[4]
    Cn_tail_damp_p_padded_transformed = results[5]
    Cn_tail_damp_q_padded_transformed = results[6]
    Cn_tail_damp_r_padded_transformed = results[7]
    Cn_wing_padded_transformed = results[8]
    Cn_wing_damp_p_padded_transformed = results[9]
    Cn_wing_damp_q_padded_transformed = results[10]
    Cn_wing_damp_r_padded_transformed = results[11]
    Cn_hover_fuse_padded = results[12]

    return Moments(
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
    )
    

def Ct_interpolation(table_name, u, Ct_lookup_tables, wing_transform, tail_transform):
    if table_name == 'tail_left':
        Ct_tail_left = interpolate_nd(
            np.array([u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Ct_lookup_tables.Ct_tail_left_lookup_table 
        )
        Ct_tail_left_padded = np.array([Ct_tail_left, 0., 0.])
        Ct_tail_left_transformed = np.dot(tail_transform, Ct_tail_left_padded * (1.225 * u.RPM_tailLeft**2 * 0.005059318992632 * 2.777777777777778e-4))
        return Ct_tail_left_transformed
    elif table_name == 'tail_right':
        Ct_tail_right = interpolate_nd(
            np.array([u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Ct_lookup_tables.Ct_tail_right_lookup_table 
        )
        Ct_tail_right_padded = np.array([Ct_tail_right, 0., 0.])
        Ct_tail_right_transformed = np.dot(tail_transform, Ct_tail_right_padded * (1.225 * u.RPM_tailRight**2 * 0.005059318992632 * 2.777777777777778e-4))
        return Ct_tail_right_transformed
    elif table_name == 'left_out1':
        Ct_left_out1 = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Ct_lookup_tables.Ct_left_out1_lookup_table 
        )
        Ct_left_out1_padded = np.array([Ct_left_out1, 0., 0.])
        Ct_left_out1_transformed = np.dot(wing_transform, Ct_left_out1_padded * (1.225 * u.RPM_leftOut1**2 * 0.021071715921 * 2.777777777777778e-4))
        return Ct_left_out1_transformed
    elif table_name == 'left_2':
        Ct_left_2 = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Ct_lookup_tables.Ct_left_2_lookup_table 
        )
        Ct_left_2_padded = np.array([Ct_left_2, 0., 0.])
        Ct_left_2_transformed = np.dot(wing_transform, Ct_left_2_padded * (1.225 * u.RPM_left2**2 * 0.021071715921 * 2.777777777777778e-4))
        return Ct_left_2_transformed
    elif table_name == 'left_3':
        Ct_left_3 = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Ct_lookup_tables.Ct_left_3_lookup_table 
        )
        Ct_left_3_padded = np.array([Ct_left_3, 0., 0.])
        Ct_left_3_transformed = np.dot(wing_transform, Ct_left_3_padded * (1.225 * u.RPM_left3**2 * 0.021071715921 * 2.777777777777778e-4))
        return Ct_left_3_transformed
    elif table_name == 'left_4':
        Ct_left_4 = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Ct_lookup_tables.Ct_left_4_lookup_table 
        )
        Ct_left_4_padded = np.array([Ct_left_4, 0., 0.])
        Ct_left_4_transformed = np.dot(wing_transform, Ct_left_4_padded * (1.225 * u.RPM_left4**2 * 0.021071715921 * 2.777777777777778e-4))
        return Ct_left_4_transformed
    elif table_name == 'left_5':
        Ct_left_5 = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Ct_lookup_tables.Ct_left_5_lookup_table 
        )
        Ct_left_5_padded = np.array([Ct_left_5, 0., 0.])
        Ct_left_5_transformed = np.dot(wing_transform, Ct_left_5_padded * (1.225 * u.RPM_left5**2 * 0.021071715921 * 2.777777777777778e-4))
        return Ct_left_5_transformed
    elif table_name == 'left_6_in':
        Ct_left_6_in = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Ct_lookup_tables.Ct_left_6_in_lookup_table 
        )
        Ct_left_6_in_padded = np.array([Ct_left_6_in, 0., 0.])
        Ct_left_6_in_transformed = np.dot(wing_transform, Ct_left_6_in_padded * (1.225 * u.RPM_left6In**2 * 0.021071715921 * 2.777777777777778e-4))
        return Ct_left_6_in_transformed
    elif table_name == 'right_7_in':
        Ct_right_7_in = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Ct_lookup_tables.Ct_right_7_in_lookup_table 
        )
        Ct_right_7_in_padded = np.array([Ct_right_7_in, 0., 0.])
        Ct_right_7_in_transformed = np.dot(wing_transform, Ct_right_7_in_padded * (1.225 * u.RPM_right7In**2 * 0.021071715921 * 2.777777777777778e-4))
        return Ct_right_7_in_transformed
    elif table_name == 'right_8':
        Ct_right_8 = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Ct_lookup_tables.Ct_right_8_lookup_table 
        )
        Ct_right_8_padded = np.array([Ct_right_8, 0., 0.])
        Ct_right_8_transformed = np.dot(wing_transform, Ct_right_8_padded * (1.225 * u.RPM_right8**2 * 0.021071715921 * 2.777777777777778e-4))
        return Ct_right_8_transformed
    elif table_name == 'right_9':
        Ct_right_9 = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Ct_lookup_tables.Ct_right_9_lookup_table 
        )
        Ct_right_9_padded = np.array([Ct_right_9, 0., 0.])
        Ct_right_9_transformed = np.dot(wing_transform, Ct_right_9_padded * (1.225 * u.RPM_right9**2 * 0.021071715921 * 2.777777777777778e-4))
        return Ct_right_9_transformed
    elif table_name == 'right_10':
        Ct_right_10 = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Ct_lookup_tables.Ct_right_10_lookup_table 
        )
        Ct_right_10_padded = np.array([Ct_right_10, 0., 0.])
        Ct_right_10_transformed = np.dot(wing_transform, Ct_right_10_padded * (1.225 * u.RPM_right10**2 * 0.021071715921 * 2.777777777777778e-4))
        return Ct_right_10_transformed
    elif table_name == 'right_11':
        Ct_right_11 = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Ct_lookup_tables.Ct_right_11_lookup_table 
        )
        Ct_right_11_padded = np.array([Ct_right_11, 0., 0.])
        Ct_right_11_transformed = np.dot(wing_transform, Ct_right_11_padded * (1.225 * u.RPM_right11**2 * 0.021071715921 * 2.777777777777778e-4))
        return Ct_right_11_transformed
    elif table_name == 'right_12_out':
        Ct_right_12_out = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Ct_lookup_tables.Ct_right_12_out_lookup_table 
        )
        Ct_right_12_out_padded = np.array([Ct_right_12_out, 0., 0.])
        Ct_right_12_out_transformed = np.dot(wing_transform, Ct_right_12_out_padded * (1.225 * u.RPM_right12Out**2 * 0.021071715921 * 2.777777777777778e-4))
        return Ct_right_12_out_transformed
    else:
        raise ValueError(f"Invalid table name: {table_name}")

        
#@jit   
def Ct(Ct_lookup_tables: CT_LOOKUP_TABLES, rpm_transforms: RPM_TRANSFORMS, u: ActuatorOutput, wing_transform: FloatScalar, tail_transform: FloatScalar) -> Tuple[Forces, Moments]:
    table_names = ['tail_left', 'tail_right', 'left_out1', 'left_2', 'left_3', 'left_4', 'left_5', 'left_6_in', 'right_7_in', 'right_8', 'right_9', 'right_10', 'right_11', 'right_12_out']
    results = []
    '''
    with mp.Pool(14) as p:
        results = p.map(ft.partial(Ct_interpolation, u=u, Ct_lookup_tables=Ct_lookup_tables, wing_transform=wing_transform, tail_transform=tail_transform), table_names)
    '''
    for table_name in table_names:
        results.append(Ct_interpolation(table_name, u, Ct_lookup_tables, wing_transform, tail_transform))

    Ct_tail_left_transformed = results[0]
    Ct_tail_right_transformed = results[1]
    Ct_left_out1_transformed = results[2]
    Ct_left_2_transformed = results[3]
    Ct_left_3_transformed = results[4]
    Ct_left_4_transformed = results[5]
    Ct_left_5_transformed = results[6]
    Ct_left_6_in_transformed = results[7]
    Ct_right_7_in_transformed = results[8]
    Ct_right_8_transformed = results[9]
    Ct_right_9_transformed = results[10]
    Ct_right_10_transformed = results[11]
    Ct_right_11_transformed = results[12]
    Ct_right_12_out_transformed = results[13]

    forces = Forces(
        Ct_tail_left_transformed[0] + Ct_tail_right_transformed[0] + Ct_left_out1_transformed[0] + Ct_left_2_transformed[0] + Ct_left_3_transformed[0] + 
        Ct_left_4_transformed[0] + Ct_left_5_transformed[0] + Ct_left_6_in_transformed[0] + Ct_right_7_in_transformed[0] + Ct_right_8_transformed[0] + 
        Ct_right_9_transformed[0] + Ct_right_10_transformed[0] + Ct_right_11_transformed[0] + Ct_right_12_out_transformed[0],
        Ct_tail_left_transformed[1] + Ct_tail_right_transformed[1] + Ct_left_out1_transformed[1] + Ct_left_2_transformed[1] + Ct_left_3_transformed[1] + 
        Ct_left_4_transformed[1] + Ct_left_5_transformed[1] + Ct_left_6_in_transformed[1] + Ct_right_7_in_transformed[1] + Ct_right_8_transformed[1] + 
        Ct_right_9_transformed[1] + Ct_right_10_transformed[1] + Ct_right_11_transformed[1] + Ct_right_12_out_transformed[1],
        Ct_tail_left_transformed[2] + Ct_tail_right_transformed[2] + Ct_left_out1_transformed[2] + Ct_left_2_transformed[2] + Ct_left_3_transformed[2] + 
        Ct_left_4_transformed[2] + Ct_left_5_transformed[2] + Ct_left_6_in_transformed[2] + Ct_right_7_in_transformed[2] + Ct_right_8_transformed[2] + 
        Ct_right_9_transformed[2] + Ct_right_10_transformed[2] + Ct_right_11_transformed[2] + Ct_right_12_out_transformed[2]
        )

    Ct_tail_left_transformed = np.cross(rpm_transforms.RPM_tail_left_trans, Ct_tail_left_transformed)
    Ct_tail_right_transformed = np.cross(rpm_transforms.RPM_tail_right_trans, Ct_tail_right_transformed)
    Ct_left_out1_transformed = np.cross(rpm_transforms.RPM_left_out1_trans, Ct_left_out1_transformed)
    Ct_left_2_transformed = np.cross(rpm_transforms.RPM_left_2_trans, Ct_left_2_transformed)
    Ct_left_3_transformed = np.cross(rpm_transforms.RPM_left_3_trans, Ct_left_3_transformed)
    Ct_left_4_transformed = np.cross(rpm_transforms.RPM_left_4_trans, Ct_left_4_transformed)
    Ct_left_5_transformed = np.cross(rpm_transforms.RPM_left_5_trans, Ct_left_5_transformed)
    Ct_left_6_in_transformed = np.cross(rpm_transforms.RPM_left_6_in_trans, Ct_left_6_in_transformed)
    Ct_right_7_in_transformed = np.cross(rpm_transforms.RPM_right_7_in_trans, Ct_right_7_in_transformed)
    Ct_right_8_transformed = np.cross(rpm_transforms.RPM_right_8_trans, Ct_right_8_transformed)
    Ct_right_9_transformed = np.cross(rpm_transforms.RPM_right_9_trans, Ct_right_9_transformed)
    Ct_right_10_transformed = np.cross(rpm_transforms.RPM_right_10_trans, Ct_right_10_transformed)
    Ct_right_11_transformed = np.cross(rpm_transforms.RPM_right_11_trans, Ct_right_11_transformed)
    Ct_right_12_out_transformed = np.cross(rpm_transforms.RPM_right_12_out_trans, Ct_right_12_out_transformed)
    

    moments = Moments(
        Ct_tail_left_transformed[0] + Ct_tail_right_transformed[0] + Ct_left_out1_transformed[0] + Ct_left_2_transformed[0] + Ct_left_3_transformed[0] + 
        Ct_left_4_transformed[0] + Ct_left_5_transformed[0] + Ct_left_6_in_transformed[0] + Ct_right_7_in_transformed[0] + Ct_right_8_transformed[0] + 
        Ct_right_9_transformed[0] + Ct_right_10_transformed[0] + Ct_right_11_transformed[0] + Ct_right_12_out_transformed[0],
        Ct_tail_left_transformed[1] + Ct_tail_right_transformed[1] + Ct_left_out1_transformed[1] + Ct_left_2_transformed[1] + Ct_left_3_transformed[1] + 
        Ct_left_4_transformed[1] + Ct_left_5_transformed[1] + Ct_left_6_in_transformed[1] + Ct_right_7_in_transformed[1] + Ct_right_8_transformed[1] + 
        Ct_right_9_transformed[1] + Ct_right_10_transformed[1] + Ct_right_11_transformed[1] + Ct_right_12_out_transformed[1],
        Ct_tail_left_transformed[2] + Ct_tail_right_transformed[2] + Ct_left_out1_transformed[2] + Ct_left_2_transformed[2] + Ct_left_3_transformed[2] + 
        Ct_left_4_transformed[2] + Ct_left_5_transformed[2] + Ct_left_6_in_transformed[2] + Ct_right_7_in_transformed[2] + Ct_right_8_transformed[2] + 
        Ct_right_9_transformed[2] + Ct_right_10_transformed[2] + Ct_right_11_transformed[2] + Ct_right_12_out_transformed[2]
        )
    
    return forces, moments

 
def Kq_interpolation(table_name, u, Kq_lookup_tables, wing_transform, tail_transform):
    if table_name == 'tail_left':
        Kq_tail_left = interpolate_nd(
            np.array([u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Kq_lookup_tables.Kq_tail_left_lookup_table
        )
        Kq_tail_left_padded = np.array([Kq_tail_left, 0., 0.])
        Kq_tail_left_transformed = np.dot(tail_transform, Kq_tail_left_padded * (-1.225 * u.RPM_tailLeft**2 * 0.001349320375335 * 2.777777777777778e-4))
        return Kq_tail_left_transformed
    elif table_name == 'tail_right':
        Kq_tail_right = interpolate_nd(
            np.array([u.U, u.tail_RPM, u.tail_prop_alpha, u.tail_prop_beta]),
            Kq_lookup_tables.Kq_tail_right_lookup_table
        )
        Kq_tail_right_padded = np.array([Kq_tail_right, 0., 0.])
        Kq_tail_right_transformed = np.dot(tail_transform, Kq_tail_right_padded * (1.225 * u.RPM_tailRight**2 * 0.001349320375335 * 2.777777777777778e-4))
        return Kq_tail_right_transformed
    elif table_name == 'left_out':
        Kq_left_out = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Kq_lookup_tables.Kq_left_out_lookup_table
        )
        Kq_left_out_padded = np.array([Kq_left_out, 0., 0.])
        Kq_left_out_transformed = np.dot(wing_transform, Kq_left_out_padded * (1.225 * u.RPM_leftOut1**2 * 0.008028323765901 * 2.777777777777778e-4))
        return Kq_left_out_transformed
    elif table_name == 'left_2':
        Kq_left_2 = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Kq_lookup_tables.Kq_left_2_lookup_table
        )
        Kq_left_2_padded = np.array([Kq_left_2, 0., 0.])
        Kq_left_2_transformed = np.dot(wing_transform, Kq_left_2_padded * (-1.225 * u.RPM_left2**2 * 0.008028323765901 * 2.777777777777778e-4))
        return Kq_left_2_transformed
    elif table_name == 'left_3':
        Kq_left_3 = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Kq_lookup_tables.Kq_left_3_lookup_table
        )
        Kq_left_3_padded = np.array([Kq_left_3, 0., 0.])
        Kq_left_3_transformed = np.dot(wing_transform, Kq_left_3_padded * (1.225 * u.RPM_left3**2 * 0.008028323765901 * 2.777777777777778e-4))
        return Kq_left_3_transformed
    elif table_name == 'left_4':
        Kq_left_4 = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Kq_lookup_tables.Kq_left_4_lookup_table
        )
        Kq_left_4_padded = np.array([Kq_left_4, 0., 0.])
        Kq_left_4_transformed = np.dot(wing_transform, Kq_left_4_padded * (-1.225 * u.RPM_left4**2 * 0.008028323765901 * 2.777777777777778e-4))
        return Kq_left_4_transformed
    elif table_name == 'left_5':
        Kq_left_5 = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Kq_lookup_tables.Kq_left_5_lookup_table
        )
        Kq_left_5_padded = np.array([Kq_left_5, 0., 0.])
        Kq_left_5_transformed = np.dot(wing_transform, Kq_left_5_padded * (1.225 * u.RPM_left5**2 * 0.008028323765901 * 2.777777777777778e-4))
        return Kq_left_5_transformed
    elif table_name == 'left_6_in':
        Kq_left_6_in = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Kq_lookup_tables.Kq_left_6_in_lookup_table
        )
        Kq_left_6_in_padded = np.array([Kq_left_6_in, 0., 0.])
        Kq_left_6_in_transformed = np.dot(wing_transform, Kq_left_6_in_padded * (-1.225 * u.RPM_left6In**2 * 0.008028323765901 * 2.777777777777778e-4))
        return Kq_left_6_in_transformed
    elif table_name == 'right_7_in':
        Kq_right_7_in = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Kq_lookup_tables.Kq_right_7_in_lookup_table
        )
        Kq_right_7_in_padded = np.array([Kq_right_7_in, 0., 0.])
        Kq_right_7_in_transformed = np.dot(wing_transform, Kq_right_7_in_padded * (-1.225 * u.RPM_right7In**2 * 0.008028323765901 * 2.777777777777778e-4))
        return Kq_right_7_in_transformed
    elif table_name == 'right_8':
        Kq_right_8 = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Kq_lookup_tables.Kq_right_8_lookup_table
        )
        Kq_right_8_padded = np.array([Kq_right_8, 0., 0.])
        Kq_right_8_transformed = np.dot(wing_transform, Kq_right_8_padded * (1.225 * u.RPM_right8**2 * 0.008028323765901 * 2.777777777777778e-4))
        return Kq_right_8_transformed
    elif table_name == 'right_9':
        Kq_right_9 = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Kq_lookup_tables.Kq_right_9_lookup_table
        )
        Kq_right_9_padded = np.array([Kq_right_9, 0., 0.])
        Kq_right_9_transformed = np.dot(wing_transform, Kq_right_9_padded * (-1.225 * u.RPM_right9**2 * 0.008028323765901 * 2.777777777777778e-4))
        return Kq_right_9_transformed
    elif table_name == 'right_10':
        Kq_right_10 = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Kq_lookup_tables.Kq_right_10_lookup_table
        )
        Kq_right_10_padded = np.array([Kq_right_10, 0., 0.])
        Kq_right_10_transformed = np.dot(wing_transform, Kq_right_10_padded * (1.225 * u.RPM_right10**2 * 0.008028323765901 * 2.777777777777778e-4))
        return Kq_right_10_transformed
    elif table_name == 'right_11':
        Kq_right_11 = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Kq_lookup_tables.Kq_right_11_lookup_table
        )
        Kq_right_11_padded = np.array([Kq_right_11, 0., 0.])
        Kq_right_11_transformed = np.dot(wing_transform, Kq_right_11_padded * (-1.225 * u.RPM_right11**2 * 0.008028323765901 * 2.777777777777778e-4))
        return Kq_right_11_transformed
    elif table_name == 'right_12_out':
        Kq_right_12_out = interpolate_nd(
            np.array([u.U, u.wing_RPM, u.wing_prop_alpha, u.wing_prop_beta]),
            Kq_lookup_tables.Kq_right_12_out_lookup_table
        )
        Kq_right_12_out_padded = np.array([Kq_right_12_out, 0., 0.])
        Kq_right_12_out_transformed = np.dot(wing_transform, Kq_right_12_out_padded * (1.225 * u.RPM_right12Out**2 * 0.008028323765901 * 2.777777777777778e-4))
        return Kq_right_12_out_transformed
    else:
        raise ValueError(f"Invalid table name: {table_name}")

#@jit
def Kq(Kq_lookup_tables: KQ_LOOKUP_TABLES, u: ActuatorOutput, wing_transform: FloatScalar, tail_transform: FloatScalar) -> Moments:
    table_names = ['tail_left', 'tail_right', 'left_out', 'left_2', 'left_3', 'left_4', 'left_5', 'left_6_in', 'right_7_in', 'right_8', 'right_9', 'right_10', 'right_11', 'right_12_out']
    results = []
    '''
    with mp.Pool(14) as p:
        results = p.map(ft.partial(Kq_interpolation, u=u, Kq_lookup_tables=Kq_lookup_tables, wing_transform=wing_transform, tail_transform=tail_transform), table_names)
    '''
    for table_name in table_names:
        results.append(Kq_interpolation(table_name, u, Kq_lookup_tables, wing_transform, tail_transform))
        
    Kq_tail_left_transformed = results[0]
    Kq_tail_right_transformed = results[1]
    Kq_left_out_transformed = results[2]
    Kq_left_2_transformed = results[3]
    Kq_left_3_transformed = results[4]
    Kq_left_4_transformed = results[5]
    Kq_left_5_transformed = results[6]
    Kq_left_6_in_transformed = results[7]
    Kq_right_7_in_transformed = results[8]
    Kq_right_8_transformed = results[9]
    Kq_right_9_transformed = results[10]
    Kq_right_10_transformed = results[11]
    Kq_right_11_transformed = results[12]
    Kq_right_12_out_transformed = results[13]

    return Moments(
        Kq_tail_left_transformed[0] + Kq_tail_right_transformed[0] + Kq_left_out_transformed[0] + Kq_left_2_transformed[0] + Kq_left_3_transformed[0] + 
        Kq_left_4_transformed[0] + Kq_left_5_transformed[0] + Kq_left_6_in_transformed[0] + Kq_right_7_in_transformed[0] + Kq_right_8_transformed[0] + 
        Kq_right_9_transformed[0] + Kq_right_10_transformed[0] + Kq_right_11_transformed[0] + Kq_right_12_out_transformed[0],
        Kq_tail_left_transformed[1] + Kq_tail_right_transformed[1] + Kq_left_out_transformed[1] + Kq_left_2_transformed[1] + Kq_left_3_transformed[1] + 
        Kq_left_4_transformed[1] + Kq_left_5_transformed[1] + Kq_left_6_in_transformed[1] + Kq_right_7_in_transformed[1] + Kq_right_8_transformed[1] + 
        Kq_right_9_transformed[1] + Kq_right_10_transformed[1] + Kq_right_11_transformed[1] + Kq_right_12_out_transformed[1],
        Kq_tail_left_transformed[2] + Kq_tail_right_transformed[2] + Kq_left_out_transformed[2] + Kq_left_2_transformed[2] + Kq_left_3_transformed[2] + 
        Kq_left_4_transformed[2] + Kq_left_5_transformed[2] + Kq_left_6_in_transformed[2] + Kq_right_7_in_transformed[2] + Kq_right_8_transformed[2] + 
        Kq_right_9_transformed[2] + Kq_right_10_transformed[2] + Kq_right_11_transformed[2] + Kq_right_12_out_transformed[2]
    )



class MavrikAero_Parallel(MavrikAero):
    def __init__(self,
                Cx_lookup_tables: CX_LOOKUP_TABLES,
                Cy_lookup_tables: CY_LOOKUP_TABLES,
                Cz_lookup_tables: CZ_LOOKUP_TABLES,
                Kq_lookup_tables: KQ_LOOKUP_TABLES,
                Ct_lookup_tables: CT_LOOKUP_TABLES,
                Cl_lookup_tables: CL_LOOKUP_TABLES,
                Cm_lookup_tables: CM_LOOKUP_TABLES,
                Cn_lookup_tables: CN_LOOKUP_TABLES,
                rpm_transforms: RPM_TRANSFORMS
    ):
        self.Cx_lookup_tables = Cx_lookup_tables
        self.Cy_lookup_tables = Cy_lookup_tables
        self.Cz_lookup_tables = Cz_lookup_tables
        self.Kq_lookup_tables = Kq_lookup_tables
        self.Ct_lookup_tables = Ct_lookup_tables
        self.Cl_lookup_tables = Cl_lookup_tables
        self.Cm_lookup_tables = Cm_lookup_tables
        self.Cn_lookup_tables = Cn_lookup_tables

        self.rpm_transforms = rpm_transforms
 

    @classmethod
    def create(cls, mavrik_setup: MavrikSetup):
        Cx_lookup_tables = get_Cx_table(mavrik_setup)
        Cy_lookup_tables = get_Cy_table(mavrik_setup)
        Cz_lookup_tables = get_Cz_table(mavrik_setup)
        Kq_lookup_tables = get_Kq_table(mavrik_setup)
        Ct_lookup_tables, rpm_transforms = get_Ct_table(mavrik_setup)
        Cl_lookup_tables = get_Cl_table(mavrik_setup)
        Cm_lookup_tables = get_Cm_table(mavrik_setup)
        Cn_lookup_tables = get_Cn_table(mavrik_setup)
        return MavrikAero_Parallel(Cx_lookup_tables, Cy_lookup_tables, Cz_lookup_tables, Kq_lookup_tables, Ct_lookup_tables, Cl_lookup_tables, Cm_lookup_tables, Cn_lookup_tables, rpm_transforms)
 
    def lookup_interpolation(self, table_name, actuator_outputs, wing_transform, tail_transform):
        if table_name == 'Ct':
            F0, M0 = Ct(self.Ct_lookup_tables, self.rpm_transforms, actuator_outputs, wing_transform, tail_transform)
            '''
            for key, value in F0._asdict().items():
                if np.isnan(value).any():
                    raise ValueError(f"NaN detected in actuator outputs {key=}: {value}")
                
            for key, value in M0._asdict().items():
                if np.isnan(value).any():
                    raise ValueError(f"NaN detected in actuator outputs {key=}: {value}")
            '''
            return (F0, M0)
        elif table_name == 'Cx':
            F1 = Cx(self.Cx_lookup_tables, actuator_outputs, wing_transform, tail_transform)
            '''
            for key, value in F1._asdict().items():
                if np.isnan(value).any():
                    raise ValueError(f"NaN detected in actuator outputs {key=}: {value}")
            ''' 
            return F1
        elif table_name == 'Cy':
            F2 = Cy(self.Cy_lookup_tables, actuator_outputs, wing_transform, tail_transform)
            '''
            for key, value in F2._asdict().items():
                if np.isnan(value).any():
                    raise ValueError(f"NaN detected in actuator outputs {key=}: {value}")
            '''
            return F2
        elif table_name == 'Cz':
            F3 = Cz(self.Cz_lookup_tables, actuator_outputs, wing_transform, tail_transform)
            '''
            for key, value in F3._asdict().items():
                if np.isnan(value).any():
                    raise ValueError(f"NaN detected in actuator outputs {key=}: {value}")
            '''
            return F3
        elif table_name == 'L':
            M1 = L(self.Cl_lookup_tables, actuator_outputs, wing_transform, tail_transform)
            '''
            for key, value in M1._asdict().items():
                if np.isnan(value).any():
                    raise ValueError(f"NaN detected in actuator outputs {key=}: {value}")
            '''
            return M1
        elif table_name == 'M':
            M2 = M(self.Cm_lookup_tables, actuator_outputs, wing_transform, tail_transform)
            '''
            for key, value in M2._asdict().items():
                if np.isnan(value).any():
                    raise ValueError(f"NaN detected in actuator outputs {key=}: {value}")
            '''
            return M2
        elif table_name == 'N':
            M3 = N(self.Cn_lookup_tables, actuator_outputs, wing_transform, tail_transform)
            '''
            for key, value in M3._asdict().items():
                if np.isnan(value).any():
                    raise ValueError(f"NaN detected in actuator outputs {key=}: {value}")
            '''
            return M3
        elif table_name == 'Kq':
            M5 = Kq(self.Kq_lookup_tables, actuator_outputs, wing_transform, tail_transform)
            '''
            for key, value in M5._asdict().items():
                if np.isnan(value).any():
                    raise ValueError(f"NaN detected in actuator outputs {key=}: {value}")
            '''
            return M5
        else:
            raise ValueError(f"Invalid table name: {table_name}")


    
    def __call__(self, state: StateVariables, control: ControlInputs) -> Tuple[Forces, Moments]:
        # Calculate forces and moments using Mavrik Aero model
        # Transform body frame velocities (u, v, w) to inertial frame velocities (Vx, Vy, Vz)
        R = euler_to_dcm(state.roll, state.pitch, state.yaw)
         
        # Body frame velocities
        body_velocities = np.array([state.VXe, state.VYe, state.VZe])
        #print(body_velocities)
        # Inertial frame velocities
        inertial_velocities = R @ body_velocities
        u, v, w = inertial_velocities
        #print(f"{inertial_velocities=} vs. {state.u=}, {state.v=}, {state.w=}")
        #print(f"beta_from_inertial_velocity={np.arctan2(v, np.sqrt(u**2 + w**2))} vs. beta_from_state_vb={np.arctan2(state.v, np.sqrt(state.u**2 + state.w**2))}")

        actuator_input_state = ActuatorInutState(
            U = np.sqrt(u**2 + v**2 + w**2),
            alpha = np.arctan2(w, u),
            beta = np.arctan2(v, np.sqrt(u**2 + w**2)),
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
        #print(f"{actuator_outputs=}")
        wing_transform = np.array([[np.cos(actuator_outputs.wing_tilt), 0, np.sin(actuator_outputs.wing_tilt)], [0, 1, 0], [-np.sin(actuator_outputs.wing_tilt), 0., np.cos(actuator_outputs.wing_tilt)]]);
        tail_transform = np.array([[np.cos(actuator_outputs.tail_tilt), 0, np.sin(actuator_outputs.tail_tilt)], [0, 1, 0], [-np.sin(actuator_outputs.tail_tilt), 0., np.cos(actuator_outputs.tail_tilt)]])

        table_names = ['Ct', 'Cx', 'Cy', 'Cz', 'L', 'M', 'N', 'Kq']
        results = []
        ''' 
        with mp.Pool(8) as p:
            results = p.map(ft.partial(self.lookup_interpolation, actuator_outputs=actuator_outputs, wing_transform=wing_transform, tail_transform=tail_transform), table_names) 
        '''
        for table_name in table_names:
            results.append(self.lookup_interpolation(table_name, actuator_outputs, wing_transform, tail_transform))
        F0 = results[0][0]
        M0 = results[0][1]
        F1 = results[1]
        F2 = results[2]
        F3 = results[3]
        M1 = results[4]
        M2 = results[5]
        M3 = results[6]
        M5 = results[7]
        
        Fx = F0.Fx + F1.Fx + F2.Fx + F3.Fx
        Fy = F0.Fy + F1.Fy + F2.Fy + F3.Fy
        Fz = F0.Fz + F1.Fz + F2.Fz + F3.Fz

        forces = Forces(Fx, Fy, Fz)
        #moments_by_forces = np.cross(np.array([state.X, state.Y, state.Z]), np.array([forces.Fx, forces.Fy, forces.Fz]))
        
        moments = Moments(M0.L + M1.L + M2.L + M3.L + M5.L, # + moments_by_forces[0], 
                          M0.M + M1.M + M2.M + M3.M + M5.M, # + moments_by_forces[1], 
                          M0.N + M1.N + M2.N + M3.N + M5.N, # + moments_by_forces[2]
                          )

        return forces, moments, actuator_outputs
     



if __name__ == "__main__":
    # Example usage of MavrikAero class

    # Initialize MavrikSetup with appropriate values
    mavrik_setup = MavrikSetup(file_path="/Users/weichaozhou/Workspace/Mavrik_JAX/jax_mavrik/aero_export.mat")

    # Initialize MavrikAero with mass and setup
    mavrik_aero = MavrikAero(mavrik_setup=mavrik_setup)

    # Define initial state variables
    state = StateVariables(
        u=29.9269, v=0.0, w=2.0927,
        Xe=0.0, Ye=0.0, Ze=0.0,
        roll=0.0, pitch=0.069813, yaw=0.0,
        VXe=30.0, VYe=0.0, VZe=0.0,
        p=0.0, q=0.0, r=0.0,
        Fx=0.0, Fy=0.0, Fz=0.0,
        L=0.0, M=0.0, N=0.0
    )

    # Define control inputs
    control = ControlInputs(
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

    # Calculate forces and moments
    forces, moments, actuator_outputs = mavrik_aero(state, control)

    # Print the results
    print("Forces:", forces)
    print("Moments:", moments)
    print("Actuator Outputs:", actuator_outputs)