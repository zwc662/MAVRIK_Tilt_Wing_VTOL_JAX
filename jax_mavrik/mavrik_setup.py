from jax_mavrik.src.aero_export import AeroExport 
import numpy as np 
 


class MavrikSetup:
    def __init__(self, file_path: str = 'aero_export.mat'):
        # Constants with fixed values from the .m file
        # Constants with fixed values from the .m file
        self.S = 0.5744
        self.c = 0.2032
        self.b = 2.8270
        self.rho = 1.225
        self.tail_prop_d4 = 0.005059318992632
        self.wing_prop_d4 = 0.021071715921
        self.tail_prop_d5 = 0.001349320375335
        self.wing_prop_d5 = 0.008028323765901
        self.C1 = 0.25
        self.C2 = 2.777777777777778e-4
        self.mass = 25
        self.inertia = np.diag([4.8518, 6.0388, 9.4731])
        self.U = 30
        self.pqr_init = np.array([0, 0, 0])
        euler_angle = 4 * np.pi / 180
        self.euler_init = np.array([0, euler_angle, 0])
        self.uvw_init = np.array([
            self.U * np.cos(self.euler_init[1]),
            0,
            self.U * np.sin(self.euler_init[1])
        ])
        self.xyz_init = np.array([0, 0, 0])
        self.aero_center = np.array([0.3353, 0, 0.0508])
        self.CG = np.array([0.3353, 0, 0.0508])
        self.position = self.aero_center - self.CG

        # Positions and transformations with negations for x and z
        self.tail_left_pos = np.array([1.26, -0.4506, 0.1295])
        self.RPM_tail_left_trans = np.array([-self.tail_left_pos[0] + self.CG[0], 
                                            self.tail_left_pos[1] - self.CG[1], 
                                            -self.tail_left_pos[2] + self.CG[2]])

        self.tail_right_pos = np.array([1.26, 0.4506, 0.1295])
        self.RPM_tail_right_trans = np.array([-self.tail_right_pos[0] + self.CG[0], 
                                            self.tail_right_pos[1] - self.CG[1], 
                                            -self.tail_right_pos[2] + self.CG[2]])

        self.left_out1_pos = np.array([0.5443, -1.4092, 0.04064])
        self.RPM_left_out1_trans = np.array([-self.left_out1_pos[0] + self.CG[0], 
                                            self.left_out1_pos[1] - self.CG[1], 
                                            -self.left_out1_pos[2] + self.CG[2]])

        self.left_2_pos = np.array([0.5159, -1.1963, 0.04064])
        self.RPM_left_2_trans = np.array([-self.left_2_pos[0] + self.CG[0], 
                                        self.left_2_pos[1] - self.CG[1], 
                                        -self.left_2_pos[2] + self.CG[2]])

        self.left_3_pos = np.array([0.5443, -0.9751, 0.04064])
        self.RPM_left_3_trans = np.array([-self.left_3_pos[0] + self.CG[0], 
                                        self.left_3_pos[1] - self.CG[1], 
                                        -self.left_3_pos[2] + self.CG[2]])

        # Repeat similarly for other transformations, applying the negations where necessary

        # For example:
        self.right_12_out_pos = np.array([0.5443, 1.4092, 0.04064])
        self.RPM_right_12_out_trans = np.array([-self.right_12_out_pos[0] + self.CG[0], 
                                                self.right_12_out_pos[1] - self.CG[1], 
                                                -self.right_12_out_pos[2] + self.CG[2]])
        
        
        
        aero = AeroExport.load_mat(file_path)

        CX_mat = aero.data['CX']

        self.CX_aileron_wing_val = CX_mat['CXaileron_wing.value']
        self.CX_aileron_wing_1 = CX_mat['CXaileron_wing.axes1.value']
        self.CX_aileron_wing_2 = CX_mat['CXaileron_wing.axes2.value']
        self.CX_aileron_wing_3 = CX_mat['CXaileron_wing.axes3.value']
        self.CX_aileron_wing_4 = CX_mat['CXaileron_wing.axes4.value']
        self.CX_aileron_wing_5 = CX_mat['CXaileron_wing.axes5.value']
        self.CX_aileron_wing_6 = CX_mat['CXaileron_wing.axes6.value']
        self.CX_aileron_wing_7 = CX_mat['CXaileron_wing.axes7.value']

        self.CX_elevator_tail_val = CX_mat['CXelevator_tail.value']
        self.CX_elevator_tail_1 = CX_mat['CXelevator_tail.axes1.value']
        self.CX_elevator_tail_2 = CX_mat['CXelevator_tail.axes2.value']
        self.CX_elevator_tail_3 = CX_mat['CXelevator_tail.axes3.value']
        self.CX_elevator_tail_4 = CX_mat['CXelevator_tail.axes4.value']
        self.CX_elevator_tail_5 = CX_mat['CXelevator_tail.axes5.value']
        self.CX_elevator_tail_6 = CX_mat['CXelevator_tail.axes6.value']
        self.CX_elevator_tail_7 = CX_mat['CXelevator_tail.axes7.value']

        # Flap Wing
        self.CX_flap_wing_val = CX_mat['CXflap_wing.value']
        self.CX_flap_wing_1 = CX_mat['CXflap_wing.axes1.value']
        self.CX_flap_wing_2 = CX_mat['CXflap_wing.axes2.value']
        self.CX_flap_wing_3 = CX_mat['CXflap_wing.axes3.value']
        self.CX_flap_wing_4 = CX_mat['CXflap_wing.axes4.value']
        self.CX_flap_wing_5 = CX_mat['CXflap_wing.axes5.value']
        self.CX_flap_wing_6 = CX_mat['CXflap_wing.axes6.value']
        self.CX_flap_wing_7 = CX_mat['CXflap_wing.axes7.value']

        # Rudder Tail
        self.CX_rudder_tail_val = CX_mat['CXrudder_tail.value']
        self.CX_rudder_tail_1 = CX_mat['CXrudder_tail.axes1.value']
        self.CX_rudder_tail_2 = CX_mat['CXrudder_tail.axes2.value']
        self.CX_rudder_tail_3 = CX_mat['CXrudder_tail.axes3.value']
        self.CX_rudder_tail_4 = CX_mat['CXrudder_tail.axes4.value']
        self.CX_rudder_tail_5 = CX_mat['CXrudder_tail.axes5.value']
        self.CX_rudder_tail_6 = CX_mat['CXrudder_tail.axes6.value']
        self.CX_rudder_tail_7 = CX_mat['CXrudder_tail.axes7.value']

        # Tail
        self.CX_tail_val = CX_mat['CXtail.value']
        self.CX_tail_1 = CX_mat['CXtail.axes1.value']
        self.CX_tail_2 = CX_mat['CXtail.axes2.value']
        self.CX_tail_3 = CX_mat['CXtail.axes3.value']
        self.CX_tail_4 = CX_mat['CXtail.axes4.value']
        self.CX_tail_5 = CX_mat['CXtail.axes5.value']
        self.CX_tail_6 = CX_mat['CXtail.axes6.value']

        # Tail Damp p
        self.CX_tail_damp_p_val = CX_mat['CXtail_damp_p.value']
        self.CX_tail_damp_p_1 = CX_mat['CXtail_damp_p.axes1.value']
        self.CX_tail_damp_p_2 = CX_mat['CXtail_damp_p.axes2.value']
        self.CX_tail_damp_p_3 = CX_mat['CXtail_damp_p.axes3.value']
        self.CX_tail_damp_p_4 = CX_mat['CXtail_damp_p.axes4.value']
        self.CX_tail_damp_p_5 = CX_mat['CXtail_damp_p.axes5.value']
        self.CX_tail_damp_p_6 = CX_mat['CXtail_damp_p.axes6.value']

        # Tail Damp q
        self.CX_tail_damp_q_val = CX_mat['CXtail_damp_q.value']
        self.CX_tail_damp_q_1 = CX_mat['CXtail_damp_q.axes1.value']
        self.CX_tail_damp_q_2 = CX_mat['CXtail_damp_q.axes2.value']
        self.CX_tail_damp_q_3 = CX_mat['CXtail_damp_q.axes3.value']
        self.CX_tail_damp_q_4 = CX_mat['CXtail_damp_q.axes4.value']
        self.CX_tail_damp_q_5 = CX_mat['CXtail_damp_q.axes5.value']
        self.CX_tail_damp_q_6 = CX_mat['CXtail_damp_q.axes6.value']

        # Tail Damp r
        self.CX_tail_damp_r_val = CX_mat['CXtail_damp_r.value']
        self.CX_tail_damp_r_1 = CX_mat['CXtail_damp_r.axes1.value']
        self.CX_tail_damp_r_2 = CX_mat['CXtail_damp_r.axes2.value']
        self.CX_tail_damp_r_3 = CX_mat['CXtail_damp_r.axes3.value']
        self.CX_tail_damp_r_4 = CX_mat['CXtail_damp_r.axes4.value']
        self.CX_tail_damp_r_5 = CX_mat['CXtail_damp_r.axes5.value']
        self.CX_tail_damp_r_6 = CX_mat['CXtail_damp_r.axes6.value']

        # Wing
        self.CX_wing_val = CX_mat['CXwing.value']
        self.CX_wing_1 = CX_mat['CXwing.axes1.value']
        self.CX_wing_2 = CX_mat['CXwing.axes2.value']
        self.CX_wing_3 = CX_mat['CXwing.axes3.value']
        self.CX_wing_4 = CX_mat['CXwing.axes4.value']
        self.CX_wing_5 = CX_mat['CXwing.axes5.value']
        self.CX_wing_6 = CX_mat['CXwing.axes6.value']

        # Wing Damp p
        self.CX_wing_damp_p_val = CX_mat['CXwing_damp_p.value']
        self.CX_wing_damp_p_1 = CX_mat['CXwing_damp_p.axes1.value']
        self.CX_wing_damp_p_2 = CX_mat['CXwing_damp_p.axes2.value']
        self.CX_wing_damp_p_3 = CX_mat['CXwing_damp_p.axes3.value']
        self.CX_wing_damp_p_4 = CX_mat['CXwing_damp_p.axes4.value']
        self.CX_wing_damp_p_5 = CX_mat['CXwing_damp_p.axes5.value']
        self.CX_wing_damp_p_6 = CX_mat['CXwing_damp_p.axes6.value']

        # Wing Damp q
        self.CX_wing_damp_q_val = CX_mat['CXwing_damp_q.value']
        self.CX_wing_damp_q_1 = CX_mat['CXwing_damp_q.axes1.value']
        self.CX_wing_damp_q_2 = CX_mat['CXwing_damp_q.axes2.value']
        self.CX_wing_damp_q_3 = CX_mat['CXwing_damp_q.axes3.value']
        self.CX_wing_damp_q_4 = CX_mat['CXwing_damp_q.axes4.value']
        self.CX_wing_damp_q_5 = CX_mat['CXwing_damp_q.axes5.value']
        self.CX_wing_damp_q_6 = CX_mat['CXwing_damp_q.axes6.value']

        # Wing Damp r
        self.CX_wing_damp_r_val = CX_mat['CXwing_damp_r.value']
        self.CX_wing_damp_r_1 = CX_mat['CXwing_damp_r.axes1.value']
        self.CX_wing_damp_r_2 = CX_mat['CXwing_damp_r.axes2.value']
        self.CX_wing_damp_r_3 = CX_mat['CXwing_damp_r.axes3.value']
        self.CX_wing_damp_r_4 = CX_mat['CXwing_damp_r.axes4.value']
        self.CX_wing_damp_r_5 = CX_mat['CXwing_damp_r.axes5.value']
        self.CX_wing_damp_r_6 = CX_mat['CXwing_damp_r.axes6.value']

        # Hover Fuse
        self.CX_hover_fuse_val = CX_mat['CXhover_fuse.value']
        self.CX_hover_fuse_1 = CX_mat['CXhover_fuse.axes1.value']
        self.CX_hover_fuse_2 = CX_mat['CXhover_fuse.axes2.value']
        self.CX_hover_fuse_3 = CX_mat['CXhover_fuse.axes3.value']


        CY_mat = aero.data['CY']

        self.CY_aileron_wing_val = CY_mat['CYaileron_wing.value']
        self.CY_aileron_wing_1 = CY_mat['CYaileron_wing.axes1.value']
        self.CY_aileron_wing_2 = CY_mat['CYaileron_wing.axes2.value']
        self.CY_aileron_wing_3 = CY_mat['CYaileron_wing.axes3.value']
        self.CY_aileron_wing_4 = CY_mat['CYaileron_wing.axes4.value']
        self.CY_aileron_wing_5 = CY_mat['CYaileron_wing.axes5.value']
        self.CY_aileron_wing_6 = CY_mat['CYaileron_wing.axes6.value']
        self.CY_aileron_wing_7 = CY_mat['CYaileron_wing.axes7.value']

        self.CY_elevator_tail_val = CY_mat['CYelevator_tail.value']
        self.CY_elevator_tail_1 = CY_mat['CYelevator_tail.axes1.value']
        self.CY_elevator_tail_2 = CY_mat['CYelevator_tail.axes2.value']
        self.CY_elevator_tail_3 = CY_mat['CYelevator_tail.axes3.value']
        self.CY_elevator_tail_4 = CY_mat['CYelevator_tail.axes4.value']
        self.CY_elevator_tail_5 = CY_mat['CYelevator_tail.axes5.value']
        self.CY_elevator_tail_6 = CY_mat['CYelevator_tail.axes6.value']
        self.CY_elevator_tail_7 = CY_mat['CYelevator_tail.axes7.value']

        # Flap Wing
        self.CY_flap_wing_val = CY_mat['CYflap_wing.value']
        self.CY_flap_wing_1 = CY_mat['CYflap_wing.axes1.value']
        self.CY_flap_wing_2 = CY_mat['CYflap_wing.axes2.value']
        self.CY_flap_wing_3 = CY_mat['CYflap_wing.axes3.value']
        self.CY_flap_wing_4 = CY_mat['CYflap_wing.axes4.value']
        self.CY_flap_wing_5 = CY_mat['CYflap_wing.axes5.value']
        self.CY_flap_wing_6 = CY_mat['CYflap_wing.axes6.value']
        self.CY_flap_wing_7 = CY_mat['CYflap_wing.axes7.value']

        # Rudder Tail
        self.CY_rudder_tail_val = CY_mat['CYrudder_tail.value']
        self.CY_rudder_tail_1 = CY_mat['CYrudder_tail.axes1.value']
        self.CY_rudder_tail_2 = CY_mat['CYrudder_tail.axes2.value']
        self.CY_rudder_tail_3 = CY_mat['CYrudder_tail.axes3.value']
        self.CY_rudder_tail_4 = CY_mat['CYrudder_tail.axes4.value']
        self.CY_rudder_tail_5 = CY_mat['CYrudder_tail.axes5.value']
        self.CY_rudder_tail_6 = CY_mat['CYrudder_tail.axes6.value']
        self.CY_rudder_tail_7 = CY_mat['CYrudder_tail.axes7.value']

        # Tail
        self.CY_tail_val = CY_mat['CYtail.value']
        self.CY_tail_1 = CY_mat['CYtail.axes1.value']
        self.CY_tail_2 = CY_mat['CYtail.axes2.value']
        self.CY_tail_3 = CY_mat['CYtail.axes3.value']
        self.CY_tail_4 = CY_mat['CYtail.axes4.value']
        self.CY_tail_5 = CY_mat['CYtail.axes5.value']
        self.CY_tail_6 = CY_mat['CYtail.axes6.value']

        # Tail Damp p
        self.CY_tail_damp_p_val = CY_mat['CYtail_damp_p.value']
        self.CY_tail_damp_p_1 = CY_mat['CYtail_damp_p.axes1.value']
        self.CY_tail_damp_p_2 = CY_mat['CYtail_damp_p.axes2.value']
        self.CY_tail_damp_p_3 = CY_mat['CYtail_damp_p.axes3.value']
        self.CY_tail_damp_p_4 = CY_mat['CYtail_damp_p.axes4.value']
        self.CY_tail_damp_p_5 = CY_mat['CYtail_damp_p.axes5.value']
        self.CY_tail_damp_p_6 = CY_mat['CYtail_damp_p.axes6.value']

        # Tail Damp q
        self.CY_tail_damp_q_val = CY_mat['CYtail_damp_q.value']
        self.CY_tail_damp_q_1 = CY_mat['CYtail_damp_q.axes1.value']
        self.CY_tail_damp_q_2 = CY_mat['CYtail_damp_q.axes2.value']
        self.CY_tail_damp_q_3 = CY_mat['CYtail_damp_q.axes3.value']
        self.CY_tail_damp_q_4 = CY_mat['CYtail_damp_q.axes4.value']
        self.CY_tail_damp_q_5 = CY_mat['CYtail_damp_q.axes5.value']
        self.CY_tail_damp_q_6 = CY_mat['CYtail_damp_q.axes6.value']

        # Tail Damp r
        self.CY_tail_damp_r_val = CY_mat['CYtail_damp_r.value']
        self.CY_tail_damp_r_1 = CY_mat['CYtail_damp_r.axes1.value']
        self.CY_tail_damp_r_2 = CY_mat['CYtail_damp_r.axes2.value']
        self.CY_tail_damp_r_3 = CY_mat['CYtail_damp_r.axes3.value']
        self.CY_tail_damp_r_4 = CY_mat['CYtail_damp_r.axes4.value']
        self.CY_tail_damp_r_5 = CY_mat['CYtail_damp_r.axes5.value']
        self.CY_tail_damp_r_6 = CY_mat['CYtail_damp_r.axes6.value']

        # Wing
        self.CY_wing_val = CY_mat['CYwing.value']
        self.CY_wing_1 = CY_mat['CYwing.axes1.value']
        self.CY_wing_2 = CY_mat['CYwing.axes2.value']
        self.CY_wing_3 = CY_mat['CYwing.axes3.value']
        self.CY_wing_4 = CY_mat['CYwing.axes4.value']
        self.CY_wing_5 = CY_mat['CYwing.axes5.value']
        self.CY_wing_6 = CY_mat['CYwing.axes6.value']

        # Wing Damp p
        self.CY_wing_damp_p_val = CY_mat['CYwing_damp_p.value']
        self.CY_wing_damp_p_1 = CY_mat['CYwing_damp_p.axes1.value']
        self.CY_wing_damp_p_2 = CY_mat['CYwing_damp_p.axes2.value']
        self.CY_wing_damp_p_3 = CY_mat['CYwing_damp_p.axes3.value']
        self.CY_wing_damp_p_4 = CY_mat['CYwing_damp_p.axes4.value']
        self.CY_wing_damp_p_5 = CY_mat['CYwing_damp_p.axes5.value']
        self.CY_wing_damp_p_6 = CY_mat['CYwing_damp_p.axes6.value']

        # Wing Damp q
        self.CY_wing_damp_q_val = CY_mat['CYwing_damp_q.value']
        self.CY_wing_damp_q_1 = CY_mat['CYwing_damp_q.axes1.value']
        self.CY_wing_damp_q_2 = CY_mat['CYwing_damp_q.axes2.value']
        self.CY_wing_damp_q_3 = CY_mat['CYwing_damp_q.axes3.value']
        self.CY_wing_damp_q_4 = CY_mat['CYwing_damp_q.axes4.value']
        self.CY_wing_damp_q_5 = CY_mat['CYwing_damp_q.axes5.value']
        self.CY_wing_damp_q_6 = CY_mat['CYwing_damp_q.axes6.value']

        # Wing Damp r
        self.CY_wing_damp_r_val = CY_mat['CYwing_damp_r.value']
        self.CY_wing_damp_r_1 = CY_mat['CYwing_damp_r.axes1.value']
        self.CY_wing_damp_r_2 = CY_mat['CYwing_damp_r.axes2.value']
        self.CY_wing_damp_r_3 = CY_mat['CYwing_damp_r.axes3.value']
        self.CY_wing_damp_r_4 = CY_mat['CYwing_damp_r.axes4.value']
        self.CY_wing_damp_r_5 = CY_mat['CYwing_damp_r.axes5.value']
        self.CY_wing_damp_r_6 = CY_mat['CYwing_damp_r.axes6.value']

        # Hover Fuse
        self.CY_hover_fuse_val = CY_mat['CYhover_fuse.value']
        self.CY_hover_fuse_1 = CY_mat['CYhover_fuse.axes1.value']
        self.CY_hover_fuse_2 = CY_mat['CYhover_fuse.axes2.value']
        self.CY_hover_fuse_3 = CY_mat['CYhover_fuse.axes3.value']


        # CZ Data
        CZ_mat = aero.data['CZ']

        # Aileron Wing
        self.CZ_aileron_wing_val = CZ_mat['CZaileron_wing.value']
        self.CZ_aileron_wing_1 = CZ_mat['CZaileron_wing.axes1.value']
        self.CZ_aileron_wing_2 = CZ_mat['CZaileron_wing.axes2.value']
        self.CZ_aileron_wing_3 = CZ_mat['CZaileron_wing.axes3.value']
        self.CZ_aileron_wing_4 = CZ_mat['CZaileron_wing.axes4.value']
        self.CZ_aileron_wing_5 = CZ_mat['CZaileron_wing.axes5.value']
        self.CZ_aileron_wing_6 = CZ_mat['CZaileron_wing.axes6.value']
        self.CZ_aileron_wing_7 = CZ_mat['CZaileron_wing.axes7.value']

        # Elevator Tail
        self.CZ_elevator_tail_val = CZ_mat['CZelevator_tail.value']
        self.CZ_elevator_tail_1 = CZ_mat['CZelevator_tail.axes1.value']
        self.CZ_elevator_tail_2 = CZ_mat['CZelevator_tail.axes2.value']
        self.CZ_elevator_tail_3 = CZ_mat['CZelevator_tail.axes3.value']
        self.CZ_elevator_tail_4 = CZ_mat['CZelevator_tail.axes4.value']
        self.CZ_elevator_tail_5 = CZ_mat['CZelevator_tail.axes5.value']
        self.CZ_elevator_tail_6 = CZ_mat['CZelevator_tail.axes6.value']
        self.CZ_elevator_tail_7 = CZ_mat['CZelevator_tail.axes7.value']

        # Flap Wing
        self.CZ_flap_wing_val = CZ_mat['CZflap_wing.value']
        self.CZ_flap_wing_1 = CZ_mat['CZflap_wing.axes1.value']
        self.CZ_flap_wing_2 = CZ_mat['CZflap_wing.axes2.value']
        self.CZ_flap_wing_3 = CZ_mat['CZflap_wing.axes3.value']
        self.CZ_flap_wing_4 = CZ_mat['CZflap_wing.axes4.value']
        self.CZ_flap_wing_5 = CZ_mat['CZflap_wing.axes5.value']
        self.CZ_flap_wing_6 = CZ_mat['CZflap_wing.axes6.value']
        self.CZ_flap_wing_7 = CZ_mat['CZflap_wing.axes7.value']

        # Rudder Tail
        self.CZ_rudder_tail_val = CZ_mat['CZrudder_tail.value']
        self.CZ_rudder_tail_1 = CZ_mat['CZrudder_tail.axes1.value']
        self.CZ_rudder_tail_2 = CZ_mat['CZrudder_tail.axes2.value']
        self.CZ_rudder_tail_3 = CZ_mat['CZrudder_tail.axes3.value']
        self.CZ_rudder_tail_4 = CZ_mat['CZrudder_tail.axes4.value']
        self.CZ_rudder_tail_5 = CZ_mat['CZrudder_tail.axes5.value']
        self.CZ_rudder_tail_6 = CZ_mat['CZrudder_tail.axes6.value']
        self.CZ_rudder_tail_7 = CZ_mat['CZrudder_tail.axes7.value']

        # Tail
        self.CZ_tail_val = CZ_mat['CZtail.value']
        self.CZ_tail_1 = CZ_mat['CZtail.axes1.value']
        self.CZ_tail_2 = CZ_mat['CZtail.axes2.value']
        self.CZ_tail_3 = CZ_mat['CZtail.axes3.value']
        self.CZ_tail_4 = CZ_mat['CZtail.axes4.value']
        self.CZ_tail_5 = CZ_mat['CZtail.axes5.value']
        self.CZ_tail_6 = CZ_mat['CZtail.axes6.value']

        # Tail Damp p
        self.CZ_tail_damp_p_val = CZ_mat['CZtail_damp_p.value']
        self.CZ_tail_damp_p_1 = CZ_mat['CZtail_damp_p.axes1.value']
        self.CZ_tail_damp_p_2 = CZ_mat['CZtail_damp_p.axes2.value']
        self.CZ_tail_damp_p_3 = CZ_mat['CZtail_damp_p.axes3.value']
        self.CZ_tail_damp_p_4 = CZ_mat['CZtail_damp_p.axes4.value']
        self.CZ_tail_damp_p_5 = CZ_mat['CZtail_damp_p.axes5.value']
        self.CZ_tail_damp_p_6 = CZ_mat['CZtail_damp_p.axes6.value']

        # Tail Damp q
        self.CZ_tail_damp_q_val = CZ_mat['CZtail_damp_q.value']
        self.CZ_tail_damp_q_1 = CZ_mat['CZtail_damp_q.axes1.value']
        self.CZ_tail_damp_q_2 = CZ_mat['CZtail_damp_q.axes2.value']
        self.CZ_tail_damp_q_3 = CZ_mat['CZtail_damp_q.axes3.value']
        self.CZ_tail_damp_q_4 = CZ_mat['CZtail_damp_q.axes4.value']
        self.CZ_tail_damp_q_5 = CZ_mat['CZtail_damp_q.axes5.value']
        self.CZ_tail_damp_q_6 = CZ_mat['CZtail_damp_q.axes6.value']

        # Tail Damp r
        self.CZ_tail_damp_r_val = CZ_mat['CZtail_damp_r.value']
        self.CZ_tail_damp_r_1 = CZ_mat['CZtail_damp_r.axes1.value']
        self.CZ_tail_damp_r_2 = CZ_mat['CZtail_damp_r.axes2.value']
        self.CZ_tail_damp_r_3 = CZ_mat['CZtail_damp_r.axes3.value']
        self.CZ_tail_damp_r_4 = CZ_mat['CZtail_damp_r.axes4.value']
        self.CZ_tail_damp_r_5 = CZ_mat['CZtail_damp_r.axes5.value']
        self.CZ_tail_damp_r_6 = CZ_mat['CZtail_damp_r.axes6.value']

        # Wing
        self.CZ_wing_val = CZ_mat['CZwing.value']
        self.CZ_wing_1 = CZ_mat['CZwing.axes1.value']
        self.CZ_wing_2 = CZ_mat['CZwing.axes2.value']
        self.CZ_wing_3 = CZ_mat['CZwing.axes3.value']
        self.CZ_wing_4 = CZ_mat['CZwing.axes4.value']
        self.CZ_wing_5 = CZ_mat['CZwing.axes5.value']
        self.CZ_wing_6 = CZ_mat['CZwing.axes6.value']

        # Wing Damp p
        self.CZ_wing_damp_p_val = CZ_mat['CZwing_damp_p.value']
        self.CZ_wing_damp_p_1 = CZ_mat['CZwing_damp_p.axes1.value']
        self.CZ_wing_damp_p_2 = CZ_mat['CZwing_damp_p.axes2.value']
        self.CZ_wing_damp_p_3 = CZ_mat['CZwing_damp_p.axes3.value']
        self.CZ_wing_damp_p_4 = CZ_mat['CZwing_damp_p.axes4.value']
        self.CZ_wing_damp_p_5 = CZ_mat['CZwing_damp_p.axes5.value']
        self.CZ_wing_damp_p_6 = CZ_mat['CZwing_damp_p.axes6.value']

        # Wing Damp q
        self.CZ_wing_damp_q_val = CZ_mat['CZwing_damp_q.value']
        self.CZ_wing_damp_q_1 = CZ_mat['CZwing_damp_q.axes1.value']
        self.CZ_wing_damp_q_2 = CZ_mat['CZwing_damp_q.axes2.value']
        self.CZ_wing_damp_q_3 = CZ_mat['CZwing_damp_q.axes3.value']
        self.CZ_wing_damp_q_4 = CZ_mat['CZwing_damp_q.axes4.value']
        self.CZ_wing_damp_q_5 = CZ_mat['CZwing_damp_q.axes5.value']
        self.CZ_wing_damp_q_6 = CZ_mat['CZwing_damp_q.axes6.value']

        # Wing Damp r
        self.CZ_wing_damp_r_val = CZ_mat['CZwing_damp_r.value']
        self.CZ_wing_damp_r_1 = CZ_mat['CZwing_damp_r.axes1.value']
        self.CZ_wing_damp_r_2 = CZ_mat['CZwing_damp_r.axes2.value']
        self.CZ_wing_damp_r_3 = CZ_mat['CZwing_damp_r.axes3.value']
        self.CZ_wing_damp_r_4 = CZ_mat['CZwing_damp_r.axes4.value']
        self.CZ_wing_damp_r_5 = CZ_mat['CZwing_damp_r.axes5.value']
        self.CZ_wing_damp_r_6 = CZ_mat['CZwing_damp_r.axes6.value']

        # Hover Fuse
        self.CZ_hover_fuse_val = CZ_mat['CZhover_fuse.value']
        self.CZ_hover_fuse_1 = CZ_mat['CZhover_fuse.axes1.value']
        self.CZ_hover_fuse_2 = CZ_mat['CZhover_fuse.axes2.value']
        self.CZ_hover_fuse_3 = CZ_mat['CZhover_fuse.axes3.value']

        # Cl Data
        Cl_mat = aero.data['Cl']

        # Aileron Wing
        self.Cl_aileron_wing_val = Cl_mat['Claileron_wing.value']
        self.Cl_aileron_wing_1 = Cl_mat['Claileron_wing.axes1.value']
        self.Cl_aileron_wing_2 = Cl_mat['Claileron_wing.axes2.value']
        self.Cl_aileron_wing_3 = Cl_mat['Claileron_wing.axes3.value']
        self.Cl_aileron_wing_4 = Cl_mat['Claileron_wing.axes4.value']
        self.Cl_aileron_wing_5 = Cl_mat['Claileron_wing.axes5.value']
        self.Cl_aileron_wing_6 = Cl_mat['Claileron_wing.axes6.value']
        self.Cl_aileron_wing_7 = Cl_mat['Claileron_wing.axes7.value']

        # Elevator Tail
        self.Cl_elevator_tail_val = Cl_mat['Clelevator_tail.value']
        self.Cl_elevator_tail_1 = Cl_mat['Clelevator_tail.axes1.value']
        self.Cl_elevator_tail_2 = Cl_mat['Clelevator_tail.axes2.value']
        self.Cl_elevator_tail_3 = Cl_mat['Clelevator_tail.axes3.value']
        self.Cl_elevator_tail_4 = Cl_mat['Clelevator_tail.axes4.value']
        self.Cl_elevator_tail_5 = Cl_mat['Clelevator_tail.axes5.value']
        self.Cl_elevator_tail_6 = Cl_mat['Clelevator_tail.axes6.value']
        self.Cl_elevator_tail_7 = Cl_mat['Clelevator_tail.axes7.value']

        # Flap Wing
        self.Cl_flap_wing_val = Cl_mat['Clflap_wing.value']
        self.Cl_flap_wing_1 = Cl_mat['Clflap_wing.axes1.value']
        self.Cl_flap_wing_2 = Cl_mat['Clflap_wing.axes2.value']
        self.Cl_flap_wing_3 = Cl_mat['Clflap_wing.axes3.value']
        self.Cl_flap_wing_4 = Cl_mat['Clflap_wing.axes4.value']
        self.Cl_flap_wing_5 = Cl_mat['Clflap_wing.axes5.value']
        self.Cl_flap_wing_6 = Cl_mat['Clflap_wing.axes6.value']
        self.Cl_flap_wing_7 = Cl_mat['Clflap_wing.axes7.value']

        # Rudder Tail
        self.Cl_rudder_tail_val = Cl_mat['Clrudder_tail.value']
        self.Cl_rudder_tail_1 = Cl_mat['Clrudder_tail.axes1.value']
        self.Cl_rudder_tail_2 = Cl_mat['Clrudder_tail.axes2.value']
        self.Cl_rudder_tail_3 = Cl_mat['Clrudder_tail.axes3.value']
        self.Cl_rudder_tail_4 = Cl_mat['Clrudder_tail.axes4.value']
        self.Cl_rudder_tail_5 = Cl_mat['Clrudder_tail.axes5.value']
        self.Cl_rudder_tail_6 = Cl_mat['Clrudder_tail.axes6.value']
        self.Cl_rudder_tail_7 = Cl_mat['Clrudder_tail.axes7.value']

        # Tail
        self.Cl_tail_val = Cl_mat['Cltail.value']
        self.Cl_tail_1 = Cl_mat['Cltail.axes1.value']
        self.Cl_tail_2 = Cl_mat['Cltail.axes2.value']
        self.Cl_tail_3 = Cl_mat['Cltail.axes3.value']
        self.Cl_tail_4 = Cl_mat['Cltail.axes4.value']
        self.Cl_tail_5 = Cl_mat['Cltail.axes5.value']
        self.Cl_tail_6 = Cl_mat['Cltail.axes6.value']

        # Tail Damp p
        self.Cl_tail_damp_p_val = Cl_mat['Cltail_damp_p.value']
        self.Cl_tail_damp_p_1 = Cl_mat['Cltail_damp_p.axes1.value']
        self.Cl_tail_damp_p_2 = Cl_mat['Cltail_damp_p.axes2.value']
        self.Cl_tail_damp_p_3 = Cl_mat['Cltail_damp_p.axes3.value']
        self.Cl_tail_damp_p_4 = Cl_mat['Cltail_damp_p.axes4.value']
        self.Cl_tail_damp_p_5 = Cl_mat['Cltail_damp_p.axes5.value']
        self.Cl_tail_damp_p_6 = Cl_mat['Cltail_damp_p.axes6.value']

        # Tail Damp q
        self.Cl_tail_damp_q_val = Cl_mat['Cltail_damp_q.value']
        self.Cl_tail_damp_q_1 = Cl_mat['Cltail_damp_q.axes1.value']
        self.Cl_tail_damp_q_2 = Cl_mat['Cltail_damp_q.axes2.value']
        self.Cl_tail_damp_q_3 = Cl_mat['Cltail_damp_q.axes3.value']
        self.Cl_tail_damp_q_4 = Cl_mat['Cltail_damp_q.axes4.value']
        self.Cl_tail_damp_q_5 = Cl_mat['Cltail_damp_q.axes5.value']
        self.Cl_tail_damp_q_6 = Cl_mat['Cltail_damp_q.axes6.value']

        # Tail Damp r
        self.Cl_tail_damp_r_val = Cl_mat['Cltail_damp_r.value']
        self.Cl_tail_damp_r_1 = Cl_mat['Cltail_damp_r.axes1.value']
        self.Cl_tail_damp_r_2 = Cl_mat['Cltail_damp_r.axes2.value']
        self.Cl_tail_damp_r_3 = Cl_mat['Cltail_damp_r.axes3.value']
        self.Cl_tail_damp_r_4 = Cl_mat['Cltail_damp_r.axes4.value']
        self.Cl_tail_damp_r_5 = Cl_mat['Cltail_damp_r.axes5.value']
        self.Cl_tail_damp_r_6 = Cl_mat['Cltail_damp_r.axes6.value']

        # Wing
        self.Cl_wing_val = Cl_mat['Clwing.value']
        self.Cl_wing_1 = Cl_mat['Clwing.axes1.value']
        self.Cl_wing_2 = Cl_mat['Clwing.axes2.value']
        self.Cl_wing_3 = Cl_mat['Clwing.axes3.value']
        self.Cl_wing_4 = Cl_mat['Clwing.axes4.value']
        self.Cl_wing_5 = Cl_mat['Clwing.axes5.value']
        self.Cl_wing_6 = Cl_mat['Clwing.axes6.value']

        # Wing Damp p
        self.Cl_wing_damp_p_val = Cl_mat['Clwing_damp_p.value']
        self.Cl_wing_damp_p_1 = Cl_mat['Clwing_damp_p.axes1.value']
        self.Cl_wing_damp_p_2 = Cl_mat['Clwing_damp_p.axes2.value']
        self.Cl_wing_damp_p_3 = Cl_mat['Clwing_damp_p.axes3.value']
        self.Cl_wing_damp_p_4 = Cl_mat['Clwing_damp_p.axes4.value']
        self.Cl_wing_damp_p_5 = Cl_mat['Clwing_damp_p.axes5.value']
        self.Cl_wing_damp_p_6 = Cl_mat['Clwing_damp_p.axes6.value']

        # Wing Damp q
        self.Cl_wing_damp_q_val = Cl_mat['Clwing_damp_q.value']
        self.Cl_wing_damp_q_1 = Cl_mat['Clwing_damp_q.axes1.value']
        self.Cl_wing_damp_q_2 = Cl_mat['Clwing_damp_q.axes2.value']
        self.Cl_wing_damp_q_3 = Cl_mat['Clwing_damp_q.axes3.value']
        self.Cl_wing_damp_q_4 = Cl_mat['Clwing_damp_q.axes4.value']
        self.Cl_wing_damp_q_5 = Cl_mat['Clwing_damp_q.axes5.value']
        self.Cl_wing_damp_q_6 = Cl_mat['Clwing_damp_q.axes6.value']

        # Wing Damp r
        self.Cl_wing_damp_r_val = Cl_mat['Clwing_damp_r.value']
        self.Cl_wing_damp_r_1 = Cl_mat['Clwing_damp_r.axes1.value']
        self.Cl_wing_damp_r_2 = Cl_mat['Clwing_damp_r.axes2.value']
        self.Cl_wing_damp_r_3 = Cl_mat['Clwing_damp_r.axes3.value']
        self.Cl_wing_damp_r_4 = Cl_mat['Clwing_damp_r.axes4.value']
        self.Cl_wing_damp_r_5 = Cl_mat['Clwing_damp_r.axes5.value']
        self.Cl_wing_damp_r_6 = Cl_mat['Clwing_damp_r.axes6.value']

        # Hover Fuse
        self.Cl_hover_fuse_val = Cl_mat['Clhover_fuse.value']
        self.Cl_hover_fuse_1 = Cl_mat['Clhover_fuse.axes1.value']
        self.Cl_hover_fuse_2 = Cl_mat['Clhover_fuse.axes2.value']
        self.Cl_hover_fuse_3 = Cl_mat['Clhover_fuse.axes3.value']


        # Cm Data
        Cm_mat = aero.data['Cm']

        # Aileron Wing
        self.Cm_aileron_wing_val = Cm_mat['Cmaileron_wing.value']
        self.Cm_aileron_wing_1 = Cm_mat['Cmaileron_wing.axes1.value']
        self.Cm_aileron_wing_2 = Cm_mat['Cmaileron_wing.axes2.value']
        self.Cm_aileron_wing_3 = Cm_mat['Cmaileron_wing.axes3.value']
        self.Cm_aileron_wing_4 = Cm_mat['Cmaileron_wing.axes4.value']
        self.Cm_aileron_wing_5 = Cm_mat['Cmaileron_wing.axes5.value']
        self.Cm_aileron_wing_6 = Cm_mat['Cmaileron_wing.axes6.value']
        self.Cm_aileron_wing_7 = Cm_mat['Cmaileron_wing.axes7.value']

        # Elevator Tail
        self.Cm_elevator_tail_val = Cm_mat['Cmelevator_tail.value']
        self.Cm_elevator_tail_1 = Cm_mat['Cmelevator_tail.axes1.value']
        self.Cm_elevator_tail_2 = Cm_mat['Cmelevator_tail.axes2.value']
        self.Cm_elevator_tail_3 = Cm_mat['Cmelevator_tail.axes3.value']
        self.Cm_elevator_tail_4 = Cm_mat['Cmelevator_tail.axes4.value']
        self.Cm_elevator_tail_5 = Cm_mat['Cmelevator_tail.axes5.value']
        self.Cm_elevator_tail_6 = Cm_mat['Cmelevator_tail.axes6.value']
        self.Cm_elevator_tail_7 = Cm_mat['Cmelevator_tail.axes7.value']

        # Flap Wing
        self.Cm_flap_wing_val = Cm_mat['Cmflap_wing.value']
        self.Cm_flap_wing_1 = Cm_mat['Cmflap_wing.axes1.value']
        self.Cm_flap_wing_2 = Cm_mat['Cmflap_wing.axes2.value']
        self.Cm_flap_wing_3 = Cm_mat['Cmflap_wing.axes3.value']
        self.Cm_flap_wing_4 = Cm_mat['Cmflap_wing.axes4.value']
        self.Cm_flap_wing_5 = Cm_mat['Cmflap_wing.axes5.value']
        self.Cm_flap_wing_6 = Cm_mat['Cmflap_wing.axes6.value']
        self.Cm_flap_wing_7 = Cm_mat['Cmflap_wing.axes7.value']

        # Rudder Tail
        self.Cm_rudder_tail_val = Cm_mat['Cmrudder_tail.value']
        self.Cm_rudder_tail_1 = Cm_mat['Cmrudder_tail.axes1.value']
        self.Cm_rudder_tail_2 = Cm_mat['Cmrudder_tail.axes2.value']
        self.Cm_rudder_tail_3 = Cm_mat['Cmrudder_tail.axes3.value']
        self.Cm_rudder_tail_4 = Cm_mat['Cmrudder_tail.axes4.value']
        self.Cm_rudder_tail_5 = Cm_mat['Cmrudder_tail.axes5.value']
        self.Cm_rudder_tail_6 = Cm_mat['Cmrudder_tail.axes6.value']
        self.Cm_rudder_tail_7 = Cm_mat['Cmrudder_tail.axes7.value']

        # Tail
        self.Cm_tail_val = Cm_mat['Cmtail.value']
        self.Cm_tail_1 = Cm_mat['Cmtail.axes1.value']
        self.Cm_tail_2 = Cm_mat['Cmtail.axes2.value']
        self.Cm_tail_3 = Cm_mat['Cmtail.axes3.value']
        self.Cm_tail_4 = Cm_mat['Cmtail.axes4.value']
        self.Cm_tail_5 = Cm_mat['Cmtail.axes5.value']
        self.Cm_tail_6 = Cm_mat['Cmtail.axes6.value']

        # Tail Damp p
        self.Cm_tail_damp_p_val = Cm_mat['Cmtail_damp_p.value']
        self.Cm_tail_damp_p_1 = Cm_mat['Cmtail_damp_p.axes1.value']
        self.Cm_tail_damp_p_2 = Cm_mat['Cmtail_damp_p.axes2.value']
        self.Cm_tail_damp_p_3 = Cm_mat['Cmtail_damp_p.axes3.value']
        self.Cm_tail_damp_p_4 = Cm_mat['Cmtail_damp_p.axes4.value']
        self.Cm_tail_damp_p_5 = Cm_mat['Cmtail_damp_p.axes5.value']
        self.Cm_tail_damp_p_6 = Cm_mat['Cmtail_damp_p.axes6.value']

        # Tail Damp q
        self.Cm_tail_damp_q_val = Cm_mat['Cmtail_damp_q.value']
        self.Cm_tail_damp_q_1 = Cm_mat['Cmtail_damp_q.axes1.value']
        self.Cm_tail_damp_q_2 = Cm_mat['Cmtail_damp_q.axes2.value']
        self.Cm_tail_damp_q_3 = Cm_mat['Cmtail_damp_q.axes3.value']
        self.Cm_tail_damp_q_4 = Cm_mat['Cmtail_damp_q.axes4.value']
        self.Cm_tail_damp_q_5 = Cm_mat['Cmtail_damp_q.axes5.value']
        self.Cm_tail_damp_q_6 = Cm_mat['Cmtail_damp_q.axes6.value']

        # Tail Damp r
        self.Cm_tail_damp_r_val = Cm_mat['Cmtail_damp_r.value']
        self.Cm_tail_damp_r_1 = Cm_mat['Cmtail_damp_r.axes1.value']
        self.Cm_tail_damp_r_2 = Cm_mat['Cmtail_damp_r.axes2.value']
        self.Cm_tail_damp_r_3 = Cm_mat['Cmtail_damp_r.axes3.value']
        self.Cm_tail_damp_r_4 = Cm_mat['Cmtail_damp_r.axes4.value']
        self.Cm_tail_damp_r_5 = Cm_mat['Cmtail_damp_r.axes5.value']
        self.Cm_tail_damp_r_6 = Cm_mat['Cmtail_damp_r.axes6.value']

        # Wing
        self.Cm_wing_val = Cm_mat['Cmwing.value']
        self.Cm_wing_1 = Cm_mat['Cmwing.axes1.value']
        self.Cm_wing_2 = Cm_mat['Cmwing.axes2.value']
        self.Cm_wing_3 = Cm_mat['Cmwing.axes3.value']
        self.Cm_wing_4 = Cm_mat['Cmwing.axes4.value']
        self.Cm_wing_5 = Cm_mat['Cmwing.axes5.value']
        self.Cm_wing_6 = Cm_mat['Cmwing.axes6.value']

        # Wing Damp p
        self.Cm_wing_damp_p_val = Cm_mat['Cmwing_damp_p.value']
        self.Cm_wing_damp_p_1 = Cm_mat['Cmwing_damp_p.axes1.value']
        self.Cm_wing_damp_p_2 = Cm_mat['Cmwing_damp_p.axes2.value']
        self.Cm_wing_damp_p_3 = Cm_mat['Cmwing_damp_p.axes3.value']
        self.Cm_wing_damp_p_4 = Cm_mat['Cmwing_damp_p.axes4.value']
        self.Cm_wing_damp_p_5 = Cm_mat['Cmwing_damp_p.axes5.value']
        self.Cm_wing_damp_p_6 = Cm_mat['Cmwing_damp_p.axes6.value']

        # Wing Damp q
        self.Cm_wing_damp_q_val = Cm_mat['Cmwing_damp_q.value']
        self.Cm_wing_damp_q_1 = Cm_mat['Cmwing_damp_q.axes1.value']
        self.Cm_wing_damp_q_2 = Cm_mat['Cmwing_damp_q.axes2.value']
        self.Cm_wing_damp_q_3 = Cm_mat['Cmwing_damp_q.axes3.value']
        self.Cm_wing_damp_q_4 = Cm_mat['Cmwing_damp_q.axes4.value']
        self.Cm_wing_damp_q_5 = Cm_mat['Cmwing_damp_q.axes5.value']
        self.Cm_wing_damp_q_6 = Cm_mat['Cmwing_damp_q.axes6.value']

        # Wing Damp r
        self.Cm_wing_damp_r_val = Cm_mat['Cmwing_damp_r.value']
        self.Cm_wing_damp_r_1 = Cm_mat['Cmwing_damp_r.axes1.value']
        self.Cm_wing_damp_r_2 = Cm_mat['Cmwing_damp_r.axes2.value']
        self.Cm_wing_damp_r_3 = Cm_mat['Cmwing_damp_r.axes3.value']
        self.Cm_wing_damp_r_4 = Cm_mat['Cmwing_damp_r.axes4.value']
        self.Cm_wing_damp_r_5 = Cm_mat['Cmwing_damp_r.axes5.value']
        self.Cm_wing_damp_r_6 = Cm_mat['Cmwing_damp_r.axes6.value']

        # Hover Fuse
        self.Cm_hover_fuse_val = Cm_mat['Cmhover_fuse.value']
        self.Cm_hover_fuse_1 = Cm_mat['Cmhover_fuse.axes1.value']
        self.Cm_hover_fuse_2 = Cm_mat['Cmhover_fuse.axes2.value']
        self.Cm_hover_fuse_3 = Cm_mat['Cmhover_fuse.axes3.value']


        # Cn Data
        Cn_mat = aero.data['Cn']

        # Aileron Wing
        self.Cn_aileron_wing_val = Cn_mat['Cnaileron_wing.value']
        self.Cn_aileron_wing_1 = Cn_mat['Cnaileron_wing.axes1.value']
        self.Cn_aileron_wing_2 = Cn_mat['Cnaileron_wing.axes2.value']
        self.Cn_aileron_wing_3 = Cn_mat['Cnaileron_wing.axes3.value']
        self.Cn_aileron_wing_4 = Cn_mat['Cnaileron_wing.axes4.value']
        self.Cn_aileron_wing_5 = Cn_mat['Cnaileron_wing.axes5.value']
        self.Cn_aileron_wing_6 = Cn_mat['Cnaileron_wing.axes6.value']
        self.Cn_aileron_wing_7 = Cn_mat['Cnaileron_wing.axes7.value']

        # Elevator Tail
        self.Cn_elevator_tail_val = Cn_mat['Cnelevator_tail.value']
        self.Cn_elevator_tail_1 = Cn_mat['Cnelevator_tail.axes1.value']
        self.Cn_elevator_tail_2 = Cn_mat['Cnelevator_tail.axes2.value']
        self.Cn_elevator_tail_3 = Cn_mat['Cnelevator_tail.axes3.value']
        self.Cn_elevator_tail_4 = Cn_mat['Cnelevator_tail.axes4.value']
        self.Cn_elevator_tail_5 = Cn_mat['Cnelevator_tail.axes5.value']
        self.Cn_elevator_tail_6 = Cn_mat['Cnelevator_tail.axes6.value']
        self.Cn_elevator_tail_7 = Cn_mat['Cnelevator_tail.axes7.value']

        # Flap Wing
        self.Cn_flap_wing_val = Cn_mat['Cnflap_wing.value']
        self.Cn_flap_wing_1 = Cn_mat['Cnflap_wing.axes1.value']
        self.Cn_flap_wing_2 = Cn_mat['Cnflap_wing.axes2.value']
        self.Cn_flap_wing_3 = Cn_mat['Cnflap_wing.axes3.value']
        self.Cn_flap_wing_4 = Cn_mat['Cnflap_wing.axes4.value']
        self.Cn_flap_wing_5 = Cn_mat['Cnflap_wing.axes5.value']
        self.Cn_flap_wing_6 = Cn_mat['Cnflap_wing.axes6.value']
        self.Cn_flap_wing_7 = Cn_mat['Cnflap_wing.axes7.value']

        # Rudder Tail
        self.Cn_rudder_tail_val = Cn_mat['Cnrudder_tail.value']
        self.Cn_rudder_tail_1 = Cn_mat['Cnrudder_tail.axes1.value']
        self.Cn_rudder_tail_2 = Cn_mat['Cnrudder_tail.axes2.value']
        self.Cn_rudder_tail_3 = Cn_mat['Cnrudder_tail.axes3.value']
        self.Cn_rudder_tail_4 = Cn_mat['Cnrudder_tail.axes4.value']
        self.Cn_rudder_tail_5 = Cn_mat['Cnrudder_tail.axes5.value']
        self.Cn_rudder_tail_6 = Cn_mat['Cnrudder_tail.axes6.value']
        self.Cn_rudder_tail_7 = Cn_mat['Cnrudder_tail.axes7.value']

        # Tail
        self.Cn_tail_val = Cn_mat['Cntail.value']
        self.Cn_tail_1 = Cn_mat['Cntail.axes1.value']
        self.Cn_tail_2 = Cn_mat['Cntail.axes2.value']
        self.Cn_tail_3 = Cn_mat['Cntail.axes3.value']
        self.Cn_tail_4 = Cn_mat['Cntail.axes4.value']
        self.Cn_tail_5 = Cn_mat['Cntail.axes5.value']
        self.Cn_tail_6 = Cn_mat['Cntail.axes6.value']

        # Tail Damp p
        self.Cn_tail_damp_p_val = Cn_mat['Cntail_damp_p.value']
        self.Cn_tail_damp_p_1 = Cn_mat['Cntail_damp_p.axes1.value']
        self.Cn_tail_damp_p_2 = Cn_mat['Cntail_damp_p.axes2.value']
        self.Cn_tail_damp_p_3 = Cn_mat['Cntail_damp_p.axes3.value']
        self.Cn_tail_damp_p_4 = Cn_mat['Cntail_damp_p.axes4.value']
        self.Cn_tail_damp_p_5 = Cn_mat['Cntail_damp_p.axes5.value']
        self.Cn_tail_damp_p_6 = Cn_mat['Cntail_damp_p.axes6.value']

        # Tail Damp q
        self.Cn_tail_damp_q_val = Cn_mat['Cntail_damp_q.value']
        self.Cn_tail_damp_q_1 = Cn_mat['Cntail_damp_q.axes1.value']
        self.Cn_tail_damp_q_2 = Cn_mat['Cntail_damp_q.axes2.value']
        self.Cn_tail_damp_q_3 = Cn_mat['Cntail_damp_q.axes3.value']
        self.Cn_tail_damp_q_4 = Cn_mat['Cntail_damp_q.axes4.value']
        self.Cn_tail_damp_q_5 = Cn_mat['Cntail_damp_q.axes5.value']
        self.Cn_tail_damp_q_6 = Cn_mat['Cntail_damp_q.axes6.value']

        # Tail Damp r
        self.Cn_tail_damp_r_val = Cn_mat['Cntail_damp_r.value']
        self.Cn_tail_damp_r_1 = Cn_mat['Cntail_damp_r.axes1.value']
        self.Cn_tail_damp_r_2 = Cn_mat['Cntail_damp_r.axes2.value']
        self.Cn_tail_damp_r_3 = Cn_mat['Cntail_damp_r.axes3.value']
        self.Cn_tail_damp_r_4 = Cn_mat['Cntail_damp_r.axes4.value']
        self.Cn_tail_damp_r_5 = Cn_mat['Cntail_damp_r.axes5.value']
        self.Cn_tail_damp_r_6 = Cn_mat['Cntail_damp_r.axes6.value']

        # Wing
        self.Cn_wing_val = Cn_mat['Cnwing.value']
        self.Cn_wing_1 = Cn_mat['Cnwing.axes1.value']
        self.Cn_wing_2 = Cn_mat['Cnwing.axes2.value']
        self.Cn_wing_3 = Cn_mat['Cnwing.axes3.value']
        self.Cn_wing_4 = Cn_mat['Cnwing.axes4.value']
        self.Cn_wing_5 = Cn_mat['Cnwing.axes5.value']
        self.Cn_wing_6 = Cn_mat['Cnwing.axes6.value']

        # Wing Damp p
        self.Cn_wing_damp_p_val = Cn_mat['Cnwing_damp_p.value']
        self.Cn_wing_damp_p_1 = Cn_mat['Cnwing_damp_p.axes1.value']
        self.Cn_wing_damp_p_2 = Cn_mat['Cnwing_damp_p.axes2.value']
        self.Cn_wing_damp_p_3 = Cn_mat['Cnwing_damp_p.axes3.value']
        self.Cn_wing_damp_p_4 = Cn_mat['Cnwing_damp_p.axes4.value']
        self.Cn_wing_damp_p_5 = Cn_mat['Cnwing_damp_p.axes5.value']
        self.Cn_wing_damp_p_6 = Cn_mat['Cnwing_damp_p.axes6.value']

        # Wing Damp q
        self.Cn_wing_damp_q_val = Cn_mat['Cnwing_damp_q.value']
        self.Cn_wing_damp_q_1 = Cn_mat['Cnwing_damp_q.axes1.value']
        self.Cn_wing_damp_q_2 = Cn_mat['Cnwing_damp_q.axes2.value']
        self.Cn_wing_damp_q_3 = Cn_mat['Cnwing_damp_q.axes3.value']
        self.Cn_wing_damp_q_4 = Cn_mat['Cnwing_damp_q.axes4.value']
        self.Cn_wing_damp_q_5 = Cn_mat['Cnwing_damp_q.axes5.value']
        self.Cn_wing_damp_q_6 = Cn_mat['Cnwing_damp_q.axes6.value']

        # Wing Damp r
        self.Cn_wing_damp_r_val = Cn_mat['Cnwing_damp_r.value']
        self.Cn_wing_damp_r_1 = Cn_mat['Cnwing_damp_r.axes1.value']
        self.Cn_wing_damp_r_2 = Cn_mat['Cnwing_damp_r.axes2.value']
        self.Cn_wing_damp_r_3 = Cn_mat['Cnwing_damp_r.axes3.value']
        self.Cn_wing_damp_r_4 = Cn_mat['Cnwing_damp_r.axes4.value']
        self.Cn_wing_damp_r_5 = Cn_mat['Cnwing_damp_r.axes5.value']
        self.Cn_wing_damp_r_6 = Cn_mat['Cnwing_damp_r.axes6.value']

        # Hover Fuse
        self.Cn_hover_fuse_val = Cn_mat['Cnhover_fuse.value']
        self.Cn_hover_fuse_1 = Cn_mat['Cnhover_fuse.axes1.value']
        self.Cn_hover_fuse_2 = Cn_mat['Cnhover_fuse.axes2.value']
        self.Cn_hover_fuse_3 = Cn_mat['Cnhover_fuse.axes3.value']

        # Ct Data
        Ct_mat = aero.data['Ct']

        # Tail Left
        self.Ct_tail_left_val = Ct_mat['Ct_tailLeft.value']
        self.Ct_tail_left_1 = Ct_mat['Ct_tailLeft.axes1.value']
        self.Ct_tail_left_2 = Ct_mat['Ct_tailLeft.axes2.value']
        self.Ct_tail_left_3 = Ct_mat['Ct_tailLeft.axes3.value']
        self.Ct_tail_left_4 = Ct_mat['Ct_tailLeft.axes4.value']

        # Tail Right
        self.Ct_tail_right_val = Ct_mat['Ct_tailRight.value']
        self.Ct_tail_right_1 = Ct_mat['Ct_tailRight.axes1.value']
        self.Ct_tail_right_2 = Ct_mat['Ct_tailRight.axes2.value']
        self.Ct_tail_right_3 = Ct_mat['Ct_tailRight.axes3.value']
        self.Ct_tail_right_4 = Ct_mat['Ct_tailRight.axes4.value']

        # Left Out
        self.Ct_left_out_val = Ct_mat['Ct_leftOut1.value']
        self.Ct_left_out_1 = Ct_mat['Ct_leftOut1.axes1.value']
        self.Ct_left_out_2 = Ct_mat['Ct_leftOut1.axes2.value']
        self.Ct_left_out_3 = Ct_mat['Ct_leftOut1.axes3.value']
        self.Ct_left_out_4 = Ct_mat['Ct_leftOut1.axes4.value']

        # Left 2
        self.Ct_left_2_val = Ct_mat['Ct_left2.value']
        self.Ct_left_2_1 = Ct_mat['Ct_left2.axes1.value']
        self.Ct_left_2_2 = Ct_mat['Ct_left2.axes2.value']
        self.Ct_left_2_3 = Ct_mat['Ct_left2.axes3.value']
        self.Ct_left_2_4 = Ct_mat['Ct_left2.axes4.value']

        # Left 3
        self.Ct_left_3_val = Ct_mat['Ct_left3.value']
        self.Ct_left_3_1 = Ct_mat['Ct_left3.axes1.value']
        self.Ct_left_3_2 = Ct_mat['Ct_left3.axes2.value']
        self.Ct_left_3_3 = Ct_mat['Ct_left3.axes3.value']
        self.Ct_left_3_4 = Ct_mat['Ct_left3.axes4.value']

        # Left 4
        self.Ct_left_4_val = Ct_mat['Ct_left4.value']
        self.Ct_left_4_1 = Ct_mat['Ct_left4.axes1.value']
        self.Ct_left_4_2 = Ct_mat['Ct_left4.axes2.value']
        self.Ct_left_4_3 = Ct_mat['Ct_left4.axes3.value']
        self.Ct_left_4_4 = Ct_mat['Ct_left4.axes4.value']

        # Left 5
        self.Ct_left_5_val = Ct_mat['Ct_left5.value']
        self.Ct_left_5_1 = Ct_mat['Ct_left5.axes1.value']
        self.Ct_left_5_2 = Ct_mat['Ct_left5.axes2.value']
        self.Ct_left_5_3 = Ct_mat['Ct_left5.axes3.value']
        self.Ct_left_5_4 = Ct_mat['Ct_left5.axes4.value']

        # Left 6 In
        self.Ct_left_6_in_val = Ct_mat['Ct_left6In.value']
        self.Ct_left_6_in_1 = Ct_mat['Ct_left6In.axes1.value']
        self.Ct_left_6_in_2 = Ct_mat['Ct_left6In.axes2.value']
        self.Ct_left_6_in_3 = Ct_mat['Ct_left6In.axes3.value']
        self.Ct_left_6_in_4 = Ct_mat['Ct_left6In.axes4.value']

        # Right 7 In
        self.Ct_right_7_in_val = Ct_mat['Ct_right7In.value']
        self.Ct_right_7_in_1 = Ct_mat['Ct_right7In.axes1.value']
        self.Ct_right_7_in_2 = Ct_mat['Ct_right7In.axes2.value']
        self.Ct_right_7_in_3 = Ct_mat['Ct_right7In.axes3.value']
        self.Ct_right_7_in_4 = Ct_mat['Ct_right7In.axes4.value']

        # Right 8
        self.Ct_right_8_val = Ct_mat['Ct_right8.value']
        self.Ct_right_8_1 = Ct_mat['Ct_right8.axes1.value']
        self.Ct_right_8_2 = Ct_mat['Ct_right8.axes2.value']
        self.Ct_right_8_3 = Ct_mat['Ct_right8.axes3.value']
        self.Ct_right_8_4 = Ct_mat['Ct_right8.axes4.value']

        # Right 9
        self.Ct_right_9_val = Ct_mat['Ct_right9.value']
        self.Ct_right_9_1 = Ct_mat['Ct_right9.axes1.value']
        self.Ct_right_9_2 = Ct_mat['Ct_right9.axes2.value']
        self.Ct_right_9_3 = Ct_mat['Ct_right9.axes3.value']
        self.Ct_right_9_4 = Ct_mat['Ct_right9.axes4.value']

        # Right 10
        self.Ct_right_10_val = Ct_mat['Ct_right10.value']
        self.Ct_right_10_1 = Ct_mat['Ct_right10.axes1.value']
        self.Ct_right_10_2 = Ct_mat['Ct_right10.axes2.value']
        self.Ct_right_10_3 = Ct_mat['Ct_right10.axes3.value']
        self.Ct_right_10_4 = Ct_mat['Ct_right10.axes4.value']

        # Right 11
        self.Ct_right_11_val = Ct_mat['Ct_right11.value']
        self.Ct_right_11_1 = Ct_mat['Ct_right11.axes1.value']
        self.Ct_right_11_2 = Ct_mat['Ct_right11.axes2.value']
        self.Ct_right_11_3 = Ct_mat['Ct_right11.axes3.value']
        self.Ct_right_11_4 = Ct_mat['Ct_right11.axes4.value']

        # Right 12 Out
        self.Ct_right_12_out_val = Ct_mat['Ct_right12Out.value']
        self.Ct_right_12_out_1 = Ct_mat['Ct_right12Out.axes1.value']
        self.Ct_right_12_out_2 = Ct_mat['Ct_right12Out.axes2.value']
        self.Ct_right_12_out_3 = Ct_mat['Ct_right12Out.axes3.value']
        self.Ct_right_12_out_4 = Ct_mat['Ct_right12Out.axes4.value']

        # Kq Data
        Kq_mat = aero.data['Kq']

        # Tail Left
        self.Kq_tail_left_val = Kq_mat['Kq_tailLeft.value']
        self.Kq_tail_left_1 = Kq_mat['Kq_tailLeft.axes1.value']
        self.Kq_tail_left_2 = Kq_mat['Kq_tailLeft.axes2.value']
        self.Kq_tail_left_3 = Kq_mat['Kq_tailLeft.axes3.value']
        self.Kq_tail_left_4 = Kq_mat['Kq_tailLeft.axes4.value']

        # Tail Right
        self.Kq_tail_right_val = Kq_mat['Kq_tailRight.value']
        self.Kq_tail_right_1 = Kq_mat['Kq_tailRight.axes1.value']
        self.Kq_tail_right_2 = Kq_mat['Kq_tailRight.axes2.value']
        self.Kq_tail_right_3 = Kq_mat['Kq_tailRight.axes3.value']
        self.Kq_tail_right_4 = Kq_mat['Kq_tailRight.axes4.value']

        # Left Out
        self.Kq_left_out_val = Kq_mat['Kq_leftOut1.value']
        self.Kq_left_out_1 = Kq_mat['Kq_leftOut1.axes1.value']
        self.Kq_left_out_2 = Kq_mat['Kq_leftOut1.axes2.value']
        self.Kq_left_out_3 = Kq_mat['Kq_leftOut1.axes3.value']
        self.Kq_left_out_4 = Kq_mat['Kq_leftOut1.axes4.value']

        # Left 2
        self.Kq_left_2_val = Kq_mat['Kq_left2.value']
        self.Kq_left_2_1 = Kq_mat['Kq_left2.axes1.value']
        self.Kq_left_2_2 = Kq_mat['Kq_left2.axes2.value']
        self.Kq_left_2_3 = Kq_mat['Kq_left2.axes3.value']
        self.Kq_left_2_4 = Kq_mat['Kq_left2.axes4.value']

        # Left 3
        self.Kq_left_3_val = Kq_mat['Kq_left3.value']
        self.Kq_left_3_1 = Kq_mat['Kq_left3.axes1.value']
        self.Kq_left_3_2 = Kq_mat['Kq_left3.axes2.value']
        self.Kq_left_3_3 = Kq_mat['Kq_left3.axes3.value']
        self.Kq_left_3_4 = Kq_mat['Kq_left3.axes4.value']

        # Left 4
        self.Kq_left_4_val = Kq_mat['Kq_left4.value']
        self.Kq_left_4_1 = Kq_mat['Kq_left4.axes1.value']
        self.Kq_left_4_2 = Kq_mat['Kq_left4.axes2.value']
        self.Kq_left_4_3 = Kq_mat['Kq_left4.axes3.value']
        self.Kq_left_4_4 = Kq_mat['Kq_left4.axes4.value']

        # Left 5
        self.Kq_left_5_val = Kq_mat['Kq_left5.value']
        self.Kq_left_5_1 = Kq_mat['Kq_left5.axes1.value']
        self.Kq_left_5_2 = Kq_mat['Kq_left5.axes2.value']
        self.Kq_left_5_3 = Kq_mat['Kq_left5.axes3.value']
        self.Kq_left_5_4 = Kq_mat['Kq_left5.axes4.value']

        # Left 6 In
        self.Kq_left_6_in_val = Kq_mat['Kq_left6In.value']
        self.Kq_left_6_in_1 = Kq_mat['Kq_left6In.axes1.value']
        self.Kq_left_6_in_2 = Kq_mat['Kq_left6In.axes2.value']
        self.Kq_left_6_in_3 = Kq_mat['Kq_left6In.axes3.value']
        self.Kq_left_6_in_4 = Kq_mat['Kq_left6In.axes4.value']

        # Right 7 In
        self.Kq_right_7_in_val = Kq_mat['Kq_right7In.value']
        self.Kq_right_7_in_1 = Kq_mat['Kq_right7In.axes1.value']
        self.Kq_right_7_in_2 = Kq_mat['Kq_right7In.axes2.value']
        self.Kq_right_7_in_3 = Kq_mat['Kq_right7In.axes3.value']
        self.Kq_right_7_in_4 = Kq_mat['Kq_right7In.axes4.value']

        # Right 8
        self.Kq_right_8_val = Kq_mat['Kq_right8.value']
        self.Kq_right_8_1 = Kq_mat['Kq_right8.axes1.value']
        self.Kq_right_8_2 = Kq_mat['Kq_right8.axes2.value']
        self.Kq_right_8_3 = Kq_mat['Kq_right8.axes3.value']
        self.Kq_right_8_4 = Kq_mat['Kq_right8.axes4.value']

        # Right 9
        self.Kq_right_9_val = Kq_mat['Kq_right9.value']
        self.Kq_right_9_1 = Kq_mat['Kq_right9.axes1.value']
        self.Kq_right_9_2 = Kq_mat['Kq_right9.axes2.value']
        self.Kq_right_9_3 = Kq_mat['Kq_right9.axes3.value']
        self.Kq_right_9_4 = Kq_mat['Kq_right9.axes4.value']

        # Right 10
        self.Kq_right_10_val = Kq_mat['Kq_right10.value']
        self.Kq_right_10_1 = Kq_mat['Kq_right10.axes1.value']
        self.Kq_right_10_2 = Kq_mat['Kq_right10.axes2.value']
        self.Kq_right_10_3 = Kq_mat['Kq_right10.axes3.value']
        self.Kq_right_10_4 = Kq_mat['Kq_right10.axes4.value']

        # Right 11
        self.Kq_right_11_val = Kq_mat['Kq_right11.value']
        self.Kq_right_11_1 = Kq_mat['Kq_right11.axes1.value']
        self.Kq_right_11_2 = Kq_mat['Kq_right11.axes2.value']
        self.Kq_right_11_3 = Kq_mat['Kq_right11.axes3.value']
        self.Kq_right_11_4 = Kq_mat['Kq_right11.axes4.value']

        # Right 12 Out
        self.Kq_right_12_out_val = Kq_mat['Kq_right12Out.value']
        self.Kq_right_12_out_1 = Kq_mat['Kq_right12Out.axes1.value']
        self.Kq_right_12_out_2 = Kq_mat['Kq_right12Out.axes2.value']
        self.Kq_right_12_out_3 = Kq_mat['Kq_right12Out.axes3.value']
        self.Kq_right_12_out_4 = Kq_mat['Kq_right12Out.axes4.value']


if __name__ == '__main__':
    # Usage Example with Perturbation and JIT Compatibility
    file_path = 'aero_export.mat'  # Replace with your .mat file path
    mavrik_setup = MavrikSetup(file_path)
    for _ in range(100):
        # Example of accessing some attributes
        print("CY_aileron_wing_val.shape:", mavrik_setup.CY_aileron_wing_val.shape)
        print("CZ_tail_val.shape:", mavrik_setup.CZ_tail_val.shape)
        print("Cl_wing_val.shape:", mavrik_setup.Cl_wing_val.shape)