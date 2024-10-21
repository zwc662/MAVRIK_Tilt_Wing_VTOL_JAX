%% Setup the Mavrik model

aero = load('aero_export.mat');

%% Scaling Values

S = 0.5744;
c = 0.2032;
b = 2.8270;
rho = 1.225;
Tail_prop_D4 = 0.005059318992632;
Wing_prop_D4 = 0.021071715921;
Tail_prop_D5 = 0.001349320375335;
Wing_prop_D5 = 0.008028323765901;

C1 = 0.25;
C2 = 2.777777777777778*10^-4;

mass = 25; 
inertia = diag([4.8518,6.0388,9.4731]);
U=30;
pqr_init=[0,0,0];
euler_init=[0,4*pi/180,0];
uvw_init=[U*cos(euler_init(2)),0,U*sin(euler_init(2))];
xyz_init=[0,0,0];

%% Positions

aero_center = [0.3353, 0, 0.0508];
CG = [0.3353, 0, 0.0508]; % Arbitrary CG to test calculations
position = aero_center - CG;

tailLeft_pos = [1.26, -0.4506, 0.1295];
RPM_tailLeft_trans = [-tailLeft_pos(1) + CG(1), tailLeft_pos(2) - CG(2), -tailLeft_pos(3) + CG(3)];

tailRight_pos = [1.26, 0.4506, 0.1295];
RPM_tailRight_trans = [-tailRight_pos(1) + CG(1), tailRight_pos(2) - CG(2), -tailRight_pos(3) + CG(3)];

leftOut1_pos = [0.5443, -1.4092, 0.04064];
RPM_leftOut1_trans = [-leftOut1_pos(1) + CG(1), leftOut1_pos(2) - CG(2), -leftOut1_pos(3) + CG(3)];

left2_pos = [0.5159, -1.1963, 0.04064];
RPM_left2_trans = [-left2_pos(1) + CG(1), left2_pos(2) - CG(2), -left2_pos(3) + CG(3)];

left3_pos = [0.5443, -0.9751, 0.04064];
RPM_left3_trans = [-left3_pos(1) + CG(1), left3_pos(2) - CG(2), -left3_pos(3) + CG(3)];

left4_pos = [0.5159, -0.7539, 0.04064];
RPM_left4_trans = [-left4_pos(1) + CG(1), left4_pos(2) - CG(2), -left4_pos(3) + CG(3)];

left5_pos = [0.5443, -0.5448, 0.04064];
RPM_left5_trans = [-left5_pos(1) + CG(1), left5_pos(2) - CG(2), -left5_pos(3) + CG(3)];

left6In_pos = [0.5159, -0.3277, 0.04064];
RPM_left6In_trans = [-left6In_pos(1) + CG(1), left6In_pos(2) - CG(2), -left6In_pos(3) + CG(3)];

right7In_pos = [0.5159, 0.3277, 0.04064];
RPM_right7In_trans = [-right7In_pos(1) + CG(1), right7In_pos(2) - CG(2), -right7In_pos(3) + CG(3)];

right8_pos = [0.5443, 0.5448, 0.04064];
RPM_right8_trans = [-right8_pos(1) + CG(1), right8_pos(2) - CG(2), -right8_pos(3) + CG(3)];

right9_pos = [0.5159, 0.7539, 0.04064];
RPM_right9_trans = [-right9_pos(1) + CG(1), right9_pos(2) - CG(2), -right9_pos(3) + CG(3)];

right10_pos = [0.5443, 0.9751, 0.04064];
RPM_right10_trans = [-right10_pos(1) + CG(1), right10_pos(2) - CG(2), -right10_pos(3) + CG(3)];

right11_pos = [0.5159, 1.1963, 0.04064];
RPM_right11_trans = [-right11_pos(1) + CG(1), right11_pos(2) - CG(2), -right11_pos(3) + CG(3)];

right12Out_pos = [0.5443, 1.4092, 0.04064];
RPM_right12Out_trans = [-right12Out_pos(1) + CG(1), right12Out_pos(2) - CG(2), -right12Out_pos(3) + CG(3)];

%% CX Data

CX_mat = aero.CX;

% Aileron Wing
CX_aileron_wing_mat = CX_mat.CXaileron_wing;
CX_aileron_wing_val = CX_mat.CXaileron_wing.value;
CX_aileron_wing_1 = CX_mat.CXaileron_wing.axes1.value;
CX_aileron_wing_2 = CX_mat.CXaileron_wing.axes2.value;
CX_aileron_wing_3 = CX_mat.CXaileron_wing.axes3.value;
CX_aileron_wing_4 = CX_mat.CXaileron_wing.axes4.value;
CX_aileron_wing_5 = CX_mat.CXaileron_wing.axes5.value;
CX_aileron_wing_6 = CX_mat.CXaileron_wing.axes6.value;
CX_aileron_wing_7 = CX_mat.CXaileron_wing.axes7.value;

% Elevator Tail
CX_elevator_tail_mat = CX_mat.CXelevator_tail;
CX_elevator_tail_val = CX_mat.CXelevator_tail.value;
CX_elevator_tail_1 = CX_mat.CXelevator_tail.axes1.value;
CX_elevator_tail_2 = CX_mat.CXelevator_tail.axes2.value;
CX_elevator_tail_3 = CX_mat.CXelevator_tail.axes3.value;
CX_elevator_tail_4 = CX_mat.CXelevator_tail.axes4.value;
CX_elevator_tail_5 = CX_mat.CXelevator_tail.axes5.value;
CX_elevator_tail_6 = CX_mat.CXelevator_tail.axes6.value;
CX_elevator_tail_7 = CX_mat.CXelevator_tail.axes7.value;

% Flap Wing
CX_flap_wing_mat = CX_mat.CXflap_wing;
CX_flap_wing_val = CX_mat.CXflap_wing.value;
CX_flap_wing_1 = CX_mat.CXflap_wing.axes1.value;
CX_flap_wing_2 = CX_mat.CXflap_wing.axes2.value;
CX_flap_wing_3 = CX_mat.CXflap_wing.axes3.value;
CX_flap_wing_4 = CX_mat.CXflap_wing.axes4.value;
CX_flap_wing_5 = CX_mat.CXflap_wing.axes5.value;
CX_flap_wing_6 = CX_mat.CXflap_wing.axes6.value;
CX_flap_wing_7 = CX_mat.CXflap_wing.axes7.value;

% Rudder Tail
CX_rudder_tail_mat = CX_mat.CXrudder_tail;
CX_rudder_tail_val = CX_mat.CXrudder_tail.value;
CX_rudder_tail_1 = CX_mat.CXrudder_tail.axes1.value;
CX_rudder_tail_2 = CX_mat.CXrudder_tail.axes2.value;
CX_rudder_tail_3 = CX_mat.CXrudder_tail.axes3.value;
CX_rudder_tail_4 = CX_mat.CXrudder_tail.axes4.value;
CX_rudder_tail_5 = CX_mat.CXrudder_tail.axes5.value;
CX_rudder_tail_6 = CX_mat.CXrudder_tail.axes6.value;
CX_rudder_tail_7 = CX_mat.CXrudder_tail.axes7.value;

% Tail
CX_tail_mat = CX_mat.CXtail;
CX_tail_val = CX_mat.CXtail.value;
CX_tail_1 = CX_mat.CXtail.axes1.value;
CX_tail_2 = CX_mat.CXtail.axes2.value;
CX_tail_3 = CX_mat.CXtail.axes3.value;
CX_tail_4 = CX_mat.CXtail.axes4.value;
CX_tail_5 = CX_mat.CXtail.axes5.value;
CX_tail_6 = CX_mat.CXtail.axes6.value;

% Tail Damp p
CX_tail_damp_p_mat = CX_mat.CXtail_damp_p;
CX_tail_damp_p_val = CX_mat.CXtail_damp_p.value;
CX_tail_damp_p_1 = CX_mat.CXtail_damp_p.axes1.value;
CX_tail_damp_p_2 = CX_mat.CXtail_damp_p.axes2.value;
CX_tail_damp_p_3 = CX_mat.CXtail_damp_p.axes3.value;
CX_tail_damp_p_4 = CX_mat.CXtail_damp_p.axes4.value;
CX_tail_damp_p_5 = CX_mat.CXtail_damp_p.axes5.value;
CX_tail_damp_p_6 = CX_mat.CXtail_damp_p.axes6.value;

% Tail Damp q
CX_tail_damp_q_mat = CX_mat.CXtail_damp_q;
CX_tail_damp_q_val = CX_mat.CXtail_damp_q.value;
CX_tail_damp_q_1 = CX_mat.CXtail_damp_q.axes1.value;
CX_tail_damp_q_2 = CX_mat.CXtail_damp_q.axes2.value;
CX_tail_damp_q_3 = CX_mat.CXtail_damp_q.axes3.value;
CX_tail_damp_q_4 = CX_mat.CXtail_damp_q.axes4.value;
CX_tail_damp_q_5 = CX_mat.CXtail_damp_q.axes5.value;
CX_tail_damp_q_6 = CX_mat.CXtail_damp_q.axes6.value;

% Tail Damp r
CX_tail_damp_r_mat = CX_mat.CXtail_damp_r;
CX_tail_damp_r_val = CX_mat.CXtail_damp_r.value;
CX_tail_damp_r_1 = CX_mat.CXtail_damp_r.axes1.value;
CX_tail_damp_r_2 = CX_mat.CXtail_damp_r.axes2.value;
CX_tail_damp_r_3 = CX_mat.CXtail_damp_r.axes3.value;
CX_tail_damp_r_4 = CX_mat.CXtail_damp_r.axes4.value;
CX_tail_damp_r_5 = CX_mat.CXtail_damp_r.axes5.value;
CX_tail_damp_r_6 = CX_mat.CXtail_damp_r.axes6.value;

% Wing
CX_wing_mat = CX_mat.CXwing;
CX_wing_val = CX_mat.CXwing.value;
CX_wing_1 = CX_mat.CXwing.axes1.value;
CX_wing_2 = CX_mat.CXwing.axes2.value;
CX_wing_3 = CX_mat.CXwing.axes3.value;
CX_wing_4 = CX_mat.CXwing.axes4.value;
CX_wing_5 = CX_mat.CXwing.axes5.value;
CX_wing_6 = CX_mat.CXwing.axes6.value;

% Wing Damp p
CX_wing_damp_p_mat = CX_mat.CXwing_damp_p;
CX_wing_damp_p_val = CX_mat.CXwing_damp_p.value;
CX_wing_damp_p_1 = CX_mat.CXwing_damp_p.axes1.value;
CX_wing_damp_p_2 = CX_mat.CXwing_damp_p.axes2.value;
CX_wing_damp_p_3 = CX_mat.CXwing_damp_p.axes3.value;
CX_wing_damp_p_4 = CX_mat.CXwing_damp_p.axes4.value;
CX_wing_damp_p_5 = CX_mat.CXwing_damp_p.axes5.value;
CX_wing_damp_p_6 = CX_mat.CXwing_damp_p.axes6.value;

% Wing Damp q
CX_wing_damp_q_mat = CX_mat.CXwing_damp_q;
CX_wing_damp_q_val = CX_mat.CXwing_damp_q.value;
CX_wing_damp_q_1 = CX_mat.CXwing_damp_q.axes1.value;
CX_wing_damp_q_2 = CX_mat.CXwing_damp_q.axes2.value;
CX_wing_damp_q_3 = CX_mat.CXwing_damp_q.axes3.value;
CX_wing_damp_q_4 = CX_mat.CXwing_damp_q.axes4.value;
CX_wing_damp_q_5 = CX_mat.CXwing_damp_q.axes5.value;
CX_wing_damp_q_6 = CX_mat.CXwing_damp_q.axes6.value;

% Wing Damp r
CX_wing_damp_r_mat = CX_mat.CXwing_damp_r;
CX_wing_damp_r_val = CX_mat.CXwing_damp_r.value;
CX_wing_damp_r_1 = CX_mat.CXwing_damp_r.axes1.value;
CX_wing_damp_r_2 = CX_mat.CXwing_damp_r.axes2.value;
CX_wing_damp_r_3 = CX_mat.CXwing_damp_r.axes3.value;
CX_wing_damp_r_4 = CX_mat.CXwing_damp_r.axes4.value;
CX_wing_damp_r_5 = CX_mat.CXwing_damp_r.axes5.value;
CX_wing_damp_r_6 = CX_mat.CXwing_damp_r.axes6.value;

% Hover Fuse
CX_hover_fuse_mat = CX_mat.CXhover_fuse;
CX_hover_fuse_val = CX_mat.CXhover_fuse.value;
CX_hover_fuse_1 = CX_mat.CXhover_fuse.axes1.value;
CX_hover_fuse_2 = CX_mat.CXhover_fuse.axes2.value;
CX_hover_fuse_3 = CX_mat.CXhover_fuse.axes3.value;

%% CY Data

CY_mat = aero.CY;

% Aileron Wing
CY_aileron_wing_mat = CY_mat.CYaileron_wing;
CY_aileron_wing_val = CY_mat.CYaileron_wing.value;
CY_aileron_wing_1 = CY_mat.CYaileron_wing.axes1.value;
CY_aileron_wing_2 = CY_mat.CYaileron_wing.axes2.value;
CY_aileron_wing_3 = CY_mat.CYaileron_wing.axes3.value;
CY_aileron_wing_4 = CY_mat.CYaileron_wing.axes4.value;
CY_aileron_wing_5 = CY_mat.CYaileron_wing.axes5.value;
CY_aileron_wing_6 = CY_mat.CYaileron_wing.axes6.value;
CY_aileron_wing_7 = CY_mat.CYaileron_wing.axes7.value;

% Elevator Tail
CY_elevator_tail_mat = CY_mat.CYelevator_tail;
CY_elevator_tail_val = CY_mat.CYelevator_tail.value;
CY_elevator_tail_1 = CY_mat.CYelevator_tail.axes1.value;
CY_elevator_tail_2 = CY_mat.CYelevator_tail.axes2.value;
CY_elevator_tail_3 = CY_mat.CYelevator_tail.axes3.value;
CY_elevator_tail_4 = CY_mat.CYelevator_tail.axes4.value;
CY_elevator_tail_5 = CY_mat.CYelevator_tail.axes5.value;
CY_elevator_tail_6 = CY_mat.CYelevator_tail.axes6.value;
CY_elevator_tail_7 = CY_mat.CYelevator_tail.axes7.value;

% Flap Wing
CY_flap_wing_mat = CY_mat.CYflap_wing;
CY_flap_wing_val = CY_mat.CYflap_wing.value;
CY_flap_wing_1 = CY_mat.CYflap_wing.axes1.value;
CY_flap_wing_2 = CY_mat.CYflap_wing.axes2.value;
CY_flap_wing_3 = CY_mat.CYflap_wing.axes3.value;
CY_flap_wing_4 = CY_mat.CYflap_wing.axes4.value;
CY_flap_wing_5 = CY_mat.CYflap_wing.axes5.value;
CY_flap_wing_6 = CY_mat.CYflap_wing.axes6.value;
CY_flap_wing_7 = CY_mat.CYflap_wing.axes7.value;

% Rudder Tail
CY_rudder_tail_mat = CY_mat.CYrudder_tail;
CY_rudder_tail_val = CY_mat.CYrudder_tail.value;
CY_rudder_tail_1 = CY_mat.CYrudder_tail.axes1.value;
CY_rudder_tail_2 = CY_mat.CYrudder_tail.axes2.value;
CY_rudder_tail_3 = CY_mat.CYrudder_tail.axes3.value;
CY_rudder_tail_4 = CY_mat.CYrudder_tail.axes4.value;
CY_rudder_tail_5 = CY_mat.CYrudder_tail.axes5.value;
CY_rudder_tail_6 = CY_mat.CYrudder_tail.axes6.value;
CY_rudder_tail_7 = CY_mat.CYrudder_tail.axes7.value;

% Tail
CY_tail_mat = CY_mat.CYtail;
CY_tail_val = CY_mat.CYtail.value;
CY_tail_1 = CY_mat.CYtail.axes1.value;
CY_tail_2 = CY_mat.CYtail.axes2.value;
CY_tail_3 = CY_mat.CYtail.axes3.value;
CY_tail_4 = CY_mat.CYtail.axes4.value;
CY_tail_5 = CY_mat.CYtail.axes5.value;
CY_tail_6 = CY_mat.CYtail.axes6.value;

% Tail Damp p
CY_tail_damp_p_mat = CY_mat.CYtail_damp_p;
CY_tail_damp_p_val = CY_mat.CYtail_damp_p.value;
CY_tail_damp_p_1 = CY_mat.CYtail_damp_p.axes1.value;
CY_tail_damp_p_2 = CY_mat.CYtail_damp_p.axes2.value;
CY_tail_damp_p_3 = CY_mat.CYtail_damp_p.axes3.value;
CY_tail_damp_p_4 = CY_mat.CYtail_damp_p.axes4.value;
CY_tail_damp_p_5 = CY_mat.CYtail_damp_p.axes5.value;
CY_tail_damp_p_6 = CY_mat.CYtail_damp_p.axes6.value;

% Tail Damp q
CY_tail_damp_q_mat = CY_mat.CYtail_damp_q;
CY_tail_damp_q_val = CY_mat.CYtail_damp_q.value;
CY_tail_damp_q_1 = CY_mat.CYtail_damp_q.axes1.value;
CY_tail_damp_q_2 = CY_mat.CYtail_damp_q.axes2.value;
CY_tail_damp_q_3 = CY_mat.CYtail_damp_q.axes3.value;
CY_tail_damp_q_4 = CY_mat.CYtail_damp_q.axes4.value;
CY_tail_damp_q_5 = CY_mat.CYtail_damp_q.axes5.value;
CY_tail_damp_q_6 = CY_mat.CYtail_damp_q.axes6.value;

% Tail Damp r
CY_tail_damp_r_mat = CY_mat.CYtail_damp_r;
CY_tail_damp_r_val = CY_mat.CYtail_damp_r.value;
CY_tail_damp_r_1 = CY_mat.CYtail_damp_r.axes1.value;
CY_tail_damp_r_2 = CY_mat.CYtail_damp_r.axes2.value;
CY_tail_damp_r_3 = CY_mat.CYtail_damp_r.axes3.value;
CY_tail_damp_r_4 = CY_mat.CYtail_damp_r.axes4.value;
CY_tail_damp_r_5 = CY_mat.CYtail_damp_r.axes5.value;
CY_tail_damp_r_6 = CY_mat.CYtail_damp_r.axes6.value;

% Wing
CY_wing_mat = CY_mat.CYwing;
CY_wing_val = CY_mat.CYwing.value;
CY_wing_1 = CY_mat.CYwing.axes1.value;
CY_wing_2 = CY_mat.CYwing.axes2.value;
CY_wing_3 = CY_mat.CYwing.axes3.value;
CY_wing_4 = CY_mat.CYwing.axes4.value;
CY_wing_5 = CY_mat.CYwing.axes5.value;
CY_wing_6 = CY_mat.CYwing.axes6.value;

% Wing Damp p
CY_wing_damp_p_mat = CY_mat.CYwing_damp_p;
CY_wing_damp_p_val = CY_mat.CYwing_damp_p.value;
CY_wing_damp_p_1 = CY_mat.CYwing_damp_p.axes1.value;
CY_wing_damp_p_2 = CY_mat.CYwing_damp_p.axes2.value;
CY_wing_damp_p_3 = CY_mat.CYwing_damp_p.axes3.value;
CY_wing_damp_p_4 = CY_mat.CYwing_damp_p.axes4.value;
CY_wing_damp_p_5 = CY_mat.CYwing_damp_p.axes5.value;
CY_wing_damp_p_6 = CY_mat.CYwing_damp_p.axes6.value;

% Wing Damp q
CY_wing_damp_q_mat = CY_mat.CYwing_damp_q;
CY_wing_damp_q_val = CY_mat.CYwing_damp_q.value;
CY_wing_damp_q_1 = CY_mat.CYwing_damp_q.axes1.value;
CY_wing_damp_q_2 = CY_mat.CYwing_damp_q.axes2.value;
CY_wing_damp_q_3 = CY_mat.CYwing_damp_q.axes3.value;
CY_wing_damp_q_4 = CY_mat.CYwing_damp_q.axes4.value;
CY_wing_damp_q_5 = CY_mat.CYwing_damp_q.axes5.value;
CY_wing_damp_q_6 = CY_mat.CYwing_damp_q.axes6.value;

% Wing Damp r
CY_wing_damp_r_mat = CY_mat.CYwing_damp_r;
CY_wing_damp_r_val = CY_mat.CYwing_damp_r.value;
CY_wing_damp_r_1 = CY_mat.CYwing_damp_r.axes1.value;
CY_wing_damp_r_2 = CY_mat.CYwing_damp_r.axes2.value;
CY_wing_damp_r_3 = CY_mat.CYwing_damp_r.axes3.value;
CY_wing_damp_r_4 = CY_mat.CYwing_damp_r.axes4.value;
CY_wing_damp_r_5 = CY_mat.CYwing_damp_r.axes5.value;
CY_wing_damp_r_6 = CY_mat.CYwing_damp_r.axes6.value;

% Hover Fuse
CY_hover_fuse_mat = CY_mat.CYhover_fuse;
CY_hover_fuse_val = CY_mat.CYhover_fuse.value;
CY_hover_fuse_1 = CY_mat.CYhover_fuse.axes1.value;
CY_hover_fuse_2 = CY_mat.CYhover_fuse.axes2.value;
CY_hover_fuse_3 = CY_mat.CYhover_fuse.axes3.value;

%% CZ Data

CZ_mat = aero.CZ;

% Aileron Wing
CZ_aileron_wing_mat = CZ_mat.CZaileron_wing;
CZ_aileron_wing_val = CZ_mat.CZaileron_wing.value;
CZ_aileron_wing_1 = CZ_mat.CZaileron_wing.axes1.value;
CZ_aileron_wing_2 = CZ_mat.CZaileron_wing.axes2.value;
CZ_aileron_wing_3 = CZ_mat.CZaileron_wing.axes3.value;
CZ_aileron_wing_4 = CZ_mat.CZaileron_wing.axes4.value;
CZ_aileron_wing_5 = CZ_mat.CZaileron_wing.axes5.value;
CZ_aileron_wing_6 = CZ_mat.CZaileron_wing.axes6.value;
CZ_aileron_wing_7 = CZ_mat.CZaileron_wing.axes7.value;

% Elevator Tail
CZ_elevator_tail_mat = CZ_mat.CZelevator_tail;
CZ_elevator_tail_val = CZ_mat.CZelevator_tail.value;
CZ_elevator_tail_1 = CZ_mat.CZelevator_tail.axes1.value;
CZ_elevator_tail_2 = CZ_mat.CZelevator_tail.axes2.value;
CZ_elevator_tail_3 = CZ_mat.CZelevator_tail.axes3.value;
CZ_elevator_tail_4 = CZ_mat.CZelevator_tail.axes4.value;
CZ_elevator_tail_5 = CZ_mat.CZelevator_tail.axes5.value;
CZ_elevator_tail_6 = CZ_mat.CZelevator_tail.axes6.value;
CZ_elevator_tail_7 = CZ_mat.CZelevator_tail.axes7.value;

% Flap Wing
CZ_flap_wing_mat = CZ_mat.CZflap_wing;
CZ_flap_wing_val = CZ_mat.CZflap_wing.value;
CZ_flap_wing_1 = CZ_mat.CZflap_wing.axes1.value;
CZ_flap_wing_2 = CZ_mat.CZflap_wing.axes2.value;
CZ_flap_wing_3 = CZ_mat.CZflap_wing.axes3.value;
CZ_flap_wing_4 = CZ_mat.CZflap_wing.axes4.value;
CZ_flap_wing_5 = CZ_mat.CZflap_wing.axes5.value;
CZ_flap_wing_6 = CZ_mat.CZflap_wing.axes6.value;
CZ_flap_wing_7 = CZ_mat.CZflap_wing.axes7.value;

% Rudder Tail
CZ_rudder_tail_mat = CZ_mat.CZrudder_tail;
CZ_rudder_tail_val = CZ_mat.CZrudder_tail.value;
CZ_rudder_tail_1 = CZ_mat.CZrudder_tail.axes1.value;
CZ_rudder_tail_2 = CZ_mat.CZrudder_tail.axes2.value;
CZ_rudder_tail_3 = CZ_mat.CZrudder_tail.axes3.value;
CZ_rudder_tail_4 = CZ_mat.CZrudder_tail.axes4.value;
CZ_rudder_tail_5 = CZ_mat.CZrudder_tail.axes5.value;
CZ_rudder_tail_6 = CZ_mat.CZrudder_tail.axes6.value;
CZ_rudder_tail_7 = CZ_mat.CZrudder_tail.axes7.value;

% Tail
CZ_tail_mat = CZ_mat.CZtail;
CZ_tail_val = CZ_mat.CZtail.value;
CZ_tail_1 = CZ_mat.CZtail.axes1.value;
CZ_tail_2 = CZ_mat.CZtail.axes2.value;
CZ_tail_3 = CZ_mat.CZtail.axes3.value;
CZ_tail_4 = CZ_mat.CZtail.axes4.value;
CZ_tail_5 = CZ_mat.CZtail.axes5.value;
CZ_tail_6 = CZ_mat.CZtail.axes6.value;

% Tail Damp p
CZ_tail_damp_p_mat = CZ_mat.CZtail_damp_p;
CZ_tail_damp_p_val = CZ_mat.CZtail_damp_p.value;
CZ_tail_damp_p_1 = CZ_mat.CZtail_damp_p.axes1.value;
CZ_tail_damp_p_2 = CZ_mat.CZtail_damp_p.axes2.value;
CZ_tail_damp_p_3 = CZ_mat.CZtail_damp_p.axes3.value;
CZ_tail_damp_p_4 = CZ_mat.CZtail_damp_p.axes4.value;
CZ_tail_damp_p_5 = CZ_mat.CZtail_damp_p.axes5.value;
CZ_tail_damp_p_6 = CZ_mat.CZtail_damp_p.axes6.value;

% Tail Damp q
CZ_tail_damp_q_mat = CZ_mat.CZtail_damp_q;
CZ_tail_damp_q_val = CZ_mat.CZtail_damp_q.value;
CZ_tail_damp_q_1 = CZ_mat.CZtail_damp_q.axes1.value;
CZ_tail_damp_q_2 = CZ_mat.CZtail_damp_q.axes2.value;
CZ_tail_damp_q_3 = CZ_mat.CZtail_damp_q.axes3.value;
CZ_tail_damp_q_4 = CZ_mat.CZtail_damp_q.axes4.value;
CZ_tail_damp_q_5 = CZ_mat.CZtail_damp_q.axes5.value;
CZ_tail_damp_q_6 = CZ_mat.CZtail_damp_q.axes6.value;

% Tail Damp r
CZ_tail_damp_r_mat = CZ_mat.CZtail_damp_r;
CZ_tail_damp_r_val = CZ_mat.CZtail_damp_r.value;
CZ_tail_damp_r_1 = CZ_mat.CZtail_damp_r.axes1.value;
CZ_tail_damp_r_2 = CZ_mat.CZtail_damp_r.axes2.value;
CZ_tail_damp_r_3 = CZ_mat.CZtail_damp_r.axes3.value;
CZ_tail_damp_r_4 = CZ_mat.CZtail_damp_r.axes4.value;
CZ_tail_damp_r_5 = CZ_mat.CZtail_damp_r.axes5.value;
CZ_tail_damp_r_6 = CZ_mat.CZtail_damp_r.axes6.value;

% Wing
CZ_wing_mat = CZ_mat.CZwing;
CZ_wing_val = CZ_mat.CZwing.value;
CZ_wing_1 = CZ_mat.CZwing.axes1.value;
CZ_wing_2 = CZ_mat.CZwing.axes2.value;
CZ_wing_3 = CZ_mat.CZwing.axes3.value;
CZ_wing_4 = CZ_mat.CZwing.axes4.value;
CZ_wing_5 = CZ_mat.CZwing.axes5.value;
CZ_wing_6 = CZ_mat.CZwing.axes6.value;

% Wing Damp p
CZ_wing_damp_p_mat = CZ_mat.CZwing_damp_p;
CZ_wing_damp_p_val = CZ_mat.CZwing_damp_p.value;
CZ_wing_damp_p_1 = CZ_mat.CZwing_damp_p.axes1.value;
CZ_wing_damp_p_2 = CZ_mat.CZwing_damp_p.axes2.value;
CZ_wing_damp_p_3 = CZ_mat.CZwing_damp_p.axes3.value;
CZ_wing_damp_p_4 = CZ_mat.CZwing_damp_p.axes4.value;
CZ_wing_damp_p_5 = CZ_mat.CZwing_damp_p.axes5.value;
CZ_wing_damp_p_6 = CZ_mat.CZwing_damp_p.axes6.value;

% Wing Damp q
CZ_wing_damp_q_mat = CZ_mat.CZwing_damp_q;
CZ_wing_damp_q_val = CZ_mat.CZwing_damp_q.value;
CZ_wing_damp_q_1 = CZ_mat.CZwing_damp_q.axes1.value;
CZ_wing_damp_q_2 = CZ_mat.CZwing_damp_q.axes2.value;
CZ_wing_damp_q_3 = CZ_mat.CZwing_damp_q.axes3.value;
CZ_wing_damp_q_4 = CZ_mat.CZwing_damp_q.axes4.value;
CZ_wing_damp_q_5 = CZ_mat.CZwing_damp_q.axes5.value;
CZ_wing_damp_q_6 = CZ_mat.CZwing_damp_q.axes6.value;

% Wing Damp r
CZ_wing_damp_r_mat = CZ_mat.CZwing_damp_r;
CZ_wing_damp_r_val = CZ_mat.CZwing_damp_r.value;
CZ_wing_damp_r_1 = CZ_mat.CZwing_damp_r.axes1.value;
CZ_wing_damp_r_2 = CZ_mat.CZwing_damp_r.axes2.value;
CZ_wing_damp_r_3 = CZ_mat.CZwing_damp_r.axes3.value;
CZ_wing_damp_r_4 = CZ_mat.CZwing_damp_r.axes4.value;
CZ_wing_damp_r_5 = CZ_mat.CZwing_damp_r.axes5.value;
CZ_wing_damp_r_6 = CZ_mat.CZwing_damp_r.axes6.value;

% Hover Fuse
CZ_hover_fuse_mat = CZ_mat.CZhover_fuse;
CZ_hover_fuse_val = CZ_mat.CZhover_fuse.value;
CZ_hover_fuse_1 = CZ_mat.CZhover_fuse.axes1.value;
CZ_hover_fuse_2 = CZ_mat.CZhover_fuse.axes2.value;
CZ_hover_fuse_3 = CZ_mat.CZhover_fuse.axes3.value;

%% Cl Data

Cl_mat = aero.Cl;

% Aileron Wing
Cl_aileron_wing_mat = Cl_mat.Claileron_wing;
Cl_aileron_wing_val = Cl_mat.Claileron_wing.value;
Cl_aileron_wing_1 = Cl_mat.Claileron_wing.axes1.value;
Cl_aileron_wing_2 = Cl_mat.Claileron_wing.axes2.value;
Cl_aileron_wing_3 = Cl_mat.Claileron_wing.axes3.value;
Cl_aileron_wing_4 = Cl_mat.Claileron_wing.axes4.value;
Cl_aileron_wing_5 = Cl_mat.Claileron_wing.axes5.value;
Cl_aileron_wing_6 = Cl_mat.Claileron_wing.axes6.value;
Cl_aileron_wing_7 = Cl_mat.Claileron_wing.axes7.value;

% Elevator Tail
Cl_elevator_tail_mat = Cl_mat.Clelevator_tail;
Cl_elevator_tail_val = Cl_mat.Clelevator_tail.value;
Cl_elevator_tail_1 = Cl_mat.Clelevator_tail.axes1.value;
Cl_elevator_tail_2 = Cl_mat.Clelevator_tail.axes2.value;
Cl_elevator_tail_3 = Cl_mat.Clelevator_tail.axes3.value;
Cl_elevator_tail_4 = Cl_mat.Clelevator_tail.axes4.value;
Cl_elevator_tail_5 = Cl_mat.Clelevator_tail.axes5.value;
Cl_elevator_tail_6 = Cl_mat.Clelevator_tail.axes6.value;
Cl_elevator_tail_7 = Cl_mat.Clelevator_tail.axes7.value;

% Flap Wing
Cl_flap_wing_mat = Cl_mat.Clflap_wing;
Cl_flap_wing_val = Cl_mat.Clflap_wing.value;
Cl_flap_wing_1 = Cl_mat.Clflap_wing.axes1.value;
Cl_flap_wing_2 = Cl_mat.Clflap_wing.axes2.value;
Cl_flap_wing_3 = Cl_mat.Clflap_wing.axes3.value;
Cl_flap_wing_4 = Cl_mat.Clflap_wing.axes4.value;
Cl_flap_wing_5 = Cl_mat.Clflap_wing.axes5.value;
Cl_flap_wing_6 = Cl_mat.Clflap_wing.axes6.value;
Cl_flap_wing_7 = Cl_mat.Clflap_wing.axes7.value;

% Rudder Tail
Cl_rudder_tail_mat = Cl_mat.Clrudder_tail;
Cl_rudder_tail_val = Cl_mat.Clrudder_tail.value;
Cl_rudder_tail_1 = Cl_mat.Clrudder_tail.axes1.value;
Cl_rudder_tail_2 = Cl_mat.Clrudder_tail.axes2.value;
Cl_rudder_tail_3 = Cl_mat.Clrudder_tail.axes3.value;
Cl_rudder_tail_4 = Cl_mat.Clrudder_tail.axes4.value;
Cl_rudder_tail_5 = Cl_mat.Clrudder_tail.axes5.value;
Cl_rudder_tail_6 = Cl_mat.Clrudder_tail.axes6.value;
Cl_rudder_tail_7 = Cl_mat.Clrudder_tail.axes7.value;

% Tail
Cl_tail_mat = Cl_mat.Cltail;
Cl_tail_val = Cl_mat.Cltail.value;
Cl_tail_1 = Cl_mat.Cltail.axes1.value;
Cl_tail_2 = Cl_mat.Cltail.axes2.value;
Cl_tail_3 = Cl_mat.Cltail.axes3.value;
Cl_tail_4 = Cl_mat.Cltail.axes4.value;
Cl_tail_5 = Cl_mat.Cltail.axes5.value;
Cl_tail_6 = Cl_mat.Cltail.axes6.value;

% Tail Damp p
Cl_tail_damp_p_mat = Cl_mat.Cltail_damp_p;
Cl_tail_damp_p_val = Cl_mat.Cltail_damp_p.value;
Cl_tail_damp_p_1 = Cl_mat.Cltail_damp_p.axes1.value;
Cl_tail_damp_p_2 = Cl_mat.Cltail_damp_p.axes2.value;
Cl_tail_damp_p_3 = Cl_mat.Cltail_damp_p.axes3.value;
Cl_tail_damp_p_4 = Cl_mat.Cltail_damp_p.axes4.value;
Cl_tail_damp_p_5 = Cl_mat.Cltail_damp_p.axes5.value;
Cl_tail_damp_p_6 = Cl_mat.Cltail_damp_p.axes6.value;

% Tail Damp q
Cl_tail_damp_q_mat = Cl_mat.Cltail_damp_q;
Cl_tail_damp_q_val = Cl_mat.Cltail_damp_q.value;
Cl_tail_damp_q_1 = Cl_mat.Cltail_damp_q.axes1.value;
Cl_tail_damp_q_2 = Cl_mat.Cltail_damp_q.axes2.value;
Cl_tail_damp_q_3 = Cl_mat.Cltail_damp_q.axes3.value;
Cl_tail_damp_q_4 = Cl_mat.Cltail_damp_q.axes4.value;
Cl_tail_damp_q_5 = Cl_mat.Cltail_damp_q.axes5.value;
Cl_tail_damp_q_6 = Cl_mat.Cltail_damp_q.axes6.value;

% Tail Damp r
Cl_tail_damp_r_mat = Cl_mat.Cltail_damp_r;
Cl_tail_damp_r_val = Cl_mat.Cltail_damp_r.value;
Cl_tail_damp_r_1 = Cl_mat.Cltail_damp_r.axes1.value;
Cl_tail_damp_r_2 = Cl_mat.Cltail_damp_r.axes2.value;
Cl_tail_damp_r_3 = Cl_mat.Cltail_damp_r.axes3.value;
Cl_tail_damp_r_4 = Cl_mat.Cltail_damp_r.axes4.value;
Cl_tail_damp_r_5 = Cl_mat.Cltail_damp_r.axes5.value;
Cl_tail_damp_r_6 = Cl_mat.Cltail_damp_r.axes6.value;

% Wing
Cl_wing_mat = Cl_mat.Clwing;
Cl_wing_val = Cl_mat.Clwing.value;
Cl_wing_1 = Cl_mat.Clwing.axes1.value;
Cl_wing_2 = Cl_mat.Clwing.axes2.value;
Cl_wing_3 = Cl_mat.Clwing.axes3.value;
Cl_wing_4 = Cl_mat.Clwing.axes4.value;
Cl_wing_5 = Cl_mat.Clwing.axes5.value;
Cl_wing_6 = Cl_mat.Clwing.axes6.value;

% Wing Damp p
Cl_wing_damp_p_mat = Cl_mat.Clwing_damp_p;
Cl_wing_damp_p_val = Cl_mat.Clwing_damp_p.value;
Cl_wing_damp_p_1 = Cl_mat.Clwing_damp_p.axes1.value;
Cl_wing_damp_p_2 = Cl_mat.Clwing_damp_p.axes2.value;
Cl_wing_damp_p_3 = Cl_mat.Clwing_damp_p.axes3.value;
Cl_wing_damp_p_4 = Cl_mat.Clwing_damp_p.axes4.value;
Cl_wing_damp_p_5 = Cl_mat.Clwing_damp_p.axes5.value;
Cl_wing_damp_p_6 = Cl_mat.Clwing_damp_p.axes6.value;

% Wing Damp q
Cl_wing_damp_q_mat = Cl_mat.Clwing_damp_q;
Cl_wing_damp_q_val = Cl_mat.Clwing_damp_q.value;
Cl_wing_damp_q_1 = Cl_mat.Clwing_damp_q.axes1.value;
Cl_wing_damp_q_2 = Cl_mat.Clwing_damp_q.axes2.value;
Cl_wing_damp_q_3 = Cl_mat.Clwing_damp_q.axes3.value;
Cl_wing_damp_q_4 = Cl_mat.Clwing_damp_q.axes4.value;
Cl_wing_damp_q_5 = Cl_mat.Clwing_damp_q.axes5.value;
Cl_wing_damp_q_6 = Cl_mat.Clwing_damp_q.axes6.value;

% Wing Damp r
Cl_wing_damp_r_mat = Cl_mat.Clwing_damp_r;
Cl_wing_damp_r_val = Cl_mat.Clwing_damp_r.value;
Cl_wing_damp_r_1 = Cl_mat.Clwing_damp_r.axes1.value;
Cl_wing_damp_r_2 = Cl_mat.Clwing_damp_r.axes2.value;
Cl_wing_damp_r_3 = Cl_mat.Clwing_damp_r.axes3.value;
Cl_wing_damp_r_4 = Cl_mat.Clwing_damp_r.axes4.value;
Cl_wing_damp_r_5 = Cl_mat.Clwing_damp_r.axes5.value;
Cl_wing_damp_r_6 = Cl_mat.Clwing_damp_r.axes6.value;

% Hover Fuse
Cl_hover_fuse_mat = Cl_mat.Clhover_fuse;
Cl_hover_fuse_val = Cl_mat.Clhover_fuse.value;
Cl_hover_fuse_1 = Cl_mat.Clhover_fuse.axes1.value;
Cl_hover_fuse_2 = Cl_mat.Clhover_fuse.axes2.value;
Cl_hover_fuse_3 = Cl_mat.Clhover_fuse.axes3.value;

%% Cm Data

Cm_mat = aero.Cm;

% Aileron Wing
Cm_aileron_wing_mat = Cm_mat.Cmaileron_wing;
Cm_aileron_wing_val = Cm_mat.Cmaileron_wing.value;
Cm_aileron_wing_1 = Cm_mat.Cmaileron_wing.axes1.value;
Cm_aileron_wing_2 = Cm_mat.Cmaileron_wing.axes2.value;
Cm_aileron_wing_3 = Cm_mat.Cmaileron_wing.axes3.value;
Cm_aileron_wing_4 = Cm_mat.Cmaileron_wing.axes4.value;
Cm_aileron_wing_5 = Cm_mat.Cmaileron_wing.axes5.value;
Cm_aileron_wing_6 = Cm_mat.Cmaileron_wing.axes6.value;
Cm_aileron_wing_7 = Cm_mat.Cmaileron_wing.axes7.value;

% Elevator Tail
Cm_elevator_tail_mat = Cm_mat.Cmelevator_tail;
Cm_elevator_tail_val = Cm_mat.Cmelevator_tail.value;
Cm_elevator_tail_1 = Cm_mat.Cmelevator_tail.axes1.value;
Cm_elevator_tail_2 = Cm_mat.Cmelevator_tail.axes2.value;
Cm_elevator_tail_3 = Cm_mat.Cmelevator_tail.axes3.value;
Cm_elevator_tail_4 = Cm_mat.Cmelevator_tail.axes4.value;
Cm_elevator_tail_5 = Cm_mat.Cmelevator_tail.axes5.value;
Cm_elevator_tail_6 = Cm_mat.Cmelevator_tail.axes6.value;
Cm_elevator_tail_7 = Cm_mat.Cmelevator_tail.axes7.value;

% Flap Wing
Cm_flap_wing_mat = Cm_mat.Cmflap_wing;
Cm_flap_wing_val = Cm_mat.Cmflap_wing.value;
Cm_flap_wing_1 = Cm_mat.Cmflap_wing.axes1.value;
Cm_flap_wing_2 = Cm_mat.Cmflap_wing.axes2.value;
Cm_flap_wing_3 = Cm_mat.Cmflap_wing.axes3.value;
Cm_flap_wing_4 = Cm_mat.Cmflap_wing.axes4.value;
Cm_flap_wing_5 = Cm_mat.Cmflap_wing.axes5.value;
Cm_flap_wing_6 = Cm_mat.Cmflap_wing.axes6.value;
Cm_flap_wing_7 = Cm_mat.Cmflap_wing.axes7.value;

% Rudder Tail
Cm_rudder_tail_mat = Cm_mat.Cmrudder_tail;
Cm_rudder_tail_val = Cm_mat.Cmrudder_tail.value;
Cm_rudder_tail_1 = Cm_mat.Cmrudder_tail.axes1.value;
Cm_rudder_tail_2 = Cm_mat.Cmrudder_tail.axes2.value;
Cm_rudder_tail_3 = Cm_mat.Cmrudder_tail.axes3.value;
Cm_rudder_tail_4 = Cm_mat.Cmrudder_tail.axes4.value;
Cm_rudder_tail_5 = Cm_mat.Cmrudder_tail.axes5.value;
Cm_rudder_tail_6 = Cm_mat.Cmrudder_tail.axes6.value;
Cm_rudder_tail_7 = Cm_mat.Cmrudder_tail.axes7.value;

% Tail
Cm_tail_mat = Cm_mat.Cmtail;
Cm_tail_val = Cm_mat.Cmtail.value;
Cm_tail_1 = Cm_mat.Cmtail.axes1.value;
Cm_tail_2 = Cm_mat.Cmtail.axes2.value;
Cm_tail_3 = Cm_mat.Cmtail.axes3.value;
Cm_tail_4 = Cm_mat.Cmtail.axes4.value;
Cm_tail_5 = Cm_mat.Cmtail.axes5.value;
Cm_tail_6 = Cm_mat.Cmtail.axes6.value;

% Tail Damp p
Cm_tail_damp_p_mat = Cm_mat.Cmtail_damp_p;
Cm_tail_damp_p_val = Cm_mat.Cmtail_damp_p.value;
Cm_tail_damp_p_1 = Cm_mat.Cmtail_damp_p.axes1.value;
Cm_tail_damp_p_2 = Cm_mat.Cmtail_damp_p.axes2.value;
Cm_tail_damp_p_3 = Cm_mat.Cmtail_damp_p.axes3.value;
Cm_tail_damp_p_4 = Cm_mat.Cmtail_damp_p.axes4.value;
Cm_tail_damp_p_5 = Cm_mat.Cmtail_damp_p.axes5.value;
Cm_tail_damp_p_6 = Cm_mat.Cmtail_damp_p.axes6.value;

% Tail Damp q
Cm_tail_damp_q_mat = Cm_mat.Cmtail_damp_q;
Cm_tail_damp_q_val = Cm_mat.Cmtail_damp_q.value;
Cm_tail_damp_q_1 = Cm_mat.Cmtail_damp_q.axes1.value;
Cm_tail_damp_q_2 = Cm_mat.Cmtail_damp_q.axes2.value;
Cm_tail_damp_q_3 = Cm_mat.Cmtail_damp_q.axes3.value;
Cm_tail_damp_q_4 = Cm_mat.Cmtail_damp_q.axes4.value;
Cm_tail_damp_q_5 = Cm_mat.Cmtail_damp_q.axes5.value;
Cm_tail_damp_q_6 = Cm_mat.Cmtail_damp_q.axes6.value;

% Tail Damp r
Cm_tail_damp_r_mat = Cm_mat.Cmtail_damp_r;
Cm_tail_damp_r_val = Cm_mat.Cmtail_damp_r.value;
Cm_tail_damp_r_1 = Cm_mat.Cmtail_damp_r.axes1.value;
Cm_tail_damp_r_2 = Cm_mat.Cmtail_damp_r.axes2.value;
Cm_tail_damp_r_3 = Cm_mat.Cmtail_damp_r.axes3.value;
Cm_tail_damp_r_4 = Cm_mat.Cmtail_damp_r.axes4.value;
Cm_tail_damp_r_5 = Cm_mat.Cmtail_damp_r.axes5.value;
Cm_tail_damp_r_6 = Cm_mat.Cmtail_damp_r.axes6.value;

% Wing
Cm_wing_mat = Cm_mat.Cmwing;
Cm_wing_val = Cm_mat.Cmwing.value;
Cm_wing_1 = Cm_mat.Cmwing.axes1.value;
Cm_wing_2 = Cm_mat.Cmwing.axes2.value;
Cm_wing_3 = Cm_mat.Cmwing.axes3.value;
Cm_wing_4 = Cm_mat.Cmwing.axes4.value;
Cm_wing_5 = Cm_mat.Cmwing.axes5.value;
Cm_wing_6 = Cm_mat.Cmwing.axes6.value;

% Wing Damp p
Cm_wing_damp_p_mat = Cm_mat.Cmwing_damp_p;
Cm_wing_damp_p_val = Cm_mat.Cmwing_damp_p.value;
Cm_wing_damp_p_1 = Cm_mat.Cmwing_damp_p.axes1.value;
Cm_wing_damp_p_2 = Cm_mat.Cmwing_damp_p.axes2.value;
Cm_wing_damp_p_3 = Cm_mat.Cmwing_damp_p.axes3.value;
Cm_wing_damp_p_4 = Cm_mat.Cmwing_damp_p.axes4.value;
Cm_wing_damp_p_5 = Cm_mat.Cmwing_damp_p.axes5.value;
Cm_wing_damp_p_6 = Cm_mat.Cmwing_damp_p.axes6.value;

% Wing Damp q
Cm_wing_damp_q_mat = Cm_mat.Cmwing_damp_q;
Cm_wing_damp_q_val = Cm_mat.Cmwing_damp_q.value;
Cm_wing_damp_q_1 = Cm_mat.Cmwing_damp_q.axes1.value;
Cm_wing_damp_q_2 = Cm_mat.Cmwing_damp_q.axes2.value;
Cm_wing_damp_q_3 = Cm_mat.Cmwing_damp_q.axes3.value;
Cm_wing_damp_q_4 = Cm_mat.Cmwing_damp_q.axes4.value;
Cm_wing_damp_q_5 = Cm_mat.Cmwing_damp_q.axes5.value;
Cm_wing_damp_q_6 = Cm_mat.Cmwing_damp_q.axes6.value;

% Wing Damp r
Cm_wing_damp_r_mat = Cm_mat.Cmwing_damp_r;
Cm_wing_damp_r_val = Cm_mat.Cmwing_damp_r.value;
Cm_wing_damp_r_1 = Cm_mat.Cmwing_damp_r.axes1.value;
Cm_wing_damp_r_2 = Cm_mat.Cmwing_damp_r.axes2.value;
Cm_wing_damp_r_3 = Cm_mat.Cmwing_damp_r.axes3.value;
Cm_wing_damp_r_4 = Cm_mat.Cmwing_damp_r.axes4.value;
Cm_wing_damp_r_5 = Cm_mat.Cmwing_damp_r.axes5.value;
Cm_wing_damp_r_6 = Cm_mat.Cmwing_damp_r.axes6.value;

% Hover Fuse
Cm_hover_fuse_mat = Cm_mat.Cmhover_fuse;
Cm_hover_fuse_val = Cm_mat.Cmhover_fuse.value;
Cm_hover_fuse_1 = Cm_mat.Cmhover_fuse.axes1.value;
Cm_hover_fuse_2 = Cm_mat.Cmhover_fuse.axes2.value;
Cm_hover_fuse_3 = Cm_mat.Cmhover_fuse.axes3.value;

%% Cn Data

Cn_mat = aero.Cn;

% Aileron Wing
Cn_aileron_wing_mat = Cn_mat.Cnaileron_wing;
Cn_aileron_wing_val = Cn_mat.Cnaileron_wing.value;
Cn_aileron_wing_1 = Cn_mat.Cnaileron_wing.axes1.value;
Cn_aileron_wing_2 = Cn_mat.Cnaileron_wing.axes2.value;
Cn_aileron_wing_3 = Cn_mat.Cnaileron_wing.axes3.value;
Cn_aileron_wing_4 = Cn_mat.Cnaileron_wing.axes4.value;
Cn_aileron_wing_5 = Cn_mat.Cnaileron_wing.axes5.value;
Cn_aileron_wing_6 = Cn_mat.Cnaileron_wing.axes6.value;
Cn_aileron_wing_7 = Cn_mat.Cnaileron_wing.axes7.value;

% Elevator Tail
Cn_elevator_tail_mat = Cn_mat.Cnelevator_tail;
Cn_elevator_tail_val = Cn_mat.Cnelevator_tail.value;
Cn_elevator_tail_1 = Cn_mat.Cnelevator_tail.axes1.value;
Cn_elevator_tail_2 = Cn_mat.Cnelevator_tail.axes2.value;
Cn_elevator_tail_3 = Cn_mat.Cnelevator_tail.axes3.value;
Cn_elevator_tail_4 = Cn_mat.Cnelevator_tail.axes4.value;
Cn_elevator_tail_5 = Cn_mat.Cnelevator_tail.axes5.value;
Cn_elevator_tail_6 = Cn_mat.Cnelevator_tail.axes6.value;
Cn_elevator_tail_7 = Cn_mat.Cnelevator_tail.axes7.value;

% Flap Wing
Cn_flap_wing_mat = Cn_mat.Cnflap_wing;
Cn_flap_wing_val = Cn_mat.Cnflap_wing.value;
Cn_flap_wing_1 = Cn_mat.Cnflap_wing.axes1.value;
Cn_flap_wing_2 = Cn_mat.Cnflap_wing.axes2.value;
Cn_flap_wing_3 = Cn_mat.Cnflap_wing.axes3.value;
Cn_flap_wing_4 = Cn_mat.Cnflap_wing.axes4.value;
Cn_flap_wing_5 = Cn_mat.Cnflap_wing.axes5.value;
Cn_flap_wing_6 = Cn_mat.Cnflap_wing.axes6.value;
Cn_flap_wing_7 = Cn_mat.Cnflap_wing.axes7.value;

% Rudder Tail
Cn_rudder_tail_mat = Cn_mat.Cnrudder_tail;
Cn_rudder_tail_val = Cn_mat.Cnrudder_tail.value;
Cn_rudder_tail_1 = Cn_mat.Cnrudder_tail.axes1.value;
Cn_rudder_tail_2 = Cn_mat.Cnrudder_tail.axes2.value;
Cn_rudder_tail_3 = Cn_mat.Cnrudder_tail.axes3.value;
Cn_rudder_tail_4 = Cn_mat.Cnrudder_tail.axes4.value;
Cn_rudder_tail_5 = Cn_mat.Cnrudder_tail.axes5.value;
Cn_rudder_tail_6 = Cn_mat.Cnrudder_tail.axes6.value;
Cn_rudder_tail_7 = Cn_mat.Cnrudder_tail.axes7.value;

% Tail
Cn_tail_mat = Cn_mat.Cntail;
Cn_tail_val = Cn_mat.Cntail.value;
Cn_tail_1 = Cn_mat.Cntail.axes1.value;
Cn_tail_2 = Cn_mat.Cntail.axes2.value;
Cn_tail_3 = Cn_mat.Cntail.axes3.value;
Cn_tail_4 = Cn_mat.Cntail.axes4.value;
Cn_tail_5 = Cn_mat.Cntail.axes5.value;
Cn_tail_6 = Cn_mat.Cntail.axes6.value;

% Tail Damp p
Cn_tail_damp_p_mat = Cn_mat.Cntail_damp_p;
Cn_tail_damp_p_val = Cn_mat.Cntail_damp_p.value;
Cn_tail_damp_p_1 = Cn_mat.Cntail_damp_p.axes1.value;
Cn_tail_damp_p_2 = Cn_mat.Cntail_damp_p.axes2.value;
Cn_tail_damp_p_3 = Cn_mat.Cntail_damp_p.axes3.value;
Cn_tail_damp_p_4 = Cn_mat.Cntail_damp_p.axes4.value;
Cn_tail_damp_p_5 = Cn_mat.Cntail_damp_p.axes5.value;
Cn_tail_damp_p_6 = Cn_mat.Cntail_damp_p.axes6.value;

% Tail Damp q
Cn_tail_damp_q_mat = Cn_mat.Cntail_damp_q;
Cn_tail_damp_q_val = Cn_mat.Cntail_damp_q.value;
Cn_tail_damp_q_1 = Cn_mat.Cntail_damp_q.axes1.value;
Cn_tail_damp_q_2 = Cn_mat.Cntail_damp_q.axes2.value;
Cn_tail_damp_q_3 = Cn_mat.Cntail_damp_q.axes3.value;
Cn_tail_damp_q_4 = Cn_mat.Cntail_damp_q.axes4.value;
Cn_tail_damp_q_5 = Cn_mat.Cntail_damp_q.axes5.value;
Cn_tail_damp_q_6 = Cn_mat.Cntail_damp_q.axes6.value;

% Tail Damp r
Cn_tail_damp_r_mat = Cn_mat.Cntail_damp_r;
Cn_tail_damp_r_val = Cn_mat.Cntail_damp_r.value;
Cn_tail_damp_r_1 = Cn_mat.Cntail_damp_r.axes1.value;
Cn_tail_damp_r_2 = Cn_mat.Cntail_damp_r.axes2.value;
Cn_tail_damp_r_3 = Cn_mat.Cntail_damp_r.axes3.value;
Cn_tail_damp_r_4 = Cn_mat.Cntail_damp_r.axes4.value;
Cn_tail_damp_r_5 = Cn_mat.Cntail_damp_r.axes5.value;
Cn_tail_damp_r_6 = Cn_mat.Cntail_damp_r.axes6.value;

% Wing
Cn_wing_mat = Cn_mat.Cnwing;
Cn_wing_val = Cn_mat.Cnwing.value;
Cn_wing_1 = Cn_mat.Cnwing.axes1.value;
Cn_wing_2 = Cn_mat.Cnwing.axes2.value;
Cn_wing_3 = Cn_mat.Cnwing.axes3.value;
Cn_wing_4 = Cn_mat.Cnwing.axes4.value;
Cn_wing_5 = Cn_mat.Cnwing.axes5.value;
Cn_wing_6 = Cn_mat.Cnwing.axes6.value;

% Wing Damp p
Cn_wing_damp_p_mat = Cn_mat.Cnwing_damp_p;
Cn_wing_damp_p_val = Cn_mat.Cnwing_damp_p.value;
Cn_wing_damp_p_1 = Cn_mat.Cnwing_damp_p.axes1.value;
Cn_wing_damp_p_2 = Cn_mat.Cnwing_damp_p.axes2.value;
Cn_wing_damp_p_3 = Cn_mat.Cnwing_damp_p.axes3.value;
Cn_wing_damp_p_4 = Cn_mat.Cnwing_damp_p.axes4.value;
Cn_wing_damp_p_5 = Cn_mat.Cnwing_damp_p.axes5.value;
Cn_wing_damp_p_6 = Cn_mat.Cnwing_damp_p.axes6.value;

% Wing Damp q
Cn_wing_damp_q_mat = Cn_mat.Cnwing_damp_q;
Cn_wing_damp_q_val = Cn_mat.Cnwing_damp_q.value;
Cn_wing_damp_q_1 = Cn_mat.Cnwing_damp_q.axes1.value;
Cn_wing_damp_q_2 = Cn_mat.Cnwing_damp_q.axes2.value;
Cn_wing_damp_q_3 = Cn_mat.Cnwing_damp_q.axes3.value;
Cn_wing_damp_q_4 = Cn_mat.Cnwing_damp_q.axes4.value;
Cn_wing_damp_q_5 = Cn_mat.Cnwing_damp_q.axes5.value;
Cn_wing_damp_q_6 = Cn_mat.Cnwing_damp_q.axes6.value;

% Wing Damp r
Cn_wing_damp_r_mat = Cn_mat.Cnwing_damp_r;
Cn_wing_damp_r_val = Cn_mat.Cnwing_damp_r.value;
Cn_wing_damp_r_1 = Cn_mat.Cnwing_damp_r.axes1.value;
Cn_wing_damp_r_2 = Cn_mat.Cnwing_damp_r.axes2.value;
Cn_wing_damp_r_3 = Cn_mat.Cnwing_damp_r.axes3.value;
Cn_wing_damp_r_4 = Cn_mat.Cnwing_damp_r.axes4.value;
Cn_wing_damp_r_5 = Cn_mat.Cnwing_damp_r.axes5.value;
Cn_wing_damp_r_6 = Cn_mat.Cnwing_damp_r.axes6.value;

% Hover Fuse
Cn_hover_fuse_mat = Cn_mat.Cnhover_fuse;
Cn_hover_fuse_val = Cn_mat.Cnhover_fuse.value;
Cn_hover_fuse_1 = Cn_mat.Cnhover_fuse.axes1.value;
Cn_hover_fuse_2 = Cn_mat.Cnhover_fuse.axes2.value;
Cn_hover_fuse_3 = Cn_mat.Cnhover_fuse.axes3.value;

%% Ch Data

% Not used

%% Ct Data

Ct_mat = aero.Ct;

% Tail Left
Ct_tail_left_mat = Ct_mat.Ct_tailLeft;
Ct_tail_left_val = Ct_mat.Ct_tailLeft.value;
Ct_tail_left_1 = Ct_mat.Ct_tailLeft.axes1.value;
Ct_tail_left_2 = Ct_mat.Ct_tailLeft.axes2.value;
Ct_tail_left_3 = Ct_mat.Ct_tailLeft.axes3.value;
Ct_tail_left_4 = Ct_mat.Ct_tailLeft.axes4.value;

% Tail Right
Ct_tail_right_mat = Ct_mat.Ct_tailRight;
Ct_tail_right_val = Ct_mat.Ct_tailRight.value;
Ct_tail_right_1 = Ct_mat.Ct_tailRight.axes1.value;
Ct_tail_right_2 = Ct_mat.Ct_tailRight.axes2.value;
Ct_tail_right_3 = Ct_mat.Ct_tailRight.axes3.value;
Ct_tail_right_4 = Ct_mat.Ct_tailRight.axes4.value;

% Left Out
Ct_left_out_mat = Ct_mat.Ct_leftOut1;
Ct_left_out_val = Ct_mat.Ct_leftOut1.value;
Ct_left_out_1 = Ct_mat.Ct_leftOut1.axes1.value;
Ct_left_out_2 = Ct_mat.Ct_leftOut1.axes2.value;
Ct_left_out_3 = Ct_mat.Ct_leftOut1.axes3.value;
Ct_left_out_4 = Ct_mat.Ct_leftOut1.axes4.value;

% Left 2
Ct_left_2_mat = Ct_mat.Ct_left2;
Ct_left_2_val = Ct_mat.Ct_left2.value;
Ct_left_2_1 = Ct_mat.Ct_left2.axes1.value;
Ct_left_2_2 = Ct_mat.Ct_left2.axes2.value;
Ct_left_2_3 = Ct_mat.Ct_left2.axes3.value;
Ct_left_2_4 = Ct_mat.Ct_left2.axes4.value;

% Left 3
Ct_left_3_mat = Ct_mat.Ct_left3;
Ct_left_3_val = Ct_mat.Ct_left3.value;
Ct_left_3_1 = Ct_mat.Ct_left3.axes1.value;
Ct_left_3_2 = Ct_mat.Ct_left3.axes2.value;
Ct_left_3_3 = Ct_mat.Ct_left3.axes3.value;
Ct_left_3_4 = Ct_mat.Ct_left3.axes4.value;

% Left 4
Ct_left_4_mat = Ct_mat.Ct_left4;
Ct_left_4_val = Ct_mat.Ct_left4.value;
Ct_left_4_1 = Ct_mat.Ct_left4.axes1.value;
Ct_left_4_2 = Ct_mat.Ct_left4.axes2.value;
Ct_left_4_3 = Ct_mat.Ct_left4.axes3.value;
Ct_left_4_4 = Ct_mat.Ct_left4.axes4.value;

% Left 5
Ct_left_5_mat = Ct_mat.Ct_left5;
Ct_left_5_val = Ct_mat.Ct_left5.value;
Ct_left_5_1 = Ct_mat.Ct_left5.axes1.value;
Ct_left_5_2 = Ct_mat.Ct_left5.axes2.value;
Ct_left_5_3 = Ct_mat.Ct_left5.axes3.value;
Ct_left_5_4 = Ct_mat.Ct_left5.axes4.value;

% Left 6 In
Ct_left_6_in_mat = Ct_mat.Ct_left6In;
Ct_left_6_in_val = Ct_mat.Ct_left6In.value;
Ct_left_6_in_1 = Ct_mat.Ct_left6In.axes1.value;
Ct_left_6_in_2 = Ct_mat.Ct_left6In.axes2.value;
Ct_left_6_in_3 = Ct_mat.Ct_left6In.axes3.value;
Ct_left_6_in_4 = Ct_mat.Ct_left6In.axes4.value;

% Right 7 In
Ct_right_7_in_mat = Ct_mat.Ct_right7In;
Ct_right_7_in_val = Ct_mat.Ct_right7In.value;
Ct_right_7_in_1 = Ct_mat.Ct_right7In.axes1.value;
Ct_right_7_in_2 = Ct_mat.Ct_right7In.axes2.value;
Ct_right_7_in_3 = Ct_mat.Ct_right7In.axes3.value;
Ct_right_7_in_4 = Ct_mat.Ct_right7In.axes4.value;

% Right 8
Ct_right_8_mat = Ct_mat.Ct_right8;
Ct_right_8_val = Ct_mat.Ct_right8.value;
Ct_right_8_1 = Ct_mat.Ct_right8.axes1.value;
Ct_right_8_2 = Ct_mat.Ct_right8.axes2.value;
Ct_right_8_3 = Ct_mat.Ct_right8.axes3.value;
Ct_right_8_4 = Ct_mat.Ct_right8.axes4.value;

% Right 9
Ct_right_9_mat = Ct_mat.Ct_right9;
Ct_right_9_val = Ct_mat.Ct_right9.value;
Ct_right_9_1 = Ct_mat.Ct_right9.axes1.value;
Ct_right_9_2 = Ct_mat.Ct_right9.axes2.value;
Ct_right_9_3 = Ct_mat.Ct_right9.axes3.value;
Ct_right_9_4 = Ct_mat.Ct_right9.axes4.value;

% Right 10
Ct_right_10_mat = Ct_mat.Ct_right10;
Ct_right_10_val = Ct_mat.Ct_right10.value;
Ct_right_10_1 = Ct_mat.Ct_right10.axes1.value;
Ct_right_10_2 = Ct_mat.Ct_right10.axes2.value;
Ct_right_10_3 = Ct_mat.Ct_right10.axes3.value;
Ct_right_10_4 = Ct_mat.Ct_right10.axes4.value;

% Right 11
Ct_right_11_mat = Ct_mat.Ct_right11;
Ct_right_11_val = Ct_mat.Ct_right11.value;
Ct_right_11_1 = Ct_mat.Ct_right11.axes1.value;
Ct_right_11_2 = Ct_mat.Ct_right11.axes2.value;
Ct_right_11_3 = Ct_mat.Ct_right11.axes3.value;
Ct_right_11_4 = Ct_mat.Ct_right11.axes4.value;

% Right 12 Out
Ct_right_12_out_mat = Ct_mat.Ct_right12Out;
Ct_right_12_out_val = Ct_mat.Ct_right12Out.value;
Ct_right_12_out_1 = Ct_mat.Ct_right12Out.axes1.value;
Ct_right_12_out_2 = Ct_mat.Ct_right12Out.axes2.value;
Ct_right_12_out_3 = Ct_mat.Ct_right12Out.axes3.value;
Ct_right_12_out_4 = Ct_mat.Ct_right12Out.axes4.value;

%% Kq Data

Kq_mat = aero.Kq;

% Tail Left
Kq_tail_left_mat = Kq_mat.Kq_tailLeft;
Kq_tail_left_val = Kq_mat.Kq_tailLeft.value;
Kq_tail_left_1 = Kq_mat.Kq_tailLeft.axes1.value;
Kq_tail_left_2 = Kq_mat.Kq_tailLeft.axes2.value;
Kq_tail_left_3 = Kq_mat.Kq_tailLeft.axes3.value;
Kq_tail_left_4 = Kq_mat.Kq_tailLeft.axes4.value;

% Tail Right
Kq_tail_right_mat = Kq_mat.Kq_tailRight;
Kq_tail_right_val = Kq_mat.Kq_tailRight.value;
Kq_tail_right_1 = Kq_mat.Kq_tailRight.axes1.value;
Kq_tail_right_2 = Kq_mat.Kq_tailRight.axes2.value;
Kq_tail_right_3 = Kq_mat.Kq_tailRight.axes3.value;
Kq_tail_right_4 = Kq_mat.Kq_tailRight.axes4.value;

% Left Out
Kq_left_out_mat = Kq_mat.Kq_leftOut1;
Kq_left_out_val = Kq_mat.Kq_leftOut1.value;
Kq_left_out_1 = Kq_mat.Kq_leftOut1.axes1.value;
Kq_left_out_2 = Kq_mat.Kq_leftOut1.axes2.value;
Kq_left_out_3 = Kq_mat.Kq_leftOut1.axes3.value;
Kq_left_out_4 = Kq_mat.Kq_leftOut1.axes4.value;

% Left 2
Kq_left_2_mat = Kq_mat.Kq_left2;
Kq_left_2_val = Kq_mat.Kq_left2.value;
Kq_left_2_1 = Kq_mat.Kq_left2.axes1.value;
Kq_left_2_2 = Kq_mat.Kq_left2.axes2.value;
Kq_left_2_3 = Kq_mat.Kq_left2.axes3.value;
Kq_left_2_4 = Kq_mat.Kq_left2.axes4.value;

% Left 3
Kq_left_3_mat = Kq_mat.Kq_left3;
Kq_left_3_val = Kq_mat.Kq_left3.value;
Kq_left_3_1 = Kq_mat.Kq_left3.axes1.value;
Kq_left_3_2 = Kq_mat.Kq_left3.axes2.value;
Kq_left_3_3 = Kq_mat.Kq_left3.axes3.value;
Kq_left_3_4 = Kq_mat.Kq_left3.axes4.value;

% Left 4
Kq_left_4_mat = Kq_mat.Kq_left4;
Kq_left_4_val = Kq_mat.Kq_left4.value;
Kq_left_4_1 = Kq_mat.Kq_left4.axes1.value;
Kq_left_4_2 = Kq_mat.Kq_left4.axes2.value;
Kq_left_4_3 = Kq_mat.Kq_left4.axes3.value;
Kq_left_4_4 = Kq_mat.Kq_left4.axes4.value;

% Left 5
Kq_left_5_mat = Kq_mat.Kq_left5;
Kq_left_5_val = Kq_mat.Kq_left5.value;
Kq_left_5_1 = Kq_mat.Kq_left5.axes1.value;
Kq_left_5_2 = Kq_mat.Kq_left5.axes2.value;
Kq_left_5_3 = Kq_mat.Kq_left5.axes3.value;
Kq_left_5_4 = Kq_mat.Kq_left5.axes4.value;

% Left 6 In
Kq_left_6_in_mat = Kq_mat.Kq_left6In;
Kq_left_6_in_val = Kq_mat.Kq_left6In.value;
Kq_left_6_in_1 = Kq_mat.Kq_left6In.axes1.value;
Kq_left_6_in_2 = Kq_mat.Kq_left6In.axes2.value;
Kq_left_6_in_3 = Kq_mat.Kq_left6In.axes3.value;
Kq_left_6_in_4 = Kq_mat.Kq_left6In.axes4.value;

% Right 7 In
Kq_right_7_in_mat = Kq_mat.Kq_right7In;
Kq_right_7_in_val = Kq_mat.Kq_right7In.value;
Kq_right_7_in_1 = Kq_mat.Kq_right7In.axes1.value;
Kq_right_7_in_2 = Kq_mat.Kq_right7In.axes2.value;
Kq_right_7_in_3 = Kq_mat.Kq_right7In.axes3.value;
Kq_right_7_in_4 = Kq_mat.Kq_right7In.axes4.value;

% Right 8
Kq_right_8_mat = Kq_mat.Kq_right8;
Kq_right_8_val = Kq_mat.Kq_right8.value;
Kq_right_8_1 = Kq_mat.Kq_right8.axes1.value;
Kq_right_8_2 = Kq_mat.Kq_right8.axes2.value;
Kq_right_8_3 = Kq_mat.Kq_right8.axes3.value;
Kq_right_8_4 = Kq_mat.Kq_right8.axes4.value;

% Right 9
Kq_right_9_mat = Kq_mat.Kq_right9;
Kq_right_9_val = Kq_mat.Kq_right9.value;
Kq_right_9_1 = Kq_mat.Kq_right9.axes1.value;
Kq_right_9_2 = Kq_mat.Kq_right9.axes2.value;
Kq_right_9_3 = Kq_mat.Kq_right9.axes3.value;
Kq_right_9_4 = Kq_mat.Kq_right9.axes4.value;

% Right 10
Kq_right_10_mat = Kq_mat.Kq_right10;
Kq_right_10_val = Kq_mat.Kq_right10.value;
Kq_right_10_1 = Kq_mat.Kq_right10.axes1.value;
Kq_right_10_2 = Kq_mat.Kq_right10.axes2.value;
Kq_right_10_3 = Kq_mat.Kq_right10.axes3.value;
Kq_right_10_4 = Kq_mat.Kq_right10.axes4.value;

% Right 11
Kq_right_11_mat = Kq_mat.Kq_right11;
Kq_right_11_val = Kq_mat.Kq_right11.value;
Kq_right_11_1 = Kq_mat.Kq_right11.axes1.value;
Kq_right_11_2 = Kq_mat.Kq_right11.axes2.value;
Kq_right_11_3 = Kq_mat.Kq_right11.axes3.value;
Kq_right_11_4 = Kq_mat.Kq_right11.axes4.value;

% Right 12 Out
Kq_right_12_out_mat = Kq_mat.Kq_right12Out;
Kq_right_12_out_val = Kq_mat.Kq_right12Out.value;
Kq_right_12_out_1 = Kq_mat.Kq_right12Out.axes1.value;
Kq_right_12_out_2 = Kq_mat.Kq_right12Out.axes2.value;
Kq_right_12_out_3 = Kq_mat.Kq_right12Out.axes3.value;
Kq_right_12_out_4 = Kq_mat.Kq_right12Out.axes4.value;
