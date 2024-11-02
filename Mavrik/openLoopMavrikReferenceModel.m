% Mavrik Reference Model
%
% This sets up the Mavrik model and runs the open-loop bare airframe
% since there's no controller

%% Pull/Define Data
setupMavrik;
mdl = 'Mavrik_Reference_model.slx';

t=[0.0];

%% Initial state
U=30; % trim speed
eulerIn= [t,[0,4*pi/180,0]]; % trim attitude
vnedIn= [t,[U*(cos(eulerIn(2))),U*sin(eulerIn(2)),0]]; % NED velocity
pqrIn= [t,[0,0,0]]; % trim rates

%% Initial Actuator settings
RPM_tailLeft = 7500;
RPM_tailRight = 7500;
RPM_leftOut1 = 7500;
RPM_left2 = 7500;
RPM_left3 = 7500;
RPM_left4 = 7500;
RPM_left5 = 7500;
RPM_left6In = 7500;
RPM_right7In = 7500;
RPM_right8 = 7500;
RPM_right9 = 7500;
RPM_right10 = 7500;
RPM_right11 = 7500;
RPM_right12Out = 7500;

wing_tilt = 0;
tail_tilt = 0;
aileron = 0;
elevator = 0;
flap = 0;
rudder = 0;

actuatorsIn = [0, wing_tilt, tail_tilt, aileron, elevator, flap, rudder, ...
    RPM_tailLeft, RPM_tailRight, RPM_leftOut1, RPM_left2, RPM_left3, ...
    RPM_left4, RPM_left5, RPM_left6In, RPM_right7In, RPM_right8, RPM_right9, ...
    RPM_right10, RPM_right11, RPM_right12Out];

actuator_values = [wing_tilt, tail_tilt, aileron, elevator, flap, rudder, ...
    RPM_tailLeft, RPM_tailRight, RPM_leftOut1, RPM_left2, RPM_left3, ...
    RPM_left4, RPM_left5, RPM_left6In, RPM_right7In, RPM_right8, RPM_right9, ...
    RPM_right10, RPM_right11, RPM_right12Out];
actuatorsIn = timeseries(actuator_values, 0);  % Creates a time series at t=0

state = [];

data = sim(mdl);

%% Extract Data

Fx_vec = data.Forces(:,1);
Fy_vec = data.Forces(:,2);
Fz_vec = data.Forces(:,3);
L_vec = data.Moments(:,1);
M_vec = data.Moments(:,2);
N_vec = data.Moments(:,3);

% state output1
state = data.yout{1}.Values.Data
