function f_dyn = Basic_Model_Arm_CasADi()
import casadi.*

%% Define CasADi variables

% Controls
t1 = SX.sym('t1'); 
t2 = SX.sym('t2'); 
Tau = [t1; t2];

% States
th1  = SX.sym('th1'); dth1  = SX.sym('dth1');
th2  = SX.sym('th2'); dth2  = SX.sym('dth2');

q   = [th1; th2];
dq  = [dth1; dth2];

% Parameters
m1 = SX.sym('m1'); m2 = SX.sym('m2');
L1 = SX.sym('L1'); L2 = SX.sym('L2');
I1 = SX.sym('I1'); I2 = SX.sym('I2');
g  = SX.sym('g');
params = [m1;m2;L1;L2;I1;I2;g];

%% Rotation matrices
R01 = Rotz(th1);
R10 = R01'; 
R12 = Rotz(th2);
R21 = R12';

R02 = Rotz(th1+th2);
R20 = R02';

%% Position vectors
r1_1 = [L1/2; 0];
r1_0 = R10*r1_1;

r2_2 = [L2/2; 0];
r2_0 = R10*[L1;0] + R20*r2_2;

%% Velocities
dr1 = jacobian(r1_0, q)*dq;
dr2 = jacobian(r2_0, q)*dq;

%% Angular velocities
w10_1 = dth1;
w20_2 = dth1 + dth2;

%% Kinetic energy
T1 = 0.5*m1*(dr1.'*dr1) + 0.5*(w10_1.'*I1*w10_1);
T2 = 0.5*m2*(dr2.'*dr2) + 0.5*(w20_2.'*I2*w20_2);
Ttot = simplify(T1+T2);

%% Potential energy
V1 = m1*g*r1_0(2);
V2 = m2*g*r2_0(2);
Vtot = simplify(V1+V2);

%% Mass matrix
M = hessian(Ttot, dq);

%% Gravity
G = jacobian(Vtot, q).';

%% Coriolis/centrifugal terms
dM = SX.zeros(length(M), length(M));
for i=1:length(M)
    for j=1:length(M)
        dM(i,j) = jacobian(M(i,j), q)*dq;
    end
end
C = dM*dq - (jacobian(Ttot,q)).';

%% Equations of motion
EOM = M*[SX.sym('ddth1');SX.sym('ddth2')] + C + G - Tau;

% Solve for ddq
ddq_sol = M \ (Tau - C - G);

%% Define CasADi function (state-space form: [q;dq] -> [dq;ddq])
x = [ th1;
      dth1;
      th2;
      dth2 ];

xdot = [ dth1;
         ddq_sol(1);
         dth2;
         ddq_sol(2) ];

f_dyn = Function('f_dyn', {x, Tau, params}, {xdot});
end

%% Helper Functions
function R = Rotz(th)
    R = [cos(th),  sin(th);
        -sin(th), cos(th)];
end
