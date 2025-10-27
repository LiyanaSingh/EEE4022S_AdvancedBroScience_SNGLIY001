function f_dyn = Basic_Model_Arm3D_CasADi()
import casadi.*

%  3DOF 3D Arm Model
% 
%    th1 = shoulder yaw (Z)
%    th2 = shoulder pitch (Y)
%    th3 = elbow pitch (Y)
%
%  State vector: x = [th1; dth1; th2; dth2; th3; dth3]
%  Control: Tau = [t1; t2; t3]
%  Parameters: [m1; m2; L1; L2; I1(3x3); I2(3x3); g]

%% --------------------------- Symbolic Variables ---------------------------
% Controls
t1 = SX.sym('t1'); t2 = SX.sym('t2'); t3 = SX.sym('t3');
Tau = [t1; t2; t3];

% States
th1 = SX.sym('th1'); dth1 = SX.sym('dth1');
th2 = SX.sym('th2'); dth2 = SX.sym('dth2');
th3 = SX.sym('th3'); dth3 = SX.sym('dth3');
q  = [th1; th2; th3];
dq = [dth1; dth2; dth3];

% Parameters
m1 = SX.sym('m1'); m2 = SX.sym('m2');
L1 = SX.sym('L1'); L2 = SX.sym('L2');
I1 = SX.sym('I1',3,3); I2 = SX.sym('I2',3,3);
g  = SX.sym('g');
params = [m1; m2; L1; L2; reshape(I1,9,1); reshape(I2,9,1); g];

%% --------------------------- Kinematics ---------------------------
% Rotation matrices 
R01 = Roty(th2) * Rotz(th1);    % pitch then yaw
R12 = Roty(th3);                 
R02 = R01 * R12;                

% Positions of CoMs
r1_1 = [L1/2; 0; 0];            
r1_0 = R01 * r1_1;              

r_elbow_0 = R01 * [L1;0;0];     
r2_2 = [L2/2; 0; 0];            
r2_0 = r_elbow_0 + R02 * r2_2;

%% --------------------------- Velocities ---------------------------
% Linear
dr1 = jacobian(r1_0,q)*dq;
dr2 = jacobian(r2_0,q)*dq;

% Angular 
R01_vec = reshape(R01,9,1);
Rdot01  = reshape(jacobian(R01_vec,q)*dq,3,3);
w1_hat  = Rdot01*R01.';
w1      = vee(w1_hat);

R02_vec = reshape(R02,9,1);
Rdot02  = reshape(jacobian(R02_vec,q)*dq,3,3);
w2_hat  = Rdot02*R02.';
w2      = vee(w2_hat);

%% --------------------------- Energies ---------------------------
% Kinetic
T1 = 0.5*m1*(dr1.'*dr1) + 0.5*(w1.'*I1*w1);
T2 = 0.5*m2*(dr2.'*dr2) + 0.5*(w2.'*I2*w2);
Ttot = simplify(T1 + T2);

% Potential
V1 = m1*g*r1_0(3);
V2 = m2*g*r2_0(3);
Vtot = simplify(V1 + V2);

%% --------------------------- Dynamics Terms ---------------------------
% Mass matrix
M = simplify(hessian(Ttot,dq));

% Gravity vector
G = jacobian(Vtot,q).';

% Coriolis/Centrifugal matrix 
n = length(dq);
dM = SX.zeros(n,n);
for i=1:n
    for j=1:n
        dM(i,j) = jacobian(M(i,j),q)*dq;
    end
end
C = dM*dq - (jacobian(Ttot,q)).';

%% --------------------------- Accelerations ---------------------------
ddq_sol = simplify(M \ (Tau - C - G));

%% --------------------------- State-Space Dynamics ---------------------------
x    = [th1; dth1; th2; dth2; th3; dth3];
xdot = [dth1; ddq_sol(1); dth2; ddq_sol(2); dth3; ddq_sol(3)];

%% --------------------------- CasADi Function ---------------------------
f_dyn = Function('f_dyn',{x,Tau,params},{xdot});

end

%% =========================== Helper Functions ===========================
function R = Roty(th)
import casadi.*
R = [cos(th),0,-sin(th);
     0,      1, 0;
     sin(th),0, cos(th)];
end

function R = Rotz(th)
import casadi.*
R = [cos(th), sin(th), 0;
    -sin(th), cos(th), 0;
     0,       0,       1];
end

function v = vee(S)
% Converts skew-symmetric matrix to vector
v = [S(3,2); S(1,3); S(2,1)];
end
