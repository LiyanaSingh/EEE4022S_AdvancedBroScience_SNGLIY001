% 2-DOF Planar Arm 
% State: [th1; dth1; th2; dth2]
% Control: tau = [tau1; tau2]
Basic_Model_Arm_CasADi;
import casadi.*

%% ----------------- Model parameters -----------------
weight = 67.8; 
m1 = weight*30.695/1000;   m2 = weight*17.035/1000;    % masses [kg]
l1 = 0.3012;   l2 = 0.2342;  % link lengths [m]

% Inertia about the center of each link
I1 = (1/12) * m1 * l1^2;
I2 = (1/12) * m2 * l2^2;

g  = 9.81;

T = 2;          % total time [s] 
N = 100;        % number of segments
dt = T / N;

%% ----------------- Decision variables ----------------
opti = Opti();

nx = 4; nu = 2;

X = opti.variable(nx, N+1);   % states at knots
TH1 = X(1,:); DTH1 = X(2,:);
TH2 = X(3,:); DTH2 = X(4,:);

% Controls at knots 
U = opti.variable(nu, N+1);   
%% ----------------- RHS----------------
% f(x,u) returns xdot = [dth1; ddth1; dth2; ddth2]
f = @(xvec, Tau) arm_rhs(xvec, Tau, m1,m2,l1,l2,I1,I2,g);

%% ----------------- Hermite–Simpson collocation ----------------
for k = 1:N
    xk  = X(:,k);
    xk1 = X(:,k+1);
    uk  = U(:,k);
    uk1 = U(:,k+1);

    fk  = f(xk,  uk);
    fk1 = f(xk1, uk1);

    xm  = (xk + xk1)/2 + (dt/8)*(fk - fk1);  
    um  = (uk + uk1)/2;
    fm  = f(xm, um);

    opti.subject_to( xk1 == xk + dt/6*(fk + 4*fm + fk1) );
end

%% ----------------- Boundary conditions----------------
% Initial
th1_0 = -pi/8;    dth1_0 = 0;
th2_0 = 5*pi/8; dth2_0 = 0;
opti.subject_to( X(:,1) == [th1_0; dth1_0; th2_0; dth2_0] );

% Final
th1_f = pi/2; th2_f = 0;
opti.subject_to( TH1(end) == th1_f );
opti.subject_to( TH2(end) == th2_f );
opti.subject_to( DTH1(end) == 0 );
opti.subject_to( DTH2(end) == 0 );

%% ----------------- Path bounds ----------------
% Joint angle constraints
opti.subject_to(-pi/8 <= TH1 <= pi/2);        % link 1
opti.subject_to(0 <= TH2 <= pi);      % link 2

% Hand cannot go behind head
x2 = l1*cos(TH1) + l2*cos(TH1+TH2);
opti.subject_to(x2 >= 0);

% Velocity limits 
vlim = 2;
opti.subject_to( -vlim <= DTH1 <= vlim );
opti.subject_to( -vlim <= DTH2 <= vlim );

% Torque limits
tau_max = [40; 40];
for i = 1:nu
    opti.subject_to( -tau_max(i) <= U(i,:) <= tau_max(i) );
end

%% ----------------- Objective ----------------
J = 0;
W = diag([1, 1]); 
for k = 1:N
    uk  = U(:,k);
    uk1 = U(:,k+1);
    um  = (uk + uk1)/2;

    J = J + dt/6*( uk.'*W*uk + 4*(um.'*W*um) + uk1.'*W*uk1 );
    J = J + 1e-1*(uk1-uk).'*(uk1-uk);
    J = J + 1e-1*(DTH1(k)^2 + DTH2(k)^2);
end

opti.minimize(J);

%% ----------------- Initial guess ----------------
tgrid = linspace(0, T, N+1);

Nmid = floor((N+1)/2);
phase1 = linspace(th1_0, pi/2, Nmid);
phase2 = linspace(pi/2, pi/2, N+1 - Nmid);
TH1_init = [phase1 phase2];

% Apply initial guess
% opti.set_initial(TH1, linspace(th1_0, th1_f, N+1));
opti.set_initial(TH1, TH1_init);
opti.set_initial(TH2, linspace(th2_0, th2_f, N+1));
% opti.set_initial(TH2, TH2_init);
opti.set_initial(DTH1, 0);
opti.set_initial(DTH2, 0);
opti.set_initial(U, 0);

%% ----------------- Solver options & solve ----------------
opti.solver('ipopt', struct('print_time',false,'ipopt',struct('tol',1e-6,'max_iter',2000)));
sol = opti.solve();

%% --------------------------- Extract Solution ---------------------------
TH1_opt = sol.value(TH1);
TH2_opt = sol.value(TH2);
U_opt   = sol.value(U);

%% --------------------------- Plot Joint Angles ---------------------------
figure('Name','Joint Angles','Color',[1 1 1]);
subplot(2,1,1); plot(tgrid, TH1_opt, 'LineWidth', 2); ylabel('\theta_1 [rad]'); grid on; title('Shoulder Joint');
subplot(2,1,2); plot(tgrid, TH2_opt, 'LineWidth', 2); ylabel('\theta_2 [rad]'); grid on; title('Elbow Joint');

%% --------------------------- Plot Joint Torques ---------------------------
figure('Name','Joint Torques','Color',[1 1 1]);
stairs(tgrid, U_opt(1,:), 'LineWidth',2); hold on;
stairs(tgrid, U_opt(2,:), 'LineWidth',2);
xlabel('Time [s]'); ylabel('Torque [Nm]'); grid on;
legend('\tau_1','\tau_2'); title('Joint Torques vs Time');

% Compute absolute torque effort
dt = tgrid(2) - tgrid(1);
abs_effort = sum(abs(U_opt),2) * dt;

% Add absolute torque effort to legend
legend(sprintf('\\tau_1 (abs %.2f)', abs_effort(1)), ...
       sprintf('\\tau_2 (abs %.2f)', abs_effort(2)));


%% --------------------------- Compute Collocation Errors ---------------------------
nx = size(X,1);           
nj = nx/2;                
collocation_error_angle = zeros(1, N);  
collocation_error_vel   = zeros(1, N);  

for k = 1:N
    
    xk  = sol.value(X(:,k));
    xk1 = sol.value(X(:,k+1));
    uk  = sol.value(U(:,k));
    uk1 = sol.value(U(:,k+1));

    
    fk  = full(f(xk, uk));
    fk1 = full(f(xk1, uk1));
    xm  = (xk + xk1)/2 + (dt/8)*(fk - fk1);
    um  = (uk + uk1)/2;
    fm  = full(f(xm, um));

    defect = xk1 - xk - dt/6*(fk + 4*fm + fk1);

    % Sum absolute defects across all joints
    angle_defects = abs(defect(1:2:end));      % all angles
    vel_defects   = abs(defect(2:2:end));      % all velocities

    collocation_error_angle(k) = sum(angle_defects);  % rad
    collocation_error_vel(k)   = sum(vel_defects);    % rad/s
end

%% --------------------------- Plot Combined Collocation Errors ---------------------------
figure('Name','Collocation Errors','Color',[1 1 1]);

subplot(2,1,1);
plot(tgrid(2:end), collocation_error_angle, 'LineWidth', 2);
grid on; ylabel('Angle defect [rad]');
title('Collocation Error (All Joints)');

subplot(2,1,2);
plot(tgrid(2:end), collocation_error_vel, 'LineWidth', 2);
grid on; xlabel('Time [s]');
ylabel('Velocity defect [rad/s]');

%% --------------------------- Precompute Link Positions ---------------------------
Nframes = length(tgrid);
shoulder_pos = [0;0];
elbow_pos = zeros(2,Nframes);
hand_pos  = zeros(2,Nframes);

for k = 1:Nframes
    th1 = TH1_opt(k); th2 = TH2_opt(k);
    p_elbow = shoulder_pos + [l1*cos(th1); l1*sin(th1)];
    p_hand  = p_elbow + [l2*cos(th1+th2); l2*sin(th1+th2)];
    elbow_pos(:,k) = p_elbow;
    hand_pos(:,k)  = p_hand;
end

%% --------------------------- 2D Arm Animation --------------------------- 
fig = figure('Name','2D Arm Animation','Color',[1 1 1]);
ax = axes('Parent',fig); hold(ax,'on'); grid(ax,'on'); axis equal;
xlim([-0.2, l1+l2+0.1]); ylim([-0.2, l1+l2+0.1]);
xlabel('X [m]'); ylabel('Z [m]'); title('2D 2DOF Arm Motion');

% init
p_elbow0 = elbow_pos(:,1);
p_hand0  = hand_pos(:,1);

init_col = [0.7 0.7 0.7];   % light grey

% Initial links 
plot(ax, [0, p_elbow0(1)], [0, p_elbow0(2)], '--', 'LineWidth', 2, 'Color', init_col);
plot(ax, [p_elbow0(1), p_hand0(1)], [p_elbow0(2), p_hand0(2)], '--', 'LineWidth', 2, 'Color', init_col);

% Initial joints 
scatter(ax, 0, 0, 100, 'filled', 'MarkerFaceColor', [0.9 0.9 0.9], 'MarkerEdgeColor', 'none');
scatter(ax, p_elbow0(1), p_elbow0(2), 100, 'filled', 'MarkerFaceColor', [0.9 0.9 0.9], 'MarkerEdgeColor', 'none');
scatter(ax, p_hand0(1), p_hand0(2), 80, 'filled', 'MarkerFaceColor', [0.9 0.9 0.9], 'MarkerEdgeColor', 'none');

% Active arm setup
h_link1 = plot([0 elbow_pos(1,1)], [0 elbow_pos(2,1)], 'LineWidth', 4, 'Color', [0.2 0.4 0.3]);
h_link2 = plot([elbow_pos(1,1) hand_pos(1,1)], [elbow_pos(2,1) hand_pos(2,1)], 'LineWidth', 3, 'Color', [0.3 0.6 0.4]);

h_shoulder = scatter(0, 0, 120, 'filled', 'MarkerFaceColor', [0.2 0.4 0.3], 'MarkerEdgeColor', 'none');
h_elbow    = scatter(elbow_pos(1,1), elbow_pos(2,1), 120, 'filled', 'MarkerFaceColor', [0.3 0.6 0.4], 'MarkerEdgeColor', 'none');
h_hand     = scatter(hand_pos(1,1), hand_pos(2,1), 100, 'filled', 'MarkerFaceColor', [0.4 0.8 0.5], 'MarkerEdgeColor', 'none');

h_trace = plot(hand_pos(1,1), hand_pos(2,1), '--', 'LineWidth', 1, 'Color', [0.4 0.8 0.5]);

%  Animation loop 
for k = 1:Nframes
    % Update link positions
    set(h_link1, 'XData', [0 elbow_pos(1,k)], 'YData', [0 elbow_pos(2,k)]);
    set(h_link2, 'XData', [elbow_pos(1,k) hand_pos(1,k)], 'YData', [elbow_pos(2,k) hand_pos(2,k)]);
    
    % Update joints
    set(h_elbow, 'XData', elbow_pos(1,k), 'YData', elbow_pos(2,k));
    set(h_hand,  'XData', hand_pos(1,k),  'YData', hand_pos(2,k));
    
    % Update trace
    set(h_trace, 'XData', hand_pos(1,1:k), 'YData', hand_pos(2,1:k));
    
    drawnow;
    pause(0.02);
end

%% ----------------- RHS helper ---------------
function xdot = arm_rhs(xvec, Tau, m1,m2,L1,L2,I1,I2,g)
    
    th1 = xvec(1); dth1 = xvec(2);
    th2 = xvec(3); dth2 = xvec(4);   
              
    % [th1; dth1; th2; dth2]
    x0 = [th1; dth1; th2; dth2];     
    tau0 = Tau;
    params = [m1; m2; L1; L2; I1; I2; g];  
    
    f_dyn = Basic_Model_Arm_CasADi();
    dx = full(f_dyn(x0, tau0, params));  

    % dx = [dth1; ddth1; dth2; ddth2]
    ddth1 = dx(2);
    ddth2 = dx(4);
   
    xdot = [ dth1;
             ddth1;
             dth2;
             ddth2 ];
end
