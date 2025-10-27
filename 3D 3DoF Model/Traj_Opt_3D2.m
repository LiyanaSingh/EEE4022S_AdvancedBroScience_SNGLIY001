% Test B3.1-B3.3


import casadi.*
Basic_Model_Arm3D_CasADi;

%% --------------------------- Model Parameters ---------------------------
weight = 67.8; %\cite{puoane2002}
m1 = weight*30.695/1000;   
m2 = weight*17.035/1000;   % \cite{drillis1964}        
L1 = 0.3012;   L2 = 0.2342;              %\cite{Reimann2019ReferenceTW}
I1 = diag([ (1/12)*m1*L1^2, 0.01, 0.01 ]);   
I2 = diag([ (1/12)*m2*L2^2, 0.008, 0.008 ]);
g  = 9.81;

T = 2;      % total motion time [s]
N = 100;     % number of intervals
dt = T / N;

%% --------------------------- Decision Variables ---------------------------
opti = Opti();
nx = 6; nu = 3;

% State variables
X = opti.variable(nx, N+1);   
TH1 = X(1,:); DTH1 = X(2,:);
TH2 = X(3,:); DTH2 = X(4,:);
TH3 = X(5,:); DTH3 = X(6,:);

% Control variables
U = opti.variable(nu, N+1);   

%% --------------------------- RHS ---------------------------
f = @(xvec, Tau) arm3d_rhs_3dof(xvec, Tau, m1, m2, L1, L2, I1, I2, g);

%% --------------------------- Hermite–Simpson Collocation ---------------------------
for k = 1:N
    xk  = X(:,k); xk1 = X(:,k+1);
    uk  = U(:,k); uk1 = U(:,k+1);

    fk  = f(xk,  uk);
    fk1 = f(xk1, uk1);

    xm  = (xk + xk1)/2 + (dt/8)*(fk - fk1);
    um  = (uk + uk1)/2;
    fm  = f(xm, um);

    opti.subject_to( xk1 == xk + dt/6*(fk + 4*fm + fk1) );
end

%% --------------------------- Boundary Conditions ---------------------------
th1_0 = opti.variable(); dth1_0 = 0;
th2_0 = -pi/8; dth2_0 = 0;
th3_0 = 5*pi/8; dth3_0 = 0;

th2_f = pi/2; th3_f = 0;

opti.subject_to( X(:,1) == [th1_0; dth1_0; th2_0; dth2_0; th3_0; dth3_0] );
opti.subject_to( TH2(end) == th2_f );
opti.subject_to( TH3(end) == th3_f );
opti.subject_to( DTH1(end) == 0 );
opti.subject_to( DTH2(end) == 0 );
opti.subject_to( DTH3(end) == 0 );

%% --------------------------- Path and Control Bounds ---------------------------
opti.subject_to(0 <= TH1 <= pi/2); 
opti.subject_to(-pi/8 <= TH2 <= pi/2);         
opti.subject_to(0 <= TH3 <= pi);        

vlim = 2;
opti.subject_to(-vlim <= DTH1 <= vlim);
opti.subject_to(-vlim <= DTH2 <= vlim);
opti.subject_to(-vlim <= DTH3 <= vlim);

tau_max = [20; 20; 20];
for i = 1:nu
    opti.subject_to(-tau_max(i) <= U(i,:) <= tau_max(i));
end

% Initial hand position
R01_0 = Roty(th2_0) * Rotz(th1_0);
R02_0 = R01_0 * Roty(th3_0);
p_hand0 = R01_0 * [L1;0;0] + R02_0 * [L2;0;0];
y_ref = p_hand0(2);

for k = 1:N+1
    th1 = X(1,k); th2 = X(3,k); th3 = X(5,k);
    R01 = Roty(th2) * Rotz(th1);
    R02 = R01 * Roty(th3);
    p_hand = R01*[L1;0;0] + R02*[L2;0;0];

    % Fix Y-position to initial
    opti.subject_to(abs(p_hand(2) - y_ref)<= 1e-2);

    % x>= 0
    opti.subject_to(p_hand(1) >= 0);
end

%% --------------------------- Objective Function ---------------------------
J = 0;
W = diag([1.5,1,1]);  % control weighting 
for k = 1:N
    uk  = U(:,k); uk1 = U(:,k+1);
    um  = (uk + uk1)/2;

    J = J + dt/6*( uk.'*W*uk + 4*(um.'*W*um) + uk1.'*W*uk1 );
   
    J = J + 1e-1*(uk1-uk).'*(uk1-uk);
    
    J = J + 1e-1*(DTH1(k)^2 + DTH2(k)^2 + DTH3(k)^2);
end

%% ------------------ Bar straight-line trajectory penalty ------------------
% Compute start and end hand positions in X-Z plane
R01_start = Roty(th2_0) * Rotz(th1_0);        
R02_start = R01_start * Roty(th3_0);
p_hand_start = R01_start*[L1;0;0] + R02_start*[L2;0;0];

R01_end = Roty(th2_f) * Rotz(th1_0);          
R02_end = R01_end * Roty(th3_f);
p_hand_end = R01_end*[L1;0;0] + R02_end*[L2;0;0];

% Linear reference 
X_ref = linspace(p_hand_start(1), p_hand_end(1), N+1);
Z_ref = linspace(p_hand_start(3), p_hand_end(3), N+1);

% Add deviation penalty
for k = 1:N+1
    th1 = X(1,k); th2 = X(3,k); th3 = X(5,k);
    R01 = Roty(th2) * Rotz(th1);
    R02 = R01 * Roty(th3);
    p_hand = R01*[L1;0;0] + R02*[L2;0;0];

    J = J + ((p_hand(1) - X_ref(k))^2 + (p_hand(3) - Z_ref(k))^2);
end

%%
opti.minimize(J);

%% --------------------------- Initial Guess ---------------------------
tgrid = linspace(0,T,N+1);
opti.set_initial(TH1, 0); % change
opti.set_initial(TH2, linspace(th2_0, th2_f, N+1));
opti.set_initial(TH3, linspace(th3_0, th3_f, N+1));
opti.set_initial(DTH1, 0);
opti.set_initial(DTH2, 0);
opti.set_initial(DTH3, 0);
opti.set_initial(U, 0);

%% --------------------------- Solver Options ---------------------------
opti.solver('ipopt', struct('print_time',false,'ipopt',struct('tol',1e-6,'max_iter',3000)));
sol = opti.solve();

%% --------------------------- Extract Solution ---------------------------
TH1_opt = sol.value(TH1);
TH2_opt = sol.value(TH2);
TH3_opt = sol.value(TH3);
U_opt   = sol.value(U);

% Optimized th1_0
th1_0_opt = sol.value(th1_0);
th1_0_opt_deg = rad2deg(th1_0_opt);

%% --------------------------- Plot Joint Angles & Torques ---------------------------
figure;
subplot(3,1,1); plot(tgrid,TH1_opt); title('th1 (shoulder yaw)');
subplot(3,1,2); plot(tgrid,TH2_opt); title('th2 (shoulder pitch)');
subplot(3,1,3); plot(tgrid,TH3_opt); title('th3 (elbow)');

dt = tgrid(2)-tgrid(1);
abs_effort = sum(abs(U_opt),2)*dt;  

% Plot torques over time
figure; 
stairs(tgrid,U_opt.'); 
title('Torques'); 
xlabel('Time [s]'); 
ylabel('Torque [Nm]');

% Create legend with absolute torque effort
legend(sprintf('tau1 (abs %.2f)', abs_effort(1)), ...
       sprintf('tau2 (abs %.2f)', abs_effort(2)), ...
       sprintf('tau3 (abs %.2f)', abs_effort(3)));

%% --------------------------- Precompute Link Positions ---------------------------
Nframes = length(tgrid);
elbow_pos = zeros(3,Nframes);
hand_pos  = zeros(3,Nframes);

for k = 1:Nframes
    th1 = TH1_opt(k); th2 = TH2_opt(k); th3 = TH3_opt(k);
    R01 = Roty(th2) * Rotz(th1);
    R12 = Roty(th3); R02 = R01 * R12;
    p_elbow = R01 * [L1;0;0];
    p_hand  = p_elbow + R02 * [L2;0;0];
    elbow_pos(:,k) = p_elbow;
    hand_pos(:,k)  = p_hand;
end

%% --------------------------- 3D Arm Animation ---------------------------
fig = figure('Name','3D Arm Animation','Color',[1 1 1]);
ax = axes('Parent',fig); 
hold(ax,'on'); grid(ax,'on'); axis(ax,'equal');

% Compute plot limits
all_pts = [zeros(3,1), elbow_pos, hand_pos];
minpt = min(all_pts,[],2)-0.1; 
maxpt = max(all_pts,[],2)+0.1;
xlim(ax,[minpt(1), maxpt(1)]); 
ylim(ax,[minpt(2), maxpt(2)]); 
zlim(ax,[minpt(3), maxpt(3)]);
view(ax,[-30,20]);
xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D 3DoF Arm Motion');

% Initial plot handles
h_link1 = plot3(ax,[0, elbow_pos(1,1)], [0, elbow_pos(2,1)], [0, elbow_pos(3,1)], ...
    'LineWidth',4,'Color',[0.2 0.4 0.3]); % dynamic link1 (current)
h_link2 = plot3(ax,[elbow_pos(1,1), hand_pos(1,1)], [elbow_pos(2,1), hand_pos(2,1)], ...
    [elbow_pos(3,1), hand_pos(3,1)], 'LineWidth',3,'Color',[0.3 0.6 0.4]); % dynamic link2

h_shoulder = scatter3(ax,0,0,0,120,'filled','MarkerFaceColor',[0.2 0.4 0.3]); % dynamic shoulder
h_elbow   = scatter3(ax, elbow_pos(1,1), elbow_pos(2,1), elbow_pos(3,1), ...
    120,'filled','MarkerFaceColor',[0.3 0.6 0.4]); % dynamic elbow
h_hand    = scatter3(ax, hand_pos(1,1), hand_pos(2,1), hand_pos(3,1), ...
    120,'filled','MarkerFaceColor',[0.4 0.8 0.5]); % dynamic hand
h_trace   = plot3(ax, hand_pos(1,1), hand_pos(2,1), hand_pos(3,1), '--', ...
    'LineWidth',1,'Color',[0.4 0.8 0.5]); % trace of hand

% init
p_elbow0 = elbow_pos(:,1);
p_hand0  = hand_pos(:,1);

init_col = [0.6 0.6 0.6];   % light grey
h_init_link1 = plot3(ax, [0, p_elbow0(1)], [0, p_elbow0(2)], [0, p_elbow0(3)], ...
    '--','LineWidth',2,'Color',init_col);
h_init_link2 = plot3(ax, [p_elbow0(1), p_hand0(1)], [p_elbow0(2), p_hand0(2)], ...
    [p_elbow0(3), p_hand0(3)], '--','LineWidth',2,'Color',init_col);

h_init_elbow = scatter3(ax, p_elbow0(1), p_elbow0(2), p_elbow0(3), ...
    80, 'MarkerEdgeColor', init_col, 'MarkerFaceColor', [0.95 0.95 0.95]);
h_init_hand = scatter3(ax, p_hand0(1), p_hand0(2), p_hand0(3), ...
    80, 'MarkerEdgeColor', init_col, 'MarkerFaceColor', [0.95 0.95 0.95]);


% Animation loop
for k = 1:Nframes
    p_elbow = elbow_pos(:,k); 
    p_hand  = hand_pos(:,k);

    % update links
    set(h_link1, 'XData',[0,p_elbow(1)], 'YData',[0,p_elbow(2)], 'ZData',[0,p_elbow(3)]);
    set(h_link2, 'XData',[p_elbow(1),p_hand(1)], 'YData',[p_elbow(2),p_hand(2)], 'ZData',[p_elbow(3),p_hand(3)]);
    
    % update joints
    set(h_elbow,'XData',p_elbow(1),'YData',p_elbow(2),'ZData',p_elbow(3));
    set(h_hand,'XData',p_hand(1),'YData',p_hand(2),'ZData',p_hand(3));
    
    % update trace
    set(h_trace,'XData',hand_pos(1,1:k),'YData',hand_pos(2,1:k),'ZData',hand_pos(3,1:k));
    
    drawnow; 
    pause(0.2);
end


%% --------------------------- Hand Trajectory Projections ---------------------------
figure('Name','Hand Trajectory Projections','Color',[1 1 1]);

subplot(1,3,1); 
plot(hand_pos(1,:), hand_pos(3,:), 'LineWidth',2, 'Color',[0.4 0.8 0.5]); 
xlabel('X'); ylabel('Z'); grid on; title('Side (X-Z Plane)'); axis equal;

subplot(1,3,2); 
plot(hand_pos(1,:), hand_pos(2,:), 'LineWidth',2, 'Color',[0.4 0.8 0.5]); 
xlabel('X'); ylabel('Y'); grid on; title('Top (X-Y Plane)'); axis equal;

subplot(1,3,3); 
plot(hand_pos(2,:), hand_pos(3,:), 'LineWidth',2, 'Color',[0.4 0.8 0.5]); 
xlabel('Y'); ylabel('Z'); grid on; title('Front (Y-Z Plane)'); axis equal;

sgtitle('Hand Trajectory Projections');
%% --- Print ratios to command window ---
biac_diam = 0.35306;
L3 = L1*cos(th2_0);
y_0 = L3*sin(th1_0_opt);
grip_width = biac_diam*(1/2) + y_0;

th1_f_val  = TH1_opt(end);             
th1_0_val  = th1_0_opt;                
grip_val   = sol.value(grip_width)/biac_diam/2*100;    

fprintf('Initial angle: %.2f° | Final angle: %.2f° | Grip width percentage: %.4f \n', ...
    rad2deg(th1_0_val), rad2deg(th1_f_val), grip_val);
%% --------------------------- Compute Collocation Errors (angles & velocities) ---------------------------
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
%% --------------------------- Helper Functions ---------------------------
function R = Roty(th)
    import casadi.*
    R = [cos(th),0,-sin(th);0,1,0;sin(th),0,cos(th)];
end

function R = Rotz(th)
    import casadi.*
    R = [cos(th),sin(th),0; -sin(th),cos(th),0; 0,0,1];
end

function xdot = arm3d_rhs_3dof(xvec, Tau, m1,m2,L1,L2,I1,I2,g)
    params = [ m1; m2; L1; L2; reshape(I1,9,1); reshape(I2,9,1); g ];
    f_dyn = Basic_Model_Arm3D_CasADi();
    xdot = full(f_dyn(xvec, Tau, params));
end
