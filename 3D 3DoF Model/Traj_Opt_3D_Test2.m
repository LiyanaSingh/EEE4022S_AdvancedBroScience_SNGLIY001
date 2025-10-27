% Test B4.1


import casadi.*
Basic_Model_Arm3D_CasADi;

%% --------------------------- Model Parameters ---------------------------
weight = 67.8; %\cite{puoane2002}
m1_ratio = 30.695/1000; m2_ratio = 17.035/1000;   % \cite{drillis1964}
L2 = 0.2342;  %\cite{Reimann2019ReferenceTW}
L1_mean = 0.3012;
L1_std  = 0.0166;

T = 2; N = 100; dt = T/N;
tgrid = linspace(0,T,N+1);

% L1 variations: -2σ, -1σ, mean, +1σ, +2σ
L1_values = L1_mean + [-2 -1 0 1 2]*L1_std;

% Custom colors (5)
colors = [ 1.0 0.5 0.7;   % pastel pink
           1.0 0.7 0.3;   % soft orange
           0.4 0.8 0.5;   % mint green
           0.4 0.7 1.0;   % sky blue
           0.7 0.5 0.9];  % lavender purple

% Storage
hand_trajectories = cell(1,numel(L1_values));
elbow_trajectories = cell(1,numel(L1_values));
torques = cell(1,numel(L1_values));   
legendEntries = cell(1, numel(L1_values));
yaw_init_deg = nan(1, numel(L1_values));   
yaw_final_deg = nan(1, numel(L1_values));  
grip_width = nan(1, numel(L1_values));



%% ---------------------- Loop over L1 values ----------------------
for ii = 1:numel(L1_values)
    L1 = L1_values(ii);
    fprintf('--- Running solve %d / %d: L1 = %.4f m (ratio L2/L1 = %.4f) ---\n', ...
            ii, numel(L1_values), L1, L2/L1);

    % Masses and inertias
    m1 = weight * m1_ratio;
    m2 = weight * m2_ratio;
    I1 = diag([ (1/12)*m1*L1^2, 0.01, 0.01 ]);
    I2 = diag([ (1/12)*m2*L2^2, 0.008, 0.008 ]);
    g = 9.81;

    % Opti problem
    opti = Opti();
    nx = 6; nu = 3;

    X = opti.variable(nx, N+1);
    TH1 = X(1,:); DTH1 = X(2,:);
    TH2 = X(3,:); DTH2 = X(4,:);
    TH3 = X(5,:); DTH3 = X(6,:);
    U = opti.variable(nu, N+1);

    f = @(xvec, Tau) arm3d_rhs_3dof(xvec, Tau, m1, m2, L1, L2, I1, I2, g);

    % Hermite-Simpson collocation
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

    % Boundary conditions
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

    % Path bounds
    opti.subject_to(0 <= TH1 <= pi/2);
    opti.subject_to(-pi/8 <= TH2 <= pi/2);
    opti.subject_to(0 <= TH3 <= pi);
    vlim = 2;
    opti.subject_to(-vlim <= DTH1 <= vlim);
    opti.subject_to(-vlim <= DTH2 <= vlim);
    opti.subject_to(-vlim <= DTH3 <= vlim);
    tau_max = [40;40;40];
    for jj = 1:nu
        opti.subject_to(-tau_max(jj) <= U(jj,:) <= tau_max(jj));
    end

    % initial hand y_ref
    R01_0 = Roty_sym(th2_0) * Rotz_sym(th1_0);
    R02_0 = R01_0 * Roty_sym(th3_0);
    p_hand0 = R01_0 * [L1;0;0] + R02_0 * [L2;0;0];
    y_ref = p_hand0(2);   

    % Hand constraints (constant Y, x >= 0)
    for k = 1:N+1
        th1 = X(1,k); th2 = X(3,k); th3 = X(5,k);
        R01 = Roty_sym(th2) * Rotz_sym(th1);
        R02 = R01 * Roty_sym(th3);
        p_hand = R01*[L1;0;0] + R02*[L2;0;0];

        % Fix Y-position to initial
        opti.subject_to(abs(p_hand(2) - y_ref)<= 1e-2);

        % x >= 0
        opti.subject_to(p_hand(1) >= 0);
    end

    % Objective
    J = 0;
    W = diag([1.5,1,1]);
    for k = 1:N
        uk  = U(:,k); uk1 = U(:,k+1);
        um  = (uk + uk1)/2;
        J = J + dt/6*( uk.'*W*uk + 4*(um.'*W*um) + uk1.'*W*uk1 );
        J = J + 1e-1*(uk1-uk).'*(uk1-uk);
        J = J + 1e-1*(DTH1(k)^2 + DTH2(k)^2 + DTH3(k)^2);
    end

    % Bar straight-line trajectory penalty
    % Compute start and end hand positions in X-Z plane
    R01_start = Roty_sym(th2_0) * Rotz_sym(th1_0);        
    R02_start = R01_start * Roty_sym(th3_0);
    p_hand_start = R01_start*[L1;0;0] + R02_start*[L2;0;0];
    
    R01_end = Roty_sym(th2_f) * Rotz_sym(th1_0);          
    R02_end = R01_end * Roty_sym(th3_f);
    p_hand_end = R01_end*[L1;0;0] + R02_end*[L2;0;0];
    
    % Linear reference
    X_ref = linspace(p_hand_start(1), p_hand_end(1), N+1);
    Z_ref = linspace(p_hand_start(3), p_hand_end(3), N+1);
    
    % Add deviation penalty
    for k = 1:N+1
        th1 = X(1,k); th2 = X(3,k); th3 = X(5,k);
        R01 = Roty_sym(th2) * Rotz_sym(th1);
        R02 = R01 * Roty_sym(th3);
        p_hand = R01*[L1;0;0] + R02*[L2;0;0];
    
        J = J + ((p_hand(1) - X_ref(k))^2 + (p_hand(3) - Z_ref(k))^2);
    end

    opti.minimize(J);

    % Initial guess
    opti.set_initial(TH1, linspace(0, 0, N+1));
    opti.set_initial(TH2, linspace(th2_0, th2_f, N+1));
    opti.set_initial(TH3, linspace(th3_0, th3_f, N+1));
    opti.set_initial(DTH1, 0);
    opti.set_initial(DTH2, 0);
    opti.set_initial(DTH3, 0);
    opti.set_initial(U, 0);

    % Solve
    try
        opti.solver('ipopt', struct('print_time',false,'ipopt',struct('tol',1e-6,'max_iter',5000)));
        sol = opti.solve();
    
        U_opt = sol.value(U);             
        torques{ii} = U_opt;  

        
        yaw_init = sol.value(th1_0);         
        TH1_sol = sol.value(TH1);            
        yaw_final = TH1_sol(end);           
    
        yaw_init_deg(ii) = rad2deg(yaw_init);
        yaw_final_deg(ii) = rad2deg(yaw_final);
    
        legendEntries{ii} = sprintf('L1=%.4f m (L2/L1=%.3f)', ...
                                    L1, L2/L1);
        biac_diam = 0.35306;
        L3 = L1*cos(th2_0);
        y_0 = L3*sin(yaw_init);
        grip_width(ii) = biac_diam*(1/2) + y_0;
    
    catch ME
        warning('Solve failed for L1 = %.4f: %s', L1, ME.message);
        hand_trajectories{ii} = [];
        elbow_trajectories{ii} = [];
        torques{ii} = [];   
        legendEntries{ii} = sprintf('L1=%.4f m (failed)', L1);  
        continue;
    end

    TH1_opt = sol.value(TH1);
    TH2_opt = sol.value(TH2);
    TH3_opt = sol.value(TH3);

    % Compute link positions 
    Nframes = length(tgrid);
    elbow_pos = zeros(3,Nframes);
    hand_pos  = zeros(3,Nframes);
    for k = 1:Nframes
        th1 = TH1_opt(k); th2 = TH2_opt(k); th3 = TH3_opt(k);
        R01 = Roty_num(th2) * Rotz_num(th1);
        R12 = Roty_num(th3); R02 = R01 * R12;
        p_elbow = R01 * [L1;0;0];
        p_hand  = p_elbow + R02 * [L2;0;0];
        elbow_pos(:,k) = p_elbow;
        hand_pos(:,k)  = p_hand;
    end

    hand_trajectories{ii}  = hand_pos;
    elbow_trajectories{ii} = elbow_pos;
end
%% ---------------------- Hand path projections ----------------------
%legendEntries = arrayfun(@(val) sprintf('L1=%.4f m (L2/L1=%.3f)', val, L2/val), L1_values, 'UniformOutput', false);

% Side (X-Z)
figure('Name','Hand Trajectory - Side (X-Z)','Color',[1 1 1]);
hold on; grid on; axis equal;
for ii = 1:numel(L1_values)
    if isempty(hand_trajectories{ii}), continue; end
    plot(hand_trajectories{ii}(1,:), hand_trajectories{ii}(3,:), ...
         'LineWidth',2, 'Color', colors(ii,:));
end
xlabel('X'); ylabel('Z'); title('Hand Trajectory - Side (X-Z Plane)');
legend(legendEntries,'Location','bestoutside');

% Top (X-Y)
figure('Name','Hand Trajectory - Top (X-Y)','Color',[1 1 1]);
hold on; grid on; axis equal;
for ii = 1:numel(L1_values)
    if isempty(hand_trajectories{ii}), continue; end
    plot(hand_trajectories{ii}(1,:), hand_trajectories{ii}(2,:), ...
         'LineWidth',2, 'Color', colors(ii,:));
end
xlabel('X'); ylabel('Y'); title('Hand Trajectory - Top (X-Y Plane)');
legend(legendEntries,'Location','bestoutside');

% Front (Y-Z)
figure('Name','Hand Trajectory - Front (Y-Z)','Color',[1 1 1]);
hold on; grid on; axis equal;
for ii = 1:numel(L1_values)
    if isempty(hand_trajectories{ii}), continue; end
    plot(hand_trajectories{ii}(2,:), hand_trajectories{ii}(3,:), ...
         'LineWidth',2, 'Color', colors(ii,:));
end
xlabel('Y'); ylabel('Z'); title('Hand Trajectory - Front (Y-Z Plane)');
legend(legendEntries,'Location','bestoutside');


%% ---------------------- Compute absolute torque----------------------
dt = tgrid(2)-tgrid(1);

legendEntries_withEffort_tau1 = cell(size(L1_values));
legendEntries_withEffort_tau2 = cell(size(L1_values));
legendEntries_withEffort_tau3 = cell(size(L1_values));

for ii = 1:numel(L1_values)
    if isempty(torques{ii})
        legendEntries_withEffort_tau1{ii} = legendEntries{ii};
        legendEntries_withEffort_tau2{ii} = legendEntries{ii};
        legendEntries_withEffort_tau3{ii} = legendEntries{ii};
        continue;
    end
    
    % absolute effort for each torque 
    absEffort_tau1 = sum(abs(torques{ii}(1,:)))*dt;
    absEffort_tau2 = sum(abs(torques{ii}(2,:)))*dt;
    absEffort_tau3 = sum(abs(torques{ii}(3,:)))*dt;
    
    % append to legend entries
    legendEntries_withEffort_tau1{ii} = sprintf('%s (abs %.2f)', legendEntries{ii}, absEffort_tau1);
    legendEntries_withEffort_tau2{ii} = sprintf('%s (abs %.2f)', legendEntries{ii}, absEffort_tau2);
    legendEntries_withEffort_tau3{ii} = sprintf('%s (abs %.2f)', legendEntries{ii}, absEffort_tau3);
end


%% ---------------------- Torque plots ----------------------
% Yaw torque (tau1)
figure('Name','Yaw Torque','Color',[1 1 1]); hold on; grid on;
for ii = 1:numel(L1_values)
    if isempty(torques{ii}), continue; end
    plot(tgrid, torques{ii}(1,:), 'LineWidth',2, 'Color', colors(ii,:));
end
xlabel('Time [s]'); ylabel('\tau_1 [Nm]');
title('Shoulder Yaw Torque (\tau_1)');
legend(legendEntries_withEffort_tau1,'Location','bestoutside');
axis tight;

% Pitch torque (tau2)
figure('Name','Pitch Torque','Color',[1 1 1]); hold on; grid on;
for ii = 1:numel(L1_values)
    if isempty(torques{ii}), continue; end
    plot(tgrid, torques{ii}(2,:), 'LineWidth',2, 'Color', colors(ii,:));
end
xlabel('Time [s]'); ylabel('\tau_2 [Nm]');
title('Shoulder Pitch Torque (\tau_2)');
legend(legendEntries_withEffort_tau2,'Location','bestoutside');
axis tight;

% Elbow torque (tau3)
figure('Name','Elbow Torque','Color',[1 1 1]); hold on; grid on;
for ii = 1:numel(L1_values)
    if isempty(torques{ii}), continue; end
    plot(tgrid, torques{ii}(3,:), 'LineWidth',2, 'Color', colors(ii,:));
end
xlabel('Time [s]'); ylabel('\tau_3 [Nm]');
title('Elbow Torque (\tau_3)');
legend(legendEntries_withEffort_tau3,'Location','bestoutside');
axis tight;

%% ---------------------- Print ratios ----------------------
fprintf('\nL2/L1 ratios and yaw:\n');
for ii = 1:numel(L1_values)
    fprintf('L1 = %.4f -> L2/L1 = %.5f\n Yaw_0 = %.4f -> Yaw_f = %.4f\n Grip width = %.4f\n\n', L1_values(ii), L2/L1_values(ii),yaw_init_deg(ii),yaw_final_deg(ii), grip_width(ii)/biac_diam/2 *100);
end
%% ---------------------- Helper functions ----------------------
function R = Roty_num(th)
    R = [cos(th), 0, -sin(th);
         0,       1,  0;
         sin(th), 0,  cos(th)];
end
function R = Rotz_num(th)
    R = [cos(th),  sin(th), 0;
        -sin(th),  cos(th), 0;
         0,        0,       1];
end

function R = Roty_sym(th)
    import casadi.*
    R = [cos(th), 0, -sin(th);
         0,       1,  0;
         sin(th), 0,  cos(th)];
end
function R = Rotz_sym(th)
    import casadi.*
    R = [cos(th),  sin(th), 0;
        -sin(th),  cos(th), 0;
         0,        0,       1];
end

function xdot = arm3d_rhs_3dof(xvec, Tau, m1,m2,L1,L2,I1,I2,g)
    params = [ m1; m2; L1; L2; reshape(I1,9,1); reshape(I2,9,1); g ];
    f_dyn = Basic_Model_Arm3D_CasADi();
    xdot_cas = f_dyn(xvec, Tau, params);
   
    try
        xdot = xdot_cas;  
    catch
        xdot = full(xdot_cas); 
    end
end
