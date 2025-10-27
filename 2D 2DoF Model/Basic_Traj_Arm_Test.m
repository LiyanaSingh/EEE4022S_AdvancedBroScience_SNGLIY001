% 2-DOF Planar Arm with L1 variation 

Basic_Model_Arm_CasADi;
import casadi.*

%% ----------------- Model parameters -----------------
weight = 67.8; 
l1_mean = 0.3012;
l1_std  = 0.0166;
l2 = 0.2342;

L1_values = l1_mean + [-2 -1 0 1 2]*l1_std;

% Pastel colours
colors = [1.0 0.5 0.7;   % pastel pink
          1.0 0.7 0.3;   % soft orange
          0.4 0.8 0.5;   % mint green
          0.4 0.7 1.0;   % sky blue
          0.7 0.5 0.9];  % lavender purple

T = 2;          
N = 100;        
dt = T / N;
tgrid = linspace(0, T, N+1);

hand_paths = cell(1,length(L1_values));
torque_paths = cell(1,length(L1_values));

%% ----------------- Loop over L1 variations -----------------
for ii = 1:length(L1_values)

    l1 = L1_values(ii);
    fprintf('Solving for L1 = %.4f (L2/L1 = %.3f)\n', l1, l2/l1);

    % Masses and inertias
    m1 = weight*30.695/1000;   
    m2 = weight*17.035/1000;   
    I1 = (1/12) * m1 * l1^2;
    I2 = (1/12) * m2 * l2^2;
    g  = 9.81;

    %% ----------------- Decision variables ----------------
    opti = Opti();
    nx = 4; nu = 2;

    X = opti.variable(nx, N+1);   
    TH1 = X(1,:); DTH1 = X(2,:);
    TH2 = X(3,:); DTH2 = X(4,:);
    U = opti.variable(nu, N+1);   

    %% ----------------- RHS ----------------
    f = @(xvec, Tau) arm_rhs(xvec, Tau, m1,m2,l1,l2,I1,I2,g);

    %% ----------------- Hermite–Simpson collocation ----------------
    for k = 1:N
        xk  = X(:,k);  xk1 = X(:,k+1);
        uk  = U(:,k);  uk1 = U(:,k+1);

        fk  = f(xk,uk);  fk1 = f(xk1,uk1);
        xm  = (xk+xk1)/2 + (dt/8)*(fk - fk1);
        um  = (uk + uk1)/2;
        fm  = f(xm,um);

        opti.subject_to(xk1 == xk + dt/6*(fk + 4*fm + fk1));
    end

    %% ----------------- Boundary conditions ----------------
    th1_0 = -pi/8; dth1_0 = 0;
    th2_0 = 5*pi/8; dth2_0 = 0;
    th1_f = pi/2; th2_f = 0;

    opti.subject_to(X(:,1) == [th1_0; dth1_0; th2_0; dth2_0]);
    opti.subject_to(TH1(end) == th1_f);
    opti.subject_to(TH2(end) == th2_f);
    opti.subject_to(DTH1(end) == 0);
    opti.subject_to(DTH2(end) == 0);

    %% ----------------- Path bounds ----------------
    opti.subject_to(-pi/8 <= TH1 <= pi/2);
    opti.subject_to(0 <= TH2 <= pi);
    x2_hand = l1*cos(TH1) + l2*cos(TH1+TH2);
    opti.subject_to(x2_hand >= 0);

    vlim = 2;
    opti.subject_to(-vlim <= DTH1 <= vlim);
    opti.subject_to(-vlim <= DTH2 <= vlim);

    tau_max = [20; 20];
    for i = 1:nu
        opti.subject_to(-tau_max(i) <= U(i,:) <= tau_max(i));
    end

    %% ----------------- Objective ----------------
    J = 0;
    W = diag([1,1]);
    for k = 1:N
        uk  = U(:,k); uk1 = U(:,k+1); um = (uk+uk1)/2;
        J = J + dt/6*(uk.'*W*uk + 4*(um.'*W*um) + uk1.'*W*uk1);
        J = J + 1e-1*(uk1-uk).'*(uk1-uk);
        J = J + 1e-1*(DTH1(k)^2 + DTH2(k)^2);
    end
    opti.minimize(J);

    %% ----------------- Initial guess ----------------
    Nmid = floor((N+1)/2);
    phase1 = linspace(th1_0, pi/2, Nmid);
    phase2 = linspace(pi/2, pi/2, N+1 - Nmid);
    TH1_init = [phase1 phase2];
    opti.set_initial(TH1, TH1_init);
    opti.set_initial(TH2, linspace(th2_0, th2_f, N+1));
    opti.set_initial(DTH1, 0);
    opti.set_initial(DTH2, 0);
    opti.set_initial(U, 0);

    %% ----------------- Solve ----------------
    try
        opti.solver('ipopt', struct('print_time',false,'ipopt',struct('tol',1e-6,'max_iter',2000)));
        sol = opti.solve();
    catch ME
        warning('Solve failed for L1 = %.4f: %s', l1, ME.message);
        hand_paths{ii} = [];
        torque_paths{ii} = [];
        continue;
    end

    TH1_opt = sol.value(TH1);
    TH2_opt = sol.value(TH2);
    U_opt   = sol.value(U);

    %% ----------------- Forward kinematics ----------------
    x1 = l1*cos(TH1_opt);
    y1 = l1*sin(TH1_opt);
    x2 = x1 + l2*cos(TH1_opt + TH2_opt);
    y2 = y1 + l2*sin(TH1_opt + TH2_opt);

    hand_paths{ii} = [x2; y2];
    torque_paths{ii} = U_opt;

end

%% ----------------- Plot bar paths ----------------
figure('Name','2D Arm Bar Paths','Color',[1 1 1]); 
hold on; grid on; axis equal;
xlabel('X [m]');
ylabel('Z [m]');
title('Bar Paths for Different Brachial Indices');

for ii = 1:length(L1_values)
    if isempty(hand_paths{ii}), continue; end
    plot(hand_paths{ii}(1,:), hand_paths{ii}(2,:), 'LineWidth',2, 'Color', colors(ii,:));
end

legend_labels = arrayfun(@(L1) sprintf('L1 = %.4f (L2/L1 = %.3f)', L1, l2/L1), ...
       L1_values, 'UniformOutput', false);
legend(legend_labels, 'Location','bestoutside');

%% ----------------- Plot joint torques τ1 ----------------
figure('Name','Joint Torques τ_1','Color',[1 1 1]); 
hold on; grid on;
xlabel('Time [s]'); ylabel('Torque [Nm]'); 
title('Shoulder Torque for Different Brachial Indices');

for ii = 1:length(L1_values)
    if isempty(torque_paths{ii}), continue; end
    U_opt = torque_paths{ii};
    plot(tgrid, U_opt(1,:), 'LineWidth',2, 'Color', colors(ii,:));
end

legend_labels = arrayfun(@(ii) ...
    sprintf('L1 = %.4f (abs %.2f)', L1_values(ii), sum(abs(torque_paths{ii}(1,:))*dt)), ...
    1:length(L1_values), 'UniformOutput', false);
legend(legend_labels, 'Location','bestoutside');

%% ----------------- Plot joint torques τ2 ----------------
figure('Name','Joint Torques τ_2','Color',[1 1 1]); 
hold on; grid on;
xlabel('Time [s]'); ylabel('Torque [Nm]'); 
title('Elbow Torque for Different Brachial Indices');

for ii = 1:length(L1_values)
    if isempty(torque_paths{ii}), continue; end
    U_opt = torque_paths{ii};
    plot(tgrid, U_opt(2,:), 'LineWidth',2, 'Color', colors(ii,:));
end

legend_labels = arrayfun(@(ii) ...
    sprintf('L1 = %.4f (abs %.2f)', L1_values(ii), sum(abs(torque_paths{ii}(2,:))*dt)), ...
    1:length(L1_values), 'UniformOutput', false);
legend(legend_labels, 'Location','bestoutside');

%% ----------------- RHS helper ----------------
function xdot = arm_rhs(xvec, Tau, m1,m2,L1,L2,I1,I2,g)
    th1 = xvec(1); dth1 = xvec(2);
    th2 = xvec(3); dth2 = xvec(4);   
    x0 = [th1; dth1; th2; dth2];     
    tau0 = Tau;
    params = [m1; m2; L1; L2; I1; I2; g];  
    f_dyn = Basic_Model_Arm_CasADi();
    dx = full(f_dyn(x0, tau0, params));
    xdot = [dth1; dx(2); dth2; dx(4)];
end
