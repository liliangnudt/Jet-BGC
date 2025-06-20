function [U,Up,V,A,Z,alpha,gamma,objall,obj] = main(X,Y,d_proj,eta,lambda,beta,Anchor)


num = size(Y,1); 
numview = length(X); 
numclass = length(unique(Y)); 
m = Anchor;

flag = 1;
iter = 0;
maxIter = 50; 
IterMax = 50; 
obj = [];
objall = [];

XX = [];
for p = 1 : numview
    XX = [XX;X{p}]; 
    X_dim(p) = size(X{p},1); 
end

for p = 1 : numview
    Up{p} = zeros(X_dim(p), d_proj);
end

V = zeros(d_proj,num); 
[XUV,~,~]=svds(XX', d_proj);
V = XUV';
clear XUV

Z = zeros(num, m);
[XZ,~,~]=svds(XX', m); 
stream = RandStream.getGlobalStream;
reset(stream);
[IDXZ,~] = kmeans(XZ, m, 'MaxIter',200,'Replicates',30); 
clear XZ
for i = 1:num
    Z(i,IDXZ(i)) = 1;
end
Z = Z/(m) + (m-1)/m/m;
clear IDXZ

A = zeros(d_proj, m);
[XA,~,~]=svds(XX', d_proj); 
stream = RandStream.getGlobalStream;
reset(stream);
[~,A_temp] = kmeans(XA, m, 'MaxIter',200,'Replicates',30); 
A = A_temp';
clear A_temp XA

alpha = ones(1,numview)/numview;


for p = 1 : numview
    AAp{p} = zeros(size(X{p}));
    Ep{p} = zeros(size(X{p}));
end

s = 3;
D = L2_distance_1(V, A);
[~, idx] = sort(D, 2); 

for ii = 1 : num
    id = idx(ii, 1:s+1);
    di = D(ii, id);
    gamma_temp(ii) = 0.5 * s * di(s+1) - 0.5 * sum(di(1:s));
end
gamma = mean(gamma_temp);
clear D di gamma_temp ii id idx XX
while flag
    iter = iter + 1;
    beta_inv = 1 / beta;

    for p = 1 : numview
        Ep{p} = L21_solver(X{p} - Up{p} * V + beta_inv * AAp{p}, alpha(p) / beta);
    end
    
    for p = 1 : numview
        Up{p} = (X{p} - Ep{p} + beta_inv * AAp{p}) * V';
    end
    
    D_m = diag(sum(Z,1));
    A = V * Z * inv(eta * Z' * Z + lambda * D_m);
    
    options = optimset('Algorithm','interior-point-convex','Display','off');
    HH = 2 * (eta * A' * A + gamma * eye(m));
    FVA = 2 * V' * A;
    AAtemp = lambda * (diag(A' * A))';
    for i = 1 : num
        Z(i,:) = quadprog(HH,(AAtemp - FVA(i,:))',[],[],ones(1,m),  1,zeros(m,1),ones(m,1),[],options);
    end
    
    Q_temp = 0;
    for p = 1 : numview
        Q_temp = Q_temp + (X{p} - Ep{p} + beta_inv * AAp{p})' * Up{p};
    end
    M = 2 * Z * A' + beta * Q_temp;
    [UV,~,VV] = svd(M,'econ');
    V = (UV * VV')';
    
    Ma = zeros(numview,1);
    for p = 1 : numview
        al_temp1 = (X{p} - Up{p} * V)';
        Ma(p) = sum(sqrt(sum(al_temp1.*al_temp1, 2)));
    end
    Mafra = Ma.^(-1);
    Qa = 1/sum(Mafra);
    alpha = Qa*Mafra;
    
    for p = 1 : numview
        AAp{p} = AAp{p} + beta * (X{p} - Up{p} * V - Ep{p});
    end
    
    [obj(end+1)] = callobj(numview, X, Up, V, A, Z, alpha, eta, lambda, gamma);
    
    beta = beta * 2;

    if (iter>9) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-3 || iter>30 || obj(iter) < 1e-10)
        [U,~,~]=svd(Z,'econ');
        flag = 0;
        clear AAp AAtemp al_temp1 D_m FVA HH M Q_temp UV VV
    end
end