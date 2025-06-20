clear;
clc;
warning off;
addpath(genpath('./'));


Dataset_Path = './';
resPath = './results/';

dataName = 'MSRCV1';

disp(dataName);
load(strcat(Dataset_Path,dataName));

numsample = size(Y,1);
numview = length(X); 
numclass = length(unique(Y));

for p = 1:numview
    X{p} = mapstd(X{p}',0,1); 
    X_dim(p) = size(X{p},1);
    X_dim_min = min(X_dim);
end

for p = 1:numview               
    index = sum(abs(X{p}),2) > 1e-8;
    X{p} = X{p}(index,:);
    X_dim(p) = sum(index);
end
X_dim_min = min(X_dim);

Anchor_set = [2]*numclass; 
feature_set = [3]*numclass;
eta_set = 10.^[0]; 
lambda_set = 10.^[0];
beta_set = 10.^[1]; 
%%
for Anchor_index = 1:length(Anchor_set)
    
    anchor = Anchor_set(Anchor_index);
    if anchor > numsample
        continue
    end
    
    for feature_index = 1:length(feature_set)
        proj_d = feature_set(feature_index);
        if proj_d > numsample | proj_d > X_dim_min
            continue
        end
        for eta_index = 1:length(eta_set)
            eta = eta_set(eta_index); 
            for lambda_index = 1:length(lambda_set) 
                lambda = lambda_set(lambda_index); 
                
                for beta_index = 1:length(beta_set) 
                    beta = beta_set(beta_index);
                    
                    [U,Up,V,A,Z,alpha,gamma,objall,obj] = main(X,Y,proj_d,eta,lambda,beta,anchor);
                    
                    [res_max,res_mean,res_std,~,PreY]= postprocess(U,Y,numclass);
                    
                    fprintf('ACC:%4.2f \t NMI:%4.2f \t Pur:%4.2f \t Fscore:%4.2f \n',...
                        [res_mean(1)*100 res_mean(2)*100 res_mean(3)*100 res_mean(4)*100]);
                end
            end
        end
    end
end