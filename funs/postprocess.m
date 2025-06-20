function [resmax,res_mean,res_std,result,PreY]= postprocess(U_temp,Y,numclass)

U = U_temp(:,1:numclass); %no

stream = RandStream.getGlobalStream;
reset(stream);
U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1,size(U,2));
maxIter = 50;

for iter = 1:maxIter
    [indx{iter},center{iter},~,sumD{iter}] = litekmeans(U_normalized,numclass,'MaxIter',100, 'Replicates',3);
    [result(iter,:),map_Y{iter}] = Clustering8Measure(Y,indx{iter});
end

[~,max_index] = max(result(:,1),[],1);
resmax = result(max_index,:);
res_mean = mean(result,1);
res_std = std(result,1);
PreY = map_Y{max_index};
