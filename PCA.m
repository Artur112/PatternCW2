function [U, diffimg, mean_feature] = PCA(data)
    %Examples must be along columns
    
    mean_feature = mean(data,2);
    diffimg = data - mean_feature;

    S_LD = 1/(length(data))*transpose(diffimg)*diffimg;
    [V_LD,D_LD] = eig(S_LD);
    U = diffimg*V_LD; %U is the eigenface matrix
    for n=1:length(U(1,:))
        U(:,n) = U(:,n)/norm(U(:,n)); %Need to normalize it in the low dimensional PCA technique
    end  

    %Need to sort the eigenvalues in descending order and then the eigenvectors
    %accordingly
    D_LD = diag(D_LD)';
    eig_values_LD = sort(D_LD,'descend');
    eig_values_LD = eig_values_LD(1:length(data)-1);
    [~,Bsort]=sort(D_LD, 'descend'); %Get the order of B
    U=U(:,Bsort);
    U = U(:,1:length(data)-1);
end

