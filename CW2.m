%% Loading in the index and label data and the feature vectors

    load('CW2_data\PR_data\cuhk03_new_protocol_config_labeled.mat');
    features = jsondecode(fileread('CW2_data\PR_data\feature_data.json'));
    
    query_features = features(query_idx,:);
    gallery_features = features(gallery_idx,:);
    train_features = features(train_idx,:);

%% Creating Validation set from training set
    
    numlabels = 100;
    %Validation set consists of 100 randomly selected individuals for whom we take all of their pics
    validationlabels = datasample(unique(labels(train_idx)),numlabels,'Replace',false);
    training_idx = train_idx; %Indices in features for training
    validationQuery_idx = []; %Indices in features for validation query
    validationGallery_idx = []; %Indices in features for validation gallery
    indx = []; %For deleting the validation features from the training ones
    
    for n = 1:length(validationlabels)
        indx = [indx;find(labels(training_idx) == validationlabels(n))];
                
        temp = find(labels == validationlabels(n));
        if(length(temp)~=10)
            validationGallery_idx = [validationGallery_idx; temp(1:floor(0.7*length(temp)))]; %Indices in train_idx that are for validation too
            validationQuery_idx = [validationQuery_idx; temp(ceil(0.7*length(temp)):end)];
        else
            validationGallery_idx = [validationGallery_idx; temp(1:7)]; %Indices in train_idx that are for validation too
            validationQuery_idx = [validationQuery_idx; temp(8:end)];
        end
    end
    training_idx(indx) = [];
    clear temp; clear n; clear validationlabels;
    
    training_features = features(training_idx,:);
    validationQuery_features = features(validationQuery_idx,:);
    validationGallery_features = features(validationGallery_idx,:);   
    
%% k-NN Query vs Gallery Baseline

    scoreskNNbaseline = Rankscores(gallery_PCA, query_PCA, gallery_idx, query_idx, labels, camId, 0);
    figure;
    plot(1:1:30, scoreskNNbaseline);
    ylabel('Rank Score / %');
    xlabel('Rank');

%% kMeans Query vs Gallery
 
    [idx, C] = kmeans(gallery_features,length(unique(labels(gallery_idx))),'Replicates',100);
    %Idx contains the cluster indices of each point, C contains the centers
  
    clusterlabels = zeros(size(C,1),1); %Store majority voted cluster labels into
    for cluster = 1:size(C,1)
        indices = find(idx == cluster); %find indices in idx that have that cluster label
        clusterlabels(cluster) = mode(labels(gallery_idx(indices))); %Majority vote label for each cluster
    end

    knnkmeans = knnsearch(C,query_kPCA,'K',10); %Standard euclidian metric

    scoreskmeans = zeros(length(query_idx),3);
    for q = 1:length(query_idx)
        a = 1;
        for rank = [1,5,10]
            for img = 1:rank
                if(clusterlabels(knnkmeans(q,img)) == labels(query_idx(q)))
                    scoreskmeans(q,a) = 1;
                    break;
                end
            end
            a = a + 1;
        end
    end
    clear q; clear a; clear img; clear rank; clear cluster; clear indices; clear clusterlabels;
    scoreskmeans = sum(scoreskmeans,1)/length(query_idx)*100;
  
    plot(1:1:5, scoreskmeans);
    ylabel('Rank Score / %');
    xlabel('Rank');
    

%% k-NN validation query vs validation gallery to decide M_PCA hyperparameter value

    [U_PCA, diffimg, mean_feature] = PCA(training_features');

    for M_PCA = 10:10:200
        validationQuery_PCA = U_PCA(:,1:M_PCA)'*(validationQuery_features' - mean_feature);
        validationGallery_PCA = U_PCA(:,1:M_PCA)'*(validationGallery_features' - mean_feature);
        tic;
        scoresVAL(M_PCA/10,:) = Rankscores(validationGallery_PCA',validationQuery_PCA', validationGallery_idx, validationQuery_idx, labels, camId, 0);
        timetaken(M_PCA/10) = toc;
    end
    
    figure;
    hold on;
    yyaxis left;
    plot(10:10:M_PCA,scoresVAL(:,1));
    plot(10:10:M_PCA,scoresVAL(:,5));
    plot(10:10:M_PCA,scoresVAL(:,10));
    ylabel('kNN Recognition accuracy / %');
    xlabel('Nr of Eigenvectors used');
    yyaxis right;
    plot(10:10:M_PCA,timetaken);
    ylabel('kNN Computational time / s');
    legend('Rank1 score','Rank5 score','Rank10 score');%,'Computational time');
    hold off;

    clear M_PCA;
    
    %50 was found to be optimal, so we'll use that
    query_PCA = (U_PCA(:,1:50)'*(query_features' - mean_feature))';
    gallery_PCA = (U_PCA(:,1:50)'*(gallery_features' - mean_feature))';
    
%% k-NN validation query vs validation gallery to decide sigma hyperparameter value for Gaussian Kernel

    M_PCA = 50;
    for para = 5:5:50
        [train_kPCA, U_kPCA, D_kPCA] = kPCA(training_features,M_PCA,'gaussian',para);

        validationQuery_kPCA = validationQuery_features*diffimg*U_kPCA(:,1:M_PCA)*diag(D_kPCA(1:M_PCA))^(-1/2);
        validationGallery_kPCA = validationGallery_features*diffimg*U_kPCA(:,1:M_PCA)*diag(D_kPCA(1:M_PCA))^(-1/2);
        scoreskNNkPCA(para/5,:) = Rankscores(validationGallery_kPCA, validationQuery_kPCA, validationGallery_idx, validationQuery_idx, labels, camId,0);
        display(num2str(para));
    end  
    clear para;

    % para = 40 decided upon, so we'll use that

    [train_kPCA, U_kPCA, D_kPCA] = kPCA(training_features,M_PCA,'gaussian', 40);
    query_kPCA = query_features*diffimg*U_kPCA(:,1:M_PCA)*diag(D_kPCA(1:M_PCA))^(-1/2);
    gallery_kPCA = gallery_features*diffimg*U_kPCA(:,1:M_PCA)*diag(D_kPCA(1:M_PCA))^(-1/2);
    
%% Largest Margin Nearest Neighbour Classifier

    run('LMNNCODE\mlcircus-lmnn-5b49cafaeb9a\lmnn3\setpaths3.m');
    
    S = zeros(length(train_idx)); %Similarity matrix
    D = zeros(length(train_idx)); %Dissimilarity matrix:
    for i = 1:length(S)
        for j = 1:length(S)
            if(labels(train_idx(i)) == labels(train_idx(j)))
                S(i,j) = 1; 
            else
                D(i,j) = 1;
            end
        end
    end
    
    
%--------------------------------------------------------------------------Running LMNN on PCA data
    iterations = 5:5:495; %Iterations to save the learnt matrices at
    gpuDevice(1); %Run code on the GPU since its faster
    
    %Learning Features now on entire training set
   % [U_PCA, diffimg, mean_feature] = PCA(train_features');
    %train_PCA = U_PCA(:,1:50)'*diffimg;
    tic;
    [L,Det,matrices] = lmnnCG(train_features',labels(train_idx,:)',4,iterations,'maxiter',500,'GPU',true);  
    timetaken = toc;
    for n = 1:length(matrices)
        Ltemp1 = [matrices{n}];
        Ltemp2 = zeros(50,50);
        row = 1;
        col = 1;
        for i = 1:length(Ltemp1)
            Ltemp2(row,col) = Ltemp1(i);
            if(col == 50)
                col = 1;
                row = row + 1;
            else
                col = col + 1;
            end
        end
        matricesPCA{n} = Ltemp2;
    end

    matricesPCA{end + 1} = L; %Found transformation matrices A saved at different iterations to find the optimal one
%%
    a = 1;        
    for n = 1:length(matricesPCA)  
        scoresLMNN(a,:) = Rankscores(gallery_kPCA, query_kPCA, gallery_idx, query_idx, labels, camId,nearestSPD([matricesPCA{n}]));
        a = a + 1;
    end
       
    %%
%--------------------------------------------------------------------------Running LMNN on gaussian kernel PCA data

    [train_kPCA, U_kPCA, D_kPCA] = kPCA(train_features, 50 ,'gaussian', 40);
    iterations = 5:5:495;  
    
    gpuDevice(1);
    [L,Det,matrices] = lmnnCG(train_kPCA',labels(train_idx,:)',4,iterations,'maxiter',500,'GPU',true);
    
    for n = 1:length(matrices)
        Ltemp1 = [matrices{n}];
        Ltemp2 = zeros(50,50);
        row = 1;
        col = 1;
        for i = 1:length(Ltemp1)
            Ltemp2(row,col) = Ltemp1(i);
            if(col == 50)
                col = 1;
                row = row + 1;
            else
                col = col + 1;
            end
        end
        matricesKPCA{n} = Ltemp2;
    end

    matricesKPCA{end + 1} = L;
    
    a = 1;        
    for n = 1:length(matricesKPCA)  
        scoresLMNNkpca(a,:) = Rankscores(gallery_kPCA, query_kPCA, gallery_idx, query_idx, labels, camId,nearestSPD([matricesKPCA{n}]));
        a = a + 1;
    end

    %Getting scores for ranks 1, 5 and 10 only. 
    scoresLMNNpcaFINAL = scoresLMNNpca(:,[1,5,10]);
    scoresLMNNkpcaFINAL = scoresLMNNkpca(:,[1,5,10]);

%% Improved kmeans - with LMNN
[idx, C] = kmeansNEW(gallery_kPCA,length(unique(labels(gallery_idx))),100,nearestSPD([matricesKPCA{25}]));
knnkmeans = knnsearch(C,query_kPCA,'K',10); %Standard euclidian metric
clusterlabels = zeros(size(C,1),1); %Store majority voted cluster labels into

for cluster = 1:size(C,1)
    indices = find(idx == cluster); %find indices in idx that have that cluster label
    clusterlabels(cluster) = mode(labels(gallery_idx(indices))); %Majority vote label for each cluster
end

scoreskmeansimproved = zeros(length(query_idx),3);
for q = 1:length(query_idx)
    a = 1;
    for rank = [1,5,10] 
        for img = 1:rank
            if(clusterlabels(knnkmeans(q,img)) == labels(query_idx(q)))
                scoreskmeansimproved(q,a) = 1;
                break;
            end
        end
        a = a + 1;
    end
end

scoreskmeansimproved = sum(scoreskmeansimproved,1)/length(query_idx)*100;

%% Plotting the projections of some randomly sampled data points from training and gallery
numlabels = 4; %number of labels for which to plot all the data points

gallerypointslabels = randsample(unique(labels(train_idx)),numlabels,false);

figure;
view(3);
hold on;
for n = 1:numlabels
    gallerypointsidx = find(labels(train_idx) == gallerypointslabels(n));
    gallerypoints = train_features(gallerypointsidx,:);
    for i = 1:size(gallerypoints,1)
        if(n == 1)
            plot3(gallerypoints(i,1), gallerypoints(i,2), gallerypoints(i,3),'b*');
        elseif(n==2)
            plot3(gallerypoints(i,1), gallerypoints(i,2), gallerypoints(i,3),'rX');
        elseif(n==3)
            plot3(gallerypoints(i,1), gallerypoints(i,2), gallerypoints(i,3),'gO');
        else
            plot3(gallerypoints(i,1), gallerypoints(i,2), gallerypoints(i,3),'kO');
        end
    end
end
hold off;
xlabel('u1');
ylabel('u2');
zlabel('u3');

gallerypointslabels = randsample(unique(labels(train_idx)),numlabels,false);

figure;
view(3);
hold on;
for n = 1:numlabels
    gallerypointsidx = find(labels(train_idx) == gallerypointslabels(n));
    gallerypoints = train_features(gallerypointsidx,:);
    for i = 1:size(gallerypoints,1)
        if(n == 1)
            plot3(gallerypoints(i,1), gallerypoints(i,2), gallerypoints(i,3),'b*');
        elseif(n==2)
            plot3(gallerypoints(i,1), gallerypoints(i,2), gallerypoints(i,3),'rX');
        elseif(n==3)
            plot3(gallerypoints(i,1), gallerypoints(i,2), gallerypoints(i,3),'gO');
        else
            plot3(gallerypoints(i,1), gallerypoints(i,2), gallerypoints(i,3),'kO');
        end
    end
end
hold off;
xlabel('u1');
ylabel('u2');
zlabel('u3');


%% Plotting example query images and their results
qur = (features(query_idx,:));
gal = (features(gallery_idx,:));
[scores,knearest] = Rankscores(gal, qur, gallery_idx, query_idx,labels, camId, 0); %NOTE modified rankscores funciton
%temporarily to give kNN results as an outpiut
%
    person2test = [3,25,342]; %query images to show results for
    row = 1;
    [ha,pos] = tight_subplot(3,10,[.01 .01],[.05 .05],[.01 .01]);
    for pers = person2test
        tempresults = knearest(pers,:); %Selecting k-NN the results for the person2test-th query image
        axes(ha(1+(row-1)*10));
        imgtemp = imread("CW2_data\PR_data\images_cuhk03\" + cell2mat(filelist(query_idx(pers))));
        imgtemp = addborder(imgtemp,10,[0,0,0],'outer');
        imshow(imgtemp);
        axis off
        if(row == 1)
            title('Query');
        end
        
        for img = 2:10
            axes(ha(img + (row-1)*10));
            imgtemp = imread("Cw2_data\PR_data\images_cuhk03\" + cell2mat(filelist(gallery_idx(knearest(pers,img-1)))));
            if(labels(query_idx(pers)) == labels(gallery_idx(knearest(pers,img-1))) && camId(query_idx(pers)) == camId(gallery_idx(knearest(pers,img-1))))
                imgtemp = addborder(imgtemp,10,[0,0,255],'outer');
            end
            imshow(imgtemp);
            axis off;
            if(row == 1)
                if(img == 2)
                    title(num2str(img-1)+"st NN");
                elseif(img ==3)
                    title(num2str(img-1)+"nd NN");
                elseif(img == 4)
                    title(num2str(img-1)+"rd NN");
                else
                    title(num2str(img-1)+"th NN");
                end 
            end
        end
        %truesize;
        row = row + 1;
    end