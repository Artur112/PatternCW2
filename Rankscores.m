function scores = Rankscores(data_compare2,data,data_compare2idx, dataidx, labels, camId, A)
    %Input examples must be along the rows
    
    if(A == 0) %If A==0, perform kNN with euclidian, otherwise with learnt mahalanobis metric
        tic;
        knearest = knnsearch(data_compare2,data,'K',35); %Standard euclidian metric
        display(toc);
    else
        knearest = knnsearch(data_compare2,data,'K',35,'Distance','mahalanobis','Cov', A);%Mahalabonis distance
    end

%--------------Removing results that have the same label and camID for each queryimage ranklist
    
    knearestRemoved = zeros(length(dataidx),30); % For storing the results with images of same camera and label removed
    for n = 1:length(dataidx) %Loop through the results for each query image               
        %Remove same label and camera images for baseline
        tempresults = knearest(n,:);
        i = 1;
        indices = [];
        for r = 1:length(tempresults)
            if(labels(data_compare2idx(tempresults(r))) == labels(dataidx(n)) && camId(data_compare2idx(tempresults(r))) == camId(dataidx(n)))
                indices(i) = r;
                i = i + 1;
            end
        end
        if(~isempty(indices))
            tempresults(indices) = []; %Remove the pictures at those indices and then store in the new NN results matrix
        end
        knearestRemoved(n,:) = tempresults(1:30);
    end
%----------------Getting prediction scores from the ranklists

    ranks = 1:1:30; %Ranks to find scores for
    scores = zeros(1,length(ranks)); %Matrix for storing score results into
    
    b = 1; %Indexes for storing baseline and learned metric results into same scores matrix
    for rank = ranks
        %Get results rank scores for baseline
        count = 0; %For counting for how many query images a correct result was obtained within rank results
        for n = 1:length(dataidx)
            for c = 1:rank
                if(labels(dataidx(n)) == labels(data_compare2idx(knearestRemoved(n,c))))
                    count = count + 1;
                    break; %Only augment count once since we want to know whether
                           %the results include atleast 1 correct result
                end
            end
        end
        scores(1,b) = count/length(dataidx)*100; %Find percentage of query image results that did contain correct result
        b = b + 1;
    end        

end

