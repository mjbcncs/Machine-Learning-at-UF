%Hand Writing Digits
clc;clear;
% read everything into one cell array
data = xlsread('hand_writing_digits');
%select out class and sample
data_class = data(:, size(data,2));
data(:,size(data,2)) = [];
%convert to matrix
data_feature = data;
classes = unique(data_class);
data_x = cell(1,length(classes)); data_y = cell(1,length(classes));
% create feature matrix and class matrix
for i = 1 : length(classes)
    inx = data_class == classes(i);
    data_y{i} = data_class(inx,:);
    data_x{i} = data_feature(inx,:);
end

for ratio = 1 : 5
training_x = []; training_y = [];
test_x = []; test_y = [];
    %create training, test & validation
    for i = 1 : length(classes)
        data_x_mat = cell2mat(data_x(i)); %convert training feature from cell to matrix
        data_y_mat = cell2mat(data_y(i));
        [t_type, inx_type] = datasample(data_x_mat, round(length(data_x_mat)*(6 - ratio)/10), 1, 'replace', false);
        training_x = [training_x; t_type];
        training_y = [training_y; data_y_mat(inx_type)];
        data_backup_x = data_x_mat; data_backup_y = data_y_mat;
        data_backup_x(inx_type,:) = []; data_backup_y(inx_type,:) = [];
        test_x = [test_x; data_backup_x];
        test_y = [test_y; data_backup_y];
    end
        t_y_inx = cvpartition(training_y, 'Kfold', 10); %kfold partitions with equally proportional classes
        % train naive bayes classifier & use cross validation
        mu_sigma = cell(t_y_inx.NumTestSets,1);
    for i = 1 : t_y_inx.NumTestSets
        P = [];
        pv = [];
        inx_training = t_y_inx.training(i);
        inx_validation = t_y_inx.test(i);
        training_cv_x = training_x(inx_training,:);
        training_cv_y = training_y(inx_training);
        validation_cv_x = training_x(inx_validation,:); %cross validation sets stemed from training
        validation_cv_y = training_y(inx_validation);
        nc=length(classes); % number of classes
        ni=size(training_cv_x,2); % number of independent variables
        ns=size(validation_cv_x,1); % cross validation set length
        % compute class probability
        for j=1:nc
            fy(j)=sum(double(training_cv_y == classes(j)))/length(training_cv_y);
        end
        % normal distribution
        % parameters from training set
        for j=1:nc
            xi=training_cv_x((training_cv_y == classes(j)),:);
            mu(j,:)=mean(xi,1);
            sigma(j,:)=std(xi,1);
        end
        % probability for test set
        for j=1:ns
            fu=normcdf(ones(nc,1)*validation_cv_x(j,:),mu,sigma);
            P(j,:)=fy.*prod(fu,2)';
        end
        % get predicted output for validation set
        [pv0,id]=max(P,[],2);
        for j=1:length(id)
            pv(j,1)=classes(id(j));
        end
        % compare predicted output with actual output from validation data
        confMat=myconfusionmat(validation_cv_y,pv);
        disp('confusion matrix:')
        disp(confMat)
        conf_validation(i)=sum(pv==validation_cv_y)/length(pv);
        disp(['accuracy = ',num2str(conf_validation(i)*100),'%'])
        mu_sigma(i) = {[mu, sigma]};
    end
    error = 1 - sum(conf_validation)/10; % cross validation error
    [max_conf, max_inx] = max(conf_validation);
    mu_sigma = mu_sigma{max_inx,:}; %retrieve the stored mu and sigma
%     mu_sigma = cell2mat(mu_sigma); %convert to matrix
    mu = mu_sigma(:, 1:ni); sigma = mu_sigma(:, ni + 1:2*ni);
    %test set
    for j=1:size(test_x,1)
        fu=normcdf(ones(nc,1)*test_x(j,:),mu,sigma);
        P(j,:)=fy.*prod(fu,2)';
    end
    % get predicted output for test set
    pt = [];
    [pt0,id]=max(P,[],2);
    for j=1:length(id)
        pt(j,1)=test_y(id(j));
    end
    % compare predicted output with actual output from test data
    confMat=myconfusionmat(test_y,pt);
    disp('confusion matrix:')
    disp(confMat)
    conf_test(ratio)=sum(pt==test_y)/length(pt);
    disp(['accuracy = ',num2str(conf_test*100),'%'])
end
percentages = [5:1:9]*0.1;
plot(percentages, 1-conf_test)
dbstop if error


