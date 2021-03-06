%Forrest type
clc;clear;
%forest-type
% read everything into one cell array
[~,~,data] = xlsread('forest_type.csv', 'A2:AB524');
%select out class and sample
data_class = data(:, size(data,2));
data(:,size(data,2)) = [];
%convert to matrix
data_feature = cell2mat(data);
classes = unique(data_class);
% data_type = zeros(length(data_class), length(classes));
data_x = cell(1,length(classes)); data_y = cell(1,length(classes));

for i = 1 : length(classes)
    inx_type = strcmp(data_class, classes(i)); %convert string labels into number labels
    data_num_class(inx_type,:) = i;
    index_unprocessed(:,i) = strfind(data_class, classes(i));
    rows_type = find(~cellfun('isempty',index_unprocessed(:,i))); %the rows of each classes in original dataset
    data_y{i} = data_num_class(rows_type,:);
    data_x{i} = data_feature(rows_type,:);
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
%         [nsample, nfeature] = size(training);
%         training_x = training(:, 1 : nfeature - 1); training_y = training(:, nfeature);
%         test_x = test(:, 1 : nfeature - 1); test_y = test(:, nfeature);
    end
        t_y_inx = cvpartition(training_y, 'Kfold', 10); %kfold partitions with equally proportional classes
        % train naive bayes classifier & use cross validation
    for i = 1 : t_y_inx.NumTestSets
        inx_training = t_y_inx.training(i);
        inx_validation = t_y_inx.test(i);
        training_cv_x = training_x(inx_training,:);
        training_cv_y = training_y(inx_training);
        validation_cv_x = training_x(inx_validation,:); %cross validation sets stemed from training
        validation_cv_y = training_y(inx_validation);
        Mdl = fitcnb(training_cv_x, training_cv_y); % train naive bayes classifier
        predicting_cv_y = predict(Mdl, validation_cv_x);
        err(i) = sum(~strcmp(predicting_cv_y, validation_cv_y));
    end
    error = sum(err)/sum(t_y_inx.TestSize);
    predicting_test = predict(Mdl, test_x);
    error_test(ratio) = 1 - sum(predicting_test == test_y)/length(test_x);
end
dbstop if error

