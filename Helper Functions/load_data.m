% This function needs to be modified for the new dataset 
function [train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes]= load_data()

    % first 154 samples are used as training set (about 90%) the rest 10%
    % are used as testing data set 
    data = readtable('data_new_extended');
    train_set = data(1:154,:);
    train_set_x_orig = train_set(:,1:end-1) ; % your train set features
    train_set_y_orig = train_set(:,end); % your train set labels

    test_set = data(155:end,:);
    test_set_x_orig = test_set(:,1:end-1);  % your test set features
    test_set_y_orig = test_set(:,end); % your test set labels

    classes = ["covid","not_covid"]; % the list of classes
     
    
    
end 
