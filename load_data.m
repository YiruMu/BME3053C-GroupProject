% This function needs to be modified for the new dataset 
function [train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes]= load_data()

    % first 154 samples are used as training set (about 90%) the rest 10%
    % are used as testing data set 
    data = readtable('data_new_extended');
    train_set = data(2:154,2:end);
    train_set_x_orig = table2array(train_set(:,1:end-1)) ; % your train set features
    train_set_y_orig = string(table2array(train_set(:,end))); % your train set labels
   
    % to convert the lables to 1 and 0; 
    for i = 1:length(train_set_y_orig)
        if train_set_y_orig(i) == 'covid'  % 1 is covid and 0 is non-covid 
            train_set_y_orig(i) = 1;
        else 
            train_set_y_orig(i) = 0;
        end 
    end 
    train_set_y_orig = str2double(train_set_y_orig);
    % to normalize the traning set data 
    for i = 1: size(train_set_x_orig,2) 
        temp = train_set_x_orig(:,i);
        maximum = max(temp);
        minimum = min(temp);
        for j = 1:length(temp)
            temp(j) = (temp(j)-minimum )/(maximum - minimum);
        end 
        train_set_x_orig(:,i) = temp; 
    end 
    % train set 15 covid and 15 non-covid 
   % train_set_x = [train_set_x_orig(1:3,:);train_set_x_orig(4:6,:); train_set_x_orig(8,:);train_set_x_orig(10:12,:);
       % train_set_x_orig(14:18,:);train_set_x_orig(20:22,:);train_set_x_orig(23:36,:)];
   % train_set_y = [ train_set_y_orig(1:3,:);train_set_y_orig(4:6,:); train_set_y_orig(8,:);train_set_y_orig(10:12,:);
       % train_set_y_orig(14:18,:);train_set_y_orig(20:22,:);train_set_y_orig(23:36,:)];
    
    % extract testing data and convert them to double 
    test_set = data(155:end,2:end);
    test_set_x_orig = table2array( test_set(:,1:end-1));  % your test set features
    test_set_y_orig = string(table2array(test_set(:,end))); % your test set labels
    for i = 1:length(test_set_y_orig)
       if (test_set_y_orig(i) == 'covid')
           test_set_y_orig(i) = 1;
        else 
           test_set_y_orig(i) = 0;
        end 
    end 
    test_set_y_orig = str2double(test_set_y_orig); 
    
    % normalize the testing data 
    for i = 1: size(test_set_x_orig,2) 
        temp = test_set_x_orig(:,i);
        maximum = max(temp);
        minimum = min(temp);
        for j = 1:length(temp)
            temp(j) = (temp(j)-minimum )/(maximum - minimum);
        end 
        test_set_x_orig(:,i) = temp; 
    end 
    
    % test set 4 covid patient + 4 non-covid patient 
   % test_set_x = [ train_set_x_orig(7,:);  train_set_x_orig(9,:); train_set_x_orig(13,:); train_set_x_orig(19,:); test_set_x_orig(1:4,:)];
   % test_set_y = [ train_set_y_orig(7,:);  train_set_y_orig(9,:); train_set_y_orig(13,:); train_set_y_orig(19,:); test_set_y_orig(1:4,:)];
    
    classes = ["covid","not_covid"]; % the list of classes
     
    
    
end 
