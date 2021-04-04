% This function needs to be modified for the new dataset 
function [train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes]= load_data()

   
     
    train_set_x_orig = h5read('train_catvnoncat.h5','/train_set_x'); % your train set features
    train_set_y_orig = h5read('train_catvnoncat.h5','/train_set_y'); % your train set labels

    
    test_set_x_orig = h5read('test_catvnoncat.h5','/test_set_x'); % your test set features
    test_set_y_orig = h5read('test_catvnoncat.h5','/test_set_y'); % your test set labels

    classes = h5read('test_catvnoncat.h5','/list_classes'); % the list of classes
     
    train_set_y_orig = reshape(train_set_y_orig, [1, size(train_set_y_orig,1)]);
    test_set_y_orig = reshape(test_set_y_orig, [1, size(test_set_y_orig,1)]);
    
end 
