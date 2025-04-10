function [ tmpResult,modelLFLI ] = lfli_main_update( train_data,train_target,test_data,test_target )
 %% Set parameter    
    [optmParameter, ~] =  LFLI_initialization_update;% parameter settings for LFLI
    optmParameter_LFLI = optmParameter;
    train_target(train_target~=1) = 0 ;
    test_target(test_target~=1) = 0 ;
 %% Training
    modelC = UpdateC(train_data, optmParameter_LFLI);
    modelLFLI  = LFLI_update( train_data, train_target, modelC.C, optmParameter_LFLI); 
 %% Prediction and evaluation LFLI
    Outputs = (test_data*modelLFLI.W)';
    fscore                 = (train_data*modelLFLI.W)';
    [ tau,  ~] = TuneThreshold( fscore, train_target', 1, 2);
    Pre_Labels             = Predict(Outputs,tau);
    fprintf('-- Evaluation LFLI\n');
    tmpResult = EvaluationAll(Pre_Labels,Outputs,test_target');
end
