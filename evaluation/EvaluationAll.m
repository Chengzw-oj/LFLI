
function ResultAll = EvaluationAll(Pre_Labels,Outputs,test_target)
% evluation for MLC algorithms, there are fifteen evaluation metrics
% 
% syntax
%   ResultAll = EvaluationAll(Pre_Labels,Outputs,test_target)
%
% input
%   test_targets        - L x num_test data matrix of groundtruth labels
%   Pre_Labels          - L x num_test data matrix of predicted labels
%   Outputs             - L x num_test data matrix of scores
%
    
    Pre_Labels(Pre_Labels~=1)=0 ;
    test_target(test_target~=1)=0 ;
    ResultAll=zeros(5,1); 

    SubsetAccuracy = SubsetAccuracyEvaluation(test_target,Pre_Labels);
    [~,LabelBasedPrecision,LabelBasedRecall,LabelBasedFmeasure] = LabelBasedMeasure(test_target,Pre_Labels);
    MicroF1Measure      = MicroFMeasure(test_target,Pre_Labels);
    
    ResultAll(1,1)  = SubsetAccuracy;
    ResultAll(2,1)  = LabelBasedPrecision;
    ResultAll(3,1)  = LabelBasedRecall; 
    ResultAll(4,1) = LabelBasedFmeasure; 
    ResultAll(5,1) = MicroF1Measure;

    Pre_Labels(Pre_Labels~=1)=-1 ;
    test_target(test_target~=1)=-1 ;
end