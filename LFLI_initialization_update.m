% The folow parameter settings are suggested to reproduct most of the 
% experimental results of LFLI, and a better performance will be obtained 
% by tuning the parameters.
function [optmParameter, modelparameter] =  LFLI_initialization_update

    optmParameter.lambda3   = 10^2;  %  positive and negative correlation bridge ||CY-YP||^2
    optmParameter.lambda4   = 10^2; %  instance correlation ||YP - Y||^2
    
    optmParameter.lambda1   = 10^-5; %  regularization of W
    optmParameter.lambda2   = 10^-3; %  regularization of P
%     optmParameter.lambda4   = 10^-3; %  regularization of N 
    optmParameter.alpha     = 10^-3; %  regularization of C
    
    optmParameter.lambda5   = 10^-5; %  regularization of second-order tr((XW)^T L_c (CX))
    optmParameter.lambda6   = 10^-5; %  regularization of second-order tr(W L_p W^T)
    
    
    optmParameter.rho       = 1;     % 2^{0,1,2,3}
    optmParameter.isBacktracking    = 1; % 0 - LFLI, 1 - LFLI
    
    optmParameter.k                 = 10;
    optmParameter.eta               = 10;
    optmParameter.maxIter           = 30;
    optmParameter.minimumLossMargin = 0.01;
    optmParameter.tuneParaOneTime   = 1;
    
   %% Model Parameters
    modelparameter.cv_num            = 5;
    modelparameter.repetitions       = 10;
end



