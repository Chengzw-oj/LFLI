
function modelC = UpdateC(X, optmParameter)
%   Multi-Label Learning for Label-Specific features using Correlation Information with Missing Label
% 
%    Syntax
%
%       [model] = LFLI_update( X, Y, optmParameter)
%
%    Input
%       X               - a n by d data matrix, n is the number of instances and d is the number of features 
%       optmParameter   - the optimization parameters for LFLI, a struct variable with several fields, 
%
%    Output
%
%       model    -  a structure variable composed of the model coefficients

   %% optimization parameters

    alpha            = optmParameter.alpha   ; %  regularization of C
  
    
    rho              = optmParameter.rho;
    eta              = optmParameter.eta;
    isBacktracking   = optmParameter.isBacktracking;
    maxIter          = optmParameter.maxIter;
    miniLossMargin   = optmParameter.minimumLossMargin;

    num_instance   = size(X,1);
    num_dim   = size(X,2);

    XXT = X*X';
    XTX = X'*X;
   %% initialization
    C_temp   = (XXT + rho*eye(num_instance)) \ (XXT); %zeros(num_instances,num_instances); %
    C=C_temp-diag(diag(C_temp));
    CTC = C'*C;
    C_1 = C; 
    C_k = C;

    iter = 1; oldloss = 0;
    bk = 1; bk_1 = 1; 
   %% solution of instance correaltion : initialization Lip_c
    Lip_c = norm(XXT)^2;
    
    while iter <= maxIter
       if isBacktracking == 0
            if alpha>0
                Lip = sqrt( Lip_c);
            end
       else
           F_v = calculateF(X, C, CTC, XTX);
           QL_v = calculateQ(X, C, CTC, XTX, XXT, Lip_c, C_k);
           while F_v > QL_v
               Lip = eta*Lip_c;
               QL_v = calculateQ(X, C, CTC, XTX, XXT, Lip_c, C_k);
           end
       end
    
      %% update C
       C_k  = C + (bk_1 - 1)/bk * (C - C_1);
       Gc_x_k = C_k - 1/Lip_c * gradientOfC(C, XXT);
       C_1  = C;
       C    = softthres(Gc_x_k,alpha/Lip_c);
       
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;
      
      %% Loss
       Loss = C*X - X;
       DiscriminantLoss = trace(Loss'* Loss);
       
       sparesC   = sum(sum(C~=0));
      
       totalloss = DiscriminantLoss + alpha*sparesC;
       loss(iter,1) = totalloss;
       if abs((oldloss - totalloss)/oldloss) <= miniLossMargin
           break;
       elseif totalloss <=0
           break;
       else
           oldloss = totalloss;
       end
       iter=iter+1;
    end

    modelC.C = C;
    modelC.loss = loss;
    modelC.optmParameter = optmParameter;
end

%% soft thresholding operator
function C = softthres(C_t,lambda)
    C = max(C_t-lambda,0) - max(-C_t-lambda,0);  
end

function gradient = gradientOfC(C, XXT)
    gradient = C*XXT-XXT;
end

function F_v = calculateF(X, C, CTC, XTX)
% calculate the value of function F(\Theta)
    F_v = 0;
    F_v = F_v + 0.5*trace(X'*CTC*X-2*X'*C'*X + XTX);
end

function QL_v = calculateQ(X, C, CTC, XTX, XXT, Lip, C_t)
    QL_v = 0;
    QL_v = QL_v + calculateF(X, C_t, CTC, XTX);
    QL_v = QL_v + 0.5*Lip*norm(C - C_t,'fro')^2;
    QL_v = QL_v + trace((C - C_t)'*gradientOfC(C_t, XXT));
end
