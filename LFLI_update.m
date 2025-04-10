
function model = LFLI_update( X, Y, C, optmParameter)
%   Multi-Label Learning for Label-Specific features using Correlation Information with Missing Label
% 
%    Syntax
%
%       [model] = LFLI_update( X, Y, optmParameter)
%
%    Input
%       X               - a n by d data matrix, n is the number of instances and d is the number of features 
%       Y               - a n by l label matrix, n is the number of instances and l is the number of labels
%       optmParameter   - the optimization parameters for LFLI, a struct variable with several fields, 
%
%    Output
%
%       model    -  a structure variable composed of the model coefficients

   %% optimization parameters
    lambda3 = optmParameter.lambda3;  %  positive and negative correlation bridge ||CY-YP||^2
    lambda4 = optmParameter.lambda4; %  instance correlation ||YP - Y||^2
    
    lambda1 = optmParameter.lambda1; %  regularization of W
    lambda2 = optmParameter.lambda2; %  regularization of P
    
    lambda5 = optmParameter.lambda5; %  regularization of second-order tr((XW)^T L_c (CX))
    lambda6 = optmParameter.lambda6;
    
    rho              = optmParameter.rho;
    eta              = optmParameter.eta;
    isBacktracking   = optmParameter.isBacktracking;
    maxIter          = optmParameter.maxIter;
    miniLossMargin   = optmParameter.minimumLossMargin;

    num_instance   = size(X,1);
    num_dim   = size(X,2);
    num_class = size(Y,2);
    I=eye (num_instance,num_instance);
    XTX = X'*X;
    XTY = X'*Y;
    YTY = Y'*Y;
    E = (I-C)*X;
    
   %% initialization
    W   = (XTX + rho*eye(num_dim)) \ (XTY); %zeros(num_dim,num_class); %
    W_1 = W; W_k = W;
    P = zeros(num_class,num_class); %eye(num_class,num_class);  positive label
    N = zeros(num_class,num_class); %eye(num_class,num_class);  negative label

    P_1 = P;
%     N_1 = N;

    iter = 1; oldloss = 0;
    bk = 1; bk_1 = 1; 
   %% solution of object function : initialization Lip
%     Lip1 = 7*norm(XTX)^2 + 7*norm(XTY)^2 + 3*norm((1+lambda2)*YTY)^2 + 7*norm(E'*E)^2;
    Lip1 = 4*norm(XTX)^2 + 4*norm(XTY)^2 + 2*norm((lambda3+lambda4+1)*YTY)^2;
    Lip = sqrt(Lip1);
    L_c = diag(sum(C,2)) - C; % graph laplacian -label correlation L_c
    while iter <= maxIter
       L_p = diag(sum(P,2)) - P; % graph laplacian -label correlation L_p
       L_n = diag(sum(N,2)) + N; % graph laplacian -label correlation L_n
       
       if isBacktracking == 0
            if lambda5>0
                Lip2 = norm(lambda5*(X'*L_c*X))^2;
            end
             if lambda6>0
                Lip2 = norm(lambda6*(L_p+L_p'))^2 + norm(lambda5*(X'*L_c*X))^2;
                Lip = sqrt( Lip1 + 4*Lip2);
            end
       else
           F_v = calculateF(W, X, Y, XTX, XTY, YTY, P, N, C, lambda3, lambda4, lambda5, lambda6, E);
           QL_v = calculateQ(W, X, Y, XTX, XTY, YTY, P, N, C, lambda3, lambda4, lambda5, lambda6, E, Lip, W_k);
           while F_v > QL_v
               Lip = eta*Lip;
               QL_v = calculateQ(W, X, Y, XTX, XTY, YTY, P, N, C, lambda3, lambda4, lambda5, lambda6, E, Lip, W_k);
           end
       end

      %% update P
       P_k  = P + (bk_1 - 1)/bk * (P - P_1);
       Gp_k = P_k - 1/Lip * gradientOfP(YTY,X,W,Y,P,C,lambda3,lambda4);
       P_1  = P;
       P    = softthres(Gp_k,lambda2/Lip); 
       P    = max(P,0);
       
      %% update N
%        N_k  = N + (bk_1 - 1)/bk * (N - N_1);
%        Gn_k = N_k - 1/Lip * gradientOfN(YTY,X,W,Y,P,N_k,lambda2);
%        N_1  = N;
%        N    = softthres(Gn_k,lambda4/Lip); 
%        N    = max(N,0);
    
      %% update W
       W_k  = W + (bk_1 - 1)/bk * (W - W_1);
       Gw_x_k = W_k - 1/Lip * gradientOfW(X,XTX,XTY,W_k,P,N,C,E,lambda5,lambda6);
       W_1  = W;
       W    = softthres(Gw_x_k,lambda1/Lip);
       
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;
      
      %% Loss
       Loss1 = X*W - Y*P;
       Loss2 = C*Y - Y*P;
       Loss3 = Y*P - Y;
       DiscriminantLoss = trace(Loss1'* Loss1) + lambda3*trace(Loss2'* Loss2) +  lambda4*trace(Loss3'* Loss3);
       
       sparesW    = sum(sum(W~=0));
       sparesP    = sum(sum(P~=0));
       
       CorrelationLoss1  = trace(W*L_p*W');
 
       CorrelationLoss2 = trace(W'*X'*L_c*X*W);
      
       totalloss = DiscriminantLoss + lambda1*sparesW + lambda2*sparesP + lambda5*CorrelationLoss2 +lambda6*CorrelationLoss1;
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
    model.W = W;
    model.C = C;
    model.P = P;
    model.N = N;
    model.loss = loss;
    model.optmParameter = optmParameter;
end

%% soft thresholding operator
function W = softthres(W_t,lambda)
    W = max(W_t-lambda,0) - max(-W_t-lambda,0);  
end

function gradient = gradientOfW(X,XTX,XTY,W,P,N,C,E,lambda5,lambda6)
    L_p = diag(sum(P,2)) - P;
    L_n = diag(sum(N,2)) + N;
    L_c = diag(sum(C,2)) - C;
%     gradient = XTX*W - XTY*P - XTY*N + lambda5*W*(L_p + L_p') + lambda6*W*(L_n + L_n') + lambda8*(X'*L_c*X*W);
    gradient = XTX*W - XTY*P + lambda6*W*(L_p + L_p') + lambda5*(X'*L_c*X*W);
end

function gradient = gradientOfP(YTY,X,W,Y,P,C,lambda3,lambda4)
    gradient = (1+lambda3+lambda4)*YTY*P - Y'*X*W  -lambda3*Y'*C*Y -lambda4*YTY;
end

% function gradient = gradientOfN(YTY,~,~,~,P,N,lambda2)
%     gradient = lambda2*YTY*N - lambda2*YTY*P;
% end

function F_v = calculateF(W, X, Y, XTX, XTY, YTY, P, N, C, lambda3, lambda4, lambda5, lambda6, E)
% calculate the value of function F(\Theta)
    F_v = 0;
    L_p = diag(sum(P,2)) - P;
    L_n = diag(sum(N,2)) + N;
    L_c = diag(sum(C,2)) - C;
    F_v = F_v + 0.5*trace(W'*XTX*W - 2*W'*XTY*P + P'*YTY*P);
    F_v = F_v + 0.5*lambda4*trace(P'*YTY*P - 2*P'*YTY + YTY);
    F_v = F_v + 0.5*lambda3*trace(Y'*(C'*C)*Y-2*P'*Y'*C*Y+P'*YTY*P);
    F_v = F_v + lambda6*trace(W*L_p*W');
    F_v = F_v + lambda5*trace(W'*X'*L_c*X*W);
    
end

function QL_v = calculateQ(W, X, Y, XTX, XTY, YTY, P, N, C, lambda3, lambda4, lambda5, lambda6, E, Lip, W_t)
% calculate the value of function Q_L(w_v,w_v_t)
    QL_v = 0;
    QL_v = QL_v + calculateF(W_t, X, Y, XTX, XTY, YTY, P, N, C, lambda3, lambda4, lambda5, lambda6, E);
    QL_v = QL_v + 0.5*Lip*norm(W - W_t,'fro')^2;
    QL_v = QL_v + trace((W - W_t)'*gradientOfW(X,XTX,XTY,W_t,P,N,C,E,lambda5,lambda6));
end
