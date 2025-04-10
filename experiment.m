clc;clear;
totalCV = 5;
repetitions = 10;
totalPercent = 0.4;

tic

%% 加载数据库
[ temp_data,target,num_data] = loadDataBase('flags');

%% 为每个数据创建缺失标签不同程度
for percent=0.1:0.1:totalPercent 
   %% 设置算法最终的数据结果记录矩阵
   Algorithm_cvResult  = zeros(5,totalCV);

   %% 开始仿真
    for repetition=1:repetitions %重复十次
        for cv=1:totalCV %进行五折
            [train_data,train_target,IncompleteTarget,J,test_data,test_target] = createData( ...
                temp_data,target,num_data,cv,totalCV,percent ); %获取五折后的数据以及缺失的标签数据集

           [ tmpResult,modelLFLI ] = lfli_main_update( train_data,IncompleteTarget,test_data,test_target ) ;
           %% 记录结果
            Algorithm_cvResult(:,cv)  = Algorithm_cvResult(:,cv) + tmpResult ;
        end
    end
    %% 计算数据的平均值
    [ cvResult ]   = cptavg( Algorithm_cvResult,repetitions ) ;
    PrintResults(cvResult)

   
    clear temp_data target train_data train_target IncompleteTarget J test_data test_target num_data
end
toc