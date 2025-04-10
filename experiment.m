clc;clear;
totalCV = 5;
repetitions = 10;
totalPercent = 0.4;

tic

%% �������ݿ�
[ temp_data,target,num_data] = loadDataBase('flags');

%% Ϊÿ�����ݴ���ȱʧ��ǩ��ͬ�̶�
for percent=0.1:0.1:totalPercent 
   %% �����㷨���յ����ݽ����¼����
   Algorithm_cvResult  = zeros(5,totalCV);

   %% ��ʼ����
    for repetition=1:repetitions %�ظ�ʮ��
        for cv=1:totalCV %��������
            [train_data,train_target,IncompleteTarget,J,test_data,test_target] = createData( ...
                temp_data,target,num_data,cv,totalCV,percent ); %��ȡ���ۺ�������Լ�ȱʧ�ı�ǩ���ݼ�

           [ tmpResult,modelLFLI ] = lfli_main_update( train_data,IncompleteTarget,test_data,test_target ) ;
           %% ��¼���
            Algorithm_cvResult(:,cv)  = Algorithm_cvResult(:,cv) + tmpResult ;
        end
    end
    %% �������ݵ�ƽ��ֵ
    [ cvResult ]   = cptavg( Algorithm_cvResult,repetitions ) ;
    PrintResults(cvResult)

   
    clear temp_data target train_data train_target IncompleteTarget J test_data test_target num_data
end
toc