
function PrintResults(Result)
fprintf('------------------------------------\n');
fprintf('Evalucation Metric    Mean    Std\n');
fprintf('------------------------------------\n');
fprintf('SubsetAccuracy        %.4f  %.4f\r',Result(1,1),Result(1,2));
fprintf('LabelBasedPrecision   %.4f  %.4f\r',Result(2,1),Result(2,2));
fprintf('LabelBasedRecall      %.4f  %.4f\r',Result(3,1),Result(3,2));
fprintf('LabelBasedFmeasure    %.4f  %.4f\r',Result(4,1),Result(4,2));
fprintf('MicroF1Measure        %.4f  %.4f\r',Result(5,1),Result(5,2));
fprintf('------------------------------------\n');
end