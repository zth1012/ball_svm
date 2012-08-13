% close all
% clear all

%  pos_IL_ID = 779;     class = 'Airship';
% pos_IL_ID = 449;        class = 'Lion';
% pos_IL_ID = 761;    class = 'Monitor';
% pos_IL_ID = 811;   class = 'Armchair';
% pos_IL_ID = 818;    class = 'Diningtable';

% pos_IL_ID = 563;    class = 'Keyboard';
% pos_IL_ID = 837;    class = 'Church';
% pos_IL_ID = 22;     class = 'Banana';
pos_IL_ID = 106;    class = 'Sunflower';

TestImgPath = '../../data/test_img/test/';
model_folder = 'svm_models/';
result_folder = 'svm_test_results/';
% Number of training examples from each negative category
NegTrainPerCat = [1 2 5 10 20 40 80 160 200];
% Number of top testing images to extract
NumTopImgs = 50;
% Row and column sizes for top images saved
ResRow = 60;
ResCol = 80;

load 'test_data.mat';

path(path,'../3rd-party/liblinear-1.8/matlab');

for K = 1 : length(NegTrainPerCat)
% 	load(['train_data_neg_' num2str(NegTrainPerCat(K)) '.mat']);
	load([model_folder class '_best_model_neg_' num2str(NegTrainPerCat(K)) '.mat']);

    % Test on testset
	[PredIgnore, AccIgnore, TestDecValues] = predict(TestLabels, ...
		TestFeatures, ...
		BestModel);

	TestAP = eval_ap(TestDecValues, TestLabels, pos_IL_ID);
	fprintf('NegTrainPerCat = %d, TestAP = %f\n', NegTrainPerCat(K), TestAP);
	
	% Extract top images based on the decision scores
	[Score, Ind] = sort(TestDecValues, 'descend');
    
    ResRow = 60;
    ResCol = 80;
    TopImgs = zeros(ResRow,ResCol,3,NumTopImgs);
    for t = 1 : NumTopImgs
        Fname = [TestImgPath, 'ILSVRC2010_test_', sprintf('%08d', Ind(t)), '.JPEG'];
        I = imread(Fname);
        if(size(I,3) == 1)
            I = imresize(I,[ResRow ResCol]);
            TopImgs(:,:,1,t) = I;
            TopImgs(:,:,2,t) = I;
            TopImgs(:,:,3,t) = I;
        else
            TopImgs(:,:,:,t) = imresize(I,[ResRow ResCol]);
        end
    end
    Fname = [result_folder class '_test_results_neg_' num2str(NegTrainPerCat(K)) '.mat'];
    save(Fname, 'TopImgs', 'TestAP', 'TestDecValues');
end