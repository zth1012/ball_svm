%---------------------------------------------------------------------------------------
% Settings
%---------------------------------------------------------------------------------------
PosImageNetID = 779;
llc_test_feat_path = '../../data/LLC_feats/test_llc_spm/';
test_gt_file = '../../data/ILSVRC2010_test_ground_truth.txt';
load '../../data/meta.mat';
pyramid_size=5120;
%number of classes
K=1000;

%---------------------------------------------------------------------------------------
% Prepare test data
%---------------------------------------------------------------------------------------
test_size=150000;
% gt=dlmread(test_gt_file);
% cls_idx = (gt==PosImageNetID);
% gt(cls_idx) = 1;
% gt(~cls_idx) = -1;
TestLabels = dlmread(test_gt_file);
TestFeatures=zeros(test_size,pyramid_size);

disp('loading test data...');
start=1;
mysize=1000;
for i=1:150
	S = load([llc_test_feat_path,'test.',num2str(i,'%.4d'),'.vldsift.mat']);
%     B = load([llc_test_feat_path,'test.',num2str(i,'%.4d'),'.sbow.mat']);
% 	assert(numel(B.image_sbow)==mysize);
	for j=1:mysize
		%convert to bag of words histogram
% 		x = histc(S.image_sbow(j).sbow.word,0:pyramid_size-1);
% 		x = x / norm(x);
		TestFeatures(start+j-1,:) = S.pyramid(j,:);
	end
	start = start + mysize;
	clear S;
end
TestFeatures = sparse(TestFeatures);
fname = ['test_data.mat'];
fprintf('Size of TestFeatures: %d by %d\n', size(TestFeatures,1), size(TestFeatures,2));
save(fname,'TestLabels', 'TestFeatures', '-v7.3');