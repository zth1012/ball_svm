%---------------------------------------------------------------------------------------
% Settings
%---------------------------------------------------------------------------------------
llc_val_feat_path = '../../data/LLC_feats/val_llc_spm/';
bow_val_feat_path = '../../data/features/val/';
val_gt_file = '../../data/ILSVRC2010_validation_ground_truth.txt';
load '../../data/meta.mat';
pyramid_size=5120;
%number of classes
K=1000;

%---------------------------------------------------------------------------------------
% Prepare validation data
%---------------------------------------------------------------------------------------

validation_size=50000;
% gt=dlmread(val_gt_file);
% cls_idx = (gt==PosImageNetID);
% gt(cls_idx) = 1;
% gt(~cls_idx) = -1;
ValidLabels = dlmread(val_gt_file);
ValidFeatures=zeros(validation_size,pyramid_size);

disp('loading validation data...');
start=1;
mysize=1000;
for i=1:50
	S = load([llc_val_feat_path,'val.',num2str(i,'%.4d'),'.vldsift.mat']);
    B = load([bow_val_feat_path,'val.',num2str(i,'%.4d'),'.sbow.mat']);
	assert(numel(B.image_sbow)==mysize);
	for j=1:mysize
		%convert to bag of words histogram
% 		x = histc(S.image_sbow(j).sbow.word,0:pyramid_size-1);
% 		x = x / norm(x);
		ValidFeatures(start+j-1,:) = S.pyramid(j,:);
	end
	start = start + mysize;
	clear S;
end
ValidFeatures = sparse(ValidFeatures);
fname = ['valid_data.mat'];
ValidLabels = ValidLabels;
ValidFeatures = ValidFeatures;
save(fname,'ValidLabels', ...
'ValidFeatures');