%---------------------------------------------------------------------------------------
% Settings
%---------------------------------------------------------------------------------------

create_data_config;
pos_train_size = 800;
warp_cache_dir = 'warp_cache/';
cnt = 0;

myRandomize;
while 1
    fprintf('cnt = %d\n', cnt);
    i = randi(14, 1);
    lockdir = [warp_cache_dir 'lock_' data_config(i).class '_pos_' num2str(pos_train_size)];
    if ~exist(lockdir, 'dir')
        mkdir(lockdir);
        pos_IL_ID = data_config(i).id;
        class = data_config(i).class;
        break;
    else
        cnt = cnt + 1;
        if cnt > 10000
            return;
        end
        continue;
    end
end

neg_train_per_synset = 40;
llc_train_feat_path='../../data/LLC_feats/train_llc_spm/';
bow_train_feat_path = '../../data/features/train/';
output_folder = 'train_data/';
load '../../data/meta.mat';
pyramid_size=5120;

%number of classes
K=1000;

%---------------------------------------------------------------------------------------
% Prepare training data
%---------------------------------------------------------------------------------------
train_sizes = [synsets.num_train_images];
train_sizes = train_sizes(1:K);

train_sizes(train_sizes > neg_train_per_synset) = neg_train_per_synset;
train_sizes(pos_IL_ID) = pos_train_size;

neg_train_size = sum(train_sizes) - pos_train_size;
pos_train_labels = ones(pos_train_size, 1);
pos_train_features = zeros(pos_train_size, pyramid_size);
neg_train_labels = -ones(neg_train_size, 1);
neg_train_features = zeros(neg_train_size, pyramid_size);

pos_train_file_id = cell(pos_train_size,1);
neg_train_file_id = cell(neg_train_size,1);

disp('loading training data...');
tic

neg_train_start=1;
for i=1:K
    fprintf('i = %d\n', i);
    S = load([llc_train_feat_path,synsets(i).WNID,'.vldsift.mat']);
    B = load([bow_train_feat_path,synsets(i).WNID,'.sbow.mat']);
    if (i == pos_IL_ID)
        for j=1:pos_train_size
            %convert to bag of words histogram
%             x = histc(S.image_sbow(j).sbow.word,0:pyramid_size-1);
%             x = x / norm(x);
            pos_train_features(j,:) = S.pyramid(j,:);
            pos_train_file_id{j} = B.image_sbow(j).ID;
        end
    else
        for j=1:train_sizes(i)
            %convert to bag of words histogram
%             x = histc(S.image_sbow(j).sbow.word,0:pyramid_size-1);
%             x = x / norm(x);
            neg_train_features(neg_train_start+j-1,:) = S.pyramid(j,:);
            neg_train_file_id{neg_train_start+j-1} = B.image_sbow(j).ID;
        end
        neg_train_start = neg_train_start + train_sizes(i);
    end
    clear S;
end

PosImageNetID = pos_IL_ID;
PosTrainLabels = pos_train_labels;
PosTrainFeatures = pos_train_features;
NegTrainLabels = neg_train_labels;
NegTrainFeatures = neg_train_features;
PosTrainFileID = pos_train_file_id;
NegTrainFileID = neg_train_file_id;
NumPosTrain = pos_train_size;

PosTrainFeatures = sparse(PosTrainFeatures);
NegTrainFeatures = sparse(NegTrainFeatures);

save([output_folder class '_neg_' num2str(neg_train_per_synset) '_pos_' num2str(pos_train_size) '.mat'], ...
	'PosImageNetID', ...
	'PosTrainLabels', ...
    'PosTrainFeatures', ...
    'NegTrainLabels', ...
    'NegTrainFeatures',...
    'PosTrainFileID',...
    'NumPosTrain',...
    'NegTrainFileID',...
    '-v7.3');
toc
