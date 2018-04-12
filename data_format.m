% This code adjusts the data to the desired format for python
clear;

% This script draws the figures with recovered depth
% load the Sigma and camLoc
kernel = '3';
load(fullfile('./experiment_data/focalFlowNet_multi_mu_s/Calib_data',strcat('calibResultk',kernel,'.mat')));

expDir = './experiment_data/focalFlowNet_multi_mu_s/test';
expListing = dir(expDir);
expListing(1:2) = [];
nExp = length(expListing);

%% Setting of the setups 
psf_func = 'apodizing';
Sigma = .001;
camLoc = camLoc/1000; % in m
pix_size = 0.00000586 * 2; % in m
f = .1;
% mu_ss = [.1352118, .1430569, .1493425, .1643094, .1280030, .1197164];
mu_ss = [...
    .1351553,.1347929,...
    .1424970,.1423434,...
    .1489287,.1486476,...
    .1643094,.1636170,...
    .1276257,.1274697,...
    .1197610,.1197205];
szx_sensor = 960;
szy_sensor = 600;

randFlag = 'rand';
noiseSigma = 0.0;
num_ave = 10;

%% Load the textures
I = zeros(1000,szy_sensor,szx_sensor,3);
Loc = zeros(1000,3,3);
mu_s = zeros(1000,1);
Z_0 = camLoc/1000;

tIdx = 1;
for expIdx = 1:nExp
    folderListing = dir(fullfile(expDir, expListing(expIdx).name));
    folderListing(1:2) = [];
    nFolder = length(folderListing);
    listing = cell(nFolder,1);
    for i = 1:nFolder
        Is_b = cell(1,1);
        zs_b = zeros(1,1);
        imgIdx = 1;
        listing{i} = dir(fullfile(expDir,expListing(expIdx).name,folderListing(i).name,'*.tif'));
        offsetVal = -str2double(folderListing(i).name(1:end-2))/1000;
        fileName = cell(length(listing{i}),1);
        for j = 1:length(listing{i})
            fileName{j} = listing{i}(j).name;
            tmpRead = textscan(fileName{j},'%.1f');
            tmpRead{1} = tmpRead{1}/1000;
            Is_b{imgIdx} = im2double(...
                imread(fullfile(...
                    expDir,...
                    expListing(expIdx).name,...
                    folderListing(i).name,...
                    fileName{j}...
                )));
            zs_b(imgIdx) = tmpRead{1} +offsetVal-camLoc-mu_ss(expIdx);
            imgIdx = imgIdx + 1;
        end
        [zs_b, idx] = sort(zs_b);
        Is_b = Is_b(idx);

        % Generate all the data
        for j = 2:length(Is_b) - 1
            I1 = imnoise(Is_b{j-1},'gaussian',0,noiseSigma);
            I2 = imnoise(Is_b{j},'gaussian',0,noiseSigma);
            I3 = imnoise(Is_b{j+1},'gaussian',0,noiseSigma);

            Loc(tIdx,:,:) = [0,0,0;0,0,0;zs_b(j-1),zs_b(j),zs_b(j+1)];
            I(tIdx,:,:,:) = cat(3,I1,I2,I3);
            mu_s(tIdx) = mu_ss(expIdx);
            tIdx = tIdx + 1;
        end
    end
end
save(fullfile('./experiment_data/focalFlowNet_multi_mu_s/test_data','train.mat'),...
    'I','mu_s','Loc','Z_0','Sigma','pix_size','psf_func','szx_sensor','szy_sensor',...
    'f','num_ave','-v7.3');


