% This code trains the network 1
clear;

% This script draws the figures with recovered depth
% load the Sigma and camLoc
kernel = '3';
load(fullfile('data/calib_data',strcat('calibResultk',kernel,'.mat')));

expDir = 'data/train';
expListing = dir(expDir);
expListing(1:3) = [];
nExp = length(expListing);

%% Setting of the setups 
psf_func = @PSF_gauss;
camParam = struct;
Sigma = 1.01;
camParam.Sigma = [Sigma,0;0,Sigma]; 
camParam.pillRad = 0;
camParam.pixSize = 0.00586 * 2; % in mm
camParam.f = 100;
camParam.mu_s = [135.2118, 143.0569, 149.3425, 164.3094, 128.0030,119.7164];
camParam.camLoc = camLoc;
camParam.szx_sensor = 600;
camParam.szy_sensor = 600;
% by default the principal point is on the center
camParam.x_prinpts = camParam.szx_sensor / 2;
camParam.y_prinpts = camParam.szy_sensor / 2;

switch func2str(psf_func)
    case 'PSF_gauss'
        camName = sprintf('Sigma=%.4f',camParam.Sigma(1,1));
    case 'PSF_gaussConvPill'
        camName = sprintf('Sigma=%.4f,pillRad=%.4f',camParam.Sigma(1,1),camParam.pillRad);
end

randFlag = 'rand';
noiseSigma = 0.0;
textNum = 1;

%% Network Parameter
netParam = struct;
netParam.diffFilter.dunits = 1;
netParam.eqSigma = Sigma;
netParam.diffFilter.ft = [-0.5,0,0.5];
netParam.netName = @focalFlowNet1_exp;

if ~exist(fullfile([expDir,'_data'],'train.mat'));
    %% Load the textures
    Is = cell(1);
    IsT = cell(1); % transpose the image
    szx = ones(1)*camParam.szx_sensor;
    szy = ones(1)*camParam.szy_sensor;
    camParams = cell(1);
    zs = zeros(1);

    tIdx = 1;
    for expIdx = 1:nExp
        folderListing = dir(fullfile(expDir, expListing(expIdx).name));
        folderListing(1:3) = [];
        nFolder = length(folderListing);
        listing = cell(nFolder,1);
        for i = 1:nFolder
            Is_b = cell(1,1);
            zs_b = zeros(1,1);
            imgIdx = 1;
            listing{i} = dir(fullfile(expDir,expListing(expIdx).name,folderListing(i).name,'*.tif'));
            offsetVal = -str2double(folderListing(i).name(1:end-2));
            fileName = cell(length(listing{i}),1);
            for j = 1:length(listing{i})
                fileName{j} = listing{i}(j).name;
                tmpRead = textscan(fileName{j},'%.1f');
                Is_b{imgIdx} = im2double(...
                    imread(fullfile(...
                        expDir,...
                        expListing(expIdx).name,...
                        folderListing(i).name,...
                        fileName{j}...
                    )));
                zs_b(imgIdx) = tmpRead{1} +offsetVal-camLoc-camParam.mu_s(expIdx);
                imgIdx = imgIdx + 1;
            end
            [zs_b, I] = sort(zs_b);
            Is_b = Is_b(I);

            % Generate all the data
            for j = 2:length(Is_b) - 1
                I1 = imnoise(Is_b{j-1},'gaussian',0,noiseSigma);
                I2 = imnoise(Is_b{j},'gaussian',0,noiseSigma);
                I3 = imnoise(Is_b{j+1},'gaussian',0,noiseSigma);

                zs(tIdx) = zs_b(j);
                szx(tIdx) = camParam.szx_sensor;
                szy(tIdx) = camParam.szy_sensor;
                % I hardcode here
                Is{tIdx} = cat(3,I1(:,180:779),I2(:,180:779),I3(:,180:779));
                IsT{tIdx} = permute(Is{tIdx},[2,1,3]);
                camParams{tIdx} = camParam;
                camParams{tIdx}.mu_s = camParam.mu_s(expIdx);
                tIdx = tIdx + 1;
            end
        end
    end
    save(fullfile([expDir,'_data'],'train.mat'),'Is','camParams','szx','szy','zs','-v7.3');
else
    load(fullfile([expDir,'_data'],'train.mat'));
    IsT = cell(1);
    for i = 1:length(Is)
        IsT{i} = permute(Is{i},[2,1,3]);
    end
end

%% Optimization process
Is = Is(:);
IsT = IsT(:);
szx = szx(:);
szy = szy(:);
camParams = camParams(:);
zs = zs(:);

% initialize the variables that records the process of optimization
val_history = [];
var_history = [];
z_ests_history = [];
learn_rate_history = [];
DenDvar_history = [];
curTime = char(datetime);
half_fx_length = 5;
netParam.er = 10;

for optIdx = 1:50
    if strcmp(randFlag,'rand')
        % randomly generate the initializations
        % we restrict the left part of the fx to be sum to positive and Sigma
        % to be positive to guarantee convergence
        fx_half = (rand(1,half_fx_length) - 0.5);
        while sum(fx_half) < 0
            fx_half = (rand(1,half_fx_length) - 0.5);
        end
        netParam.diffFilter.fx = [fx_half, 0, -fx_half(end:-1:1)];
        netParam.diffFilter.fy = netParam.diffFilter.fx';
    elseif strcmp(randFlag,'fixed')
        % manually set the initializations
        netParam.diffFilter.fx = [0,0,0,0,0.5,0,-0.5,0,0,0,0];
        netParam.diffFilter.fy = netParam.diffFilter.fx';
    end

    %% Mini-batch Stochastic Gradient Descent
    % netParam.eqSigma = fitGauss(psf_func,netParam.camParam,1);
    
    % We make the optimization to be a two step thing, the first step is to
    % use the energy_half function, as this function can converge within a
    % large range
    energy_func = @energy_half_exp;
    energy_func_Dz_est = str2func([func2str(energy_func),'_Dz_est']);
    stepIdx = 1;
    val_old = inf;
    fprintf('Starting step %d.\n',stepIdx);
    
    % Generate the random delta_t
    % CAUTION: the delta_t must be consistent if the us are shared!!!!
    delta_ts = (rand(length(zs),1) - 0.5) * 20;
    delta_ts = delta_ts + sign(delta_ts);
    % NOTE: we only compute the energy of Is, ignoring IsT, as those two
    % energies are the same
    [val_new, z_ests, us] = energy_func(Is, zs, delta_ts, szx, szy, netParam, camParams);
    fprintf('Initial energy = %f\n',val_new);
    %% Draw the predicted depths vs. ground truth depths
    figure;
    for imgIdx = 1:length(zs)/length(textNum):length(zs)
        plot(zs(imgIdx:imgIdx + length(zs)/length(textNum) - 1),...
            z_ests(imgIdx:imgIdx + length(zs)/length(textNum) - 1),'.');
        hold on;
        plot(zs(imgIdx:imgIdx + length(zs)/length(textNum) - 1),...
            zs(imgIdx:imgIdx + length(zs)/length(textNum) - 1));
    end
    title(camName);
    hold off;
    
    % initialize the variables that records the process of optimization
    val_history = [val_history, val_new];
    var_history = [var_history, [netParam.diffFilter.fx(:);netParam.eqSigma(1,1)]];
    z_ests_history = [z_ests_history, z_ests];
    learn_rate_history = [learn_rate_history,nan];
    DenDvar_history = [DenDvar_history,nan(size(var_history,1),1)];
    
    stop_cri_val = 1e-6;
    stop_cri_der = 1e-1;
    batch_size = max(floor(length(zs) * 0.01),30);
    validation_flag1 = 0;
    validation_flag2 = 0;
    epoch_num = 1;
    DenDvar = 1;
    learn_rate = 0.1;
    learn_rate_thre = 1e-4;
    while sum(abs(DenDvar)) > stop_cri_der && abs(val_old - val_new) > stop_cri_val
        % add elements to the minibatch
        if strcmp(func2str(energy_func),'energy_robust')
            diff = abs(z_ests - zs);
            idxWR = find(diff < netParam.er * 2); % this 2 is obtained by looking at the function on Wolfram Alpha
            if length(idxWR) > batch_size
                idx = idxWR(randperm(length(idxWR),batch_size));
            else
                idx = idxWR;
            end
        else
            idx = randperm(length(zs),batch_size);
        end
        
        %% Compute the derivative
        u = us(idx,:);
        z_est = z_ests(idx);
        delta_t = delta_ts(idx); % WARNING: the random velocity must be consistent if reusing the u
        
        % compute the derivative of the minibatch
        Dz_estDfx = zeros(length(idx),size(netParam.diffFilter.fx,1),size(netParam.diffFilter.fx,2));
        Dz_estDfxT = zeros(length(idx),size(netParam.diffFilter.fx,1),size(netParam.diffFilter.fx,2));
        DzDSigma = zeros(length(idx),1);
        
        
        Is_temp = Is(idx);
        IsT_temp = IsT(idx);
        szx_temp = szx(idx);
        szy_temp = szy(idx);
        camParams_temp = camParams(idx);
        zs_temp = zs(idx);
        parfor i = 1:length(idx)
            netParamTemp = netParam;
            % First do it with original direction
            netParamTemp.diffFilter.tunits = delta_t(i);
            netParamTemp.w = ones(szy_temp(i) - 2 * (size(netParamTemp.diffFilter.fx,2) - 1)...
                ,szx_temp(i) - 2 * (size(netParamTemp.diffFilter.fx,2) - 1));
            
            Dz_estDfx(i,:,:) = focalFlowNet1_dfx_exp(Is_temp{i},netParamTemp,camParams_temp{i},validation_flag1 ,u(i,:)');
            DzDSigma(i) = focalFlowNet1_dSigma_exp(Is_temp{i},netParamTemp,camParams_temp{i},validation_flag1, u(i,:)');
            
            % Transpose the images and do it again
            netParamTemp.diffFilter.tunits = delta_t(i);
            netParamTemp.w = ones(szy_temp(i) - 2 * (length(netParamTemp.diffFilter.fy) - 1)...
                ,szx_temp(i) - 2 * (length(netParamTemp.diffFilter.fx) - 1));
            
            % NOTE: as the last two elements of u does not change after transpose, we use the original u here for convenience
            Dz_estDfxT(i,:,:) = focalFlowNet1_dfx_exp(IsT_temp{i}, netParamTemp,camParams_temp{i},validation_flag1, u(i,:)');
        end
        
        % We should add up four directions as the image can be placed in four
        % directions
        Dz_estDfx = (Dz_estDfx - Dz_estDfx(:,:,end:-1:1) + Dz_estDfx(:,end:-1:1,:) ...
            - Dz_estDfx(:,end:-1:1,end:-1:1) ...
            + Dz_estDfxT - Dz_estDfxT(:,:,end:-1:1) + Dz_estDfxT(:,end:-1:1,:) ...
            - Dz_estDfxT(:,end:-1:1,end:-1:1)) / 8;
        % NOTE: we only compute the derivative of energy of Is, ignoring IsT, as those two
        % energies are the same, just duplicate it
        DenDz_est = energy_func_Dz_est(Is_temp, zs_temp, delta_t, szx_temp, szy_temp, netParam, camParams_temp, z_est);
        
        % The derivative of energy w.r.t. fx and Sigma
        DenDfx = reshape(sum(repmat(DenDz_est, 1, size(Dz_estDfx,2), size(Dz_estDfx,3)) ...
            .* Dz_estDfx,1),size(netParam.diffFilter.fx,1),size(netParam.diffFilter.fx,2));
        
        DenDSigma = DenDz_est' * DzDSigma;
        
        % We think the variables to be arranged as f11; f21; ..., f12; ... fmn;
        % Sigma
        DenDvar = [DenDfx(:);DenDSigma];
        
        
        for valIdx = 1:validation_flag2
            %% Validation of derivatives
            % Validate DenDz_est
            DenDz_est_val = zeros(size(DenDz_est));
            delta = 0.0000001;
            for smpIdx = 1:length(DenDz_est_val)
                z_estTemp1 = z_est;
                z_estTemp2 = z_est;
                z_estTemp1(smpIdx) = z_estTemp1(smpIdx) - delta;
                z_estTemp2(smpIdx) = z_estTemp2(smpIdx) + delta;
                val_temp1 = energy_func(Is_temp, zs_temp, delta_t, szx_temp, szy_temp, netParam, camParams_temp, z_estTemp1);
                val_temp2 = energy_func(Is_temp, zs_temp, delta_t, szx_temp, szy_temp, netParam, camParams_temp, z_estTemp2);
                DenDz_est_val(smpIdx) = (val_temp2 - val_temp1) / (2 * delta);
            end
            diff_DenDz_est = abs((DenDz_est - DenDz_est_val) ./ DenDz_est);
            % VALIDATION OF DenDz_est PASSED
            
            
            % Validate DenDfx
            colIdx = 1;
            rowIdx = 1;
            delta = 0.0000001;
            DenDfx_val = zeros(size(DenDfx));
            for colIdx = 1:size(netParam.diffFilter.fx, 1)
                for rowIdx =  1:size(netParam.diffFilter.fx, 2)
                    netParam_temp1 = netParam;
                    netParam_temp2 = netParam;
                    netParam_temp1.diffFilter.fx(colIdx,rowIdx) = netParam_temp1.diffFilter.fx(colIdx,rowIdx) - delta;
                    netParam_temp2.diffFilter.fx(colIdx,rowIdx) = netParam_temp2.diffFilter.fx(colIdx,rowIdx) + delta;
                    val_temp1 = energy_func(Is_temp, zs_temp, delta_t, szx_temp, szy_temp, netParam_temp1, camParams_temp);
                    val_temp2 = energy_func(Is_temp, zs_temp, delta_t, szx_temp, szy_temp, netParam_temp2, camParams_temp);
                    
                    val_temp3 = energy_func(IsT_temp, zs_temp, delta_t, szx_temp, szy_temp, netParam_temp1, camParams_temp);
                    val_temp4 = energy_func(IsT_temp, zs_temp, delta_t, szx_temp, szy_temp, netParam_temp2, camParams_temp);
                    
                    DenDfx_val(colIdx,rowIdx) = (val_temp2 - val_temp1 + val_temp4 - val_temp3) / (4 * delta);
                end
            end
            DenDfx_val = (DenDfx_val - DenDfx_val(:,end:-1:1) + DenDfx_val(end:-1:1,:)...
                - DenDfx_val(end:-1:1,end:-1:1))/4;
            diff_DenDfx = abs((DenDfx - DenDfx_val) ./ DenDfx);
            % VALIDATION OF DenDfx PASSED
            
            % Validate DenDSigma
            delta = 0.0000001;
            netParam_temp1 = netParam;
            netParam_temp2 = netParam;
            netParam_temp1.camParam.eqSigma = netParam_temp1.camParam.eqSigma - delta * eye(2);
            netParam_temp2.camParam.eqSigma = netParam_temp2.camParam.eqSigma + delta * eye(2);
            val_temp1 = energy_func(Is_temp, zs_temp, delta_t, szx_temp, szy_temp, netParam_temp1, camParams_temp);
            val_temp2 = energy_func(Is_temp, zs_temp, delta_t, szx_temp, szy_temp, netParam_temp2, camParams_temp);
            val_temp3 = energy_func(IsT_temp, zs_temp, delta_t, szx_temp, szy_temp, netParam_temp1, camParams_temp);
            val_temp4 = energy_func(IsT_temp, zs_temp, delta_t, szx_temp, szy_temp, netParam_temp2, camParams_temp);
            DenDSigma_val = (val_temp2 - val_temp1 + val_temp4 - val_temp3) / (4 * delta);
            
            diff_DenDSigma = abs((DenDSigma - DenDSigma_val) ./ DenDSigma);
            % VALIDATION OF DenDSigma PASSED
            
            thre = 1e-4;
            if max(diff_DenDSigma(:)) > thre || max(diff_DenDfx(:)) > thre
                disp('Oops! The derivative is not accurate enough :(');
            end
        end
        
        % Generate new random delta_t
        delta_ts = (rand(length(zs),1) - 0.5) * 20;
        delta_ts = delta_ts + sign(delta_ts);
        
        %% Update the variables
        val_temp = val_new;
        learn_rate = learn_rate * 2;
        while val_new >= val_temp && learn_rate > learn_rate_thre
            learn_rate = learn_rate / 2;
            var = [netParam.diffFilter.fx(:);netParam.eqSigma(1,1)];
            var = var - learn_rate * DenDvar;
            netParam_temp = netParam;
            netParam_temp.diffFilter.fx = reshape(var(1:end-1),size(netParam.diffFilter.fx,1),...
                size(netParam.diffFilter.fx,2));
            netParam_temp.diffFilter.fy = netParam_temp.diffFilter.fx';
            netParam_temp.camParam.eqSigma = var(end) * eye(2);
            %% Update the energy
            [val_new, z_ests, us] = energy_func(Is, zs, delta_ts, szx, szy, netParam_temp, camParams);
        end

        %% Update the learning rate
        if learn_rate < learn_rate_thre
            % if the algorthm cannot find a better solution, then redo the
            % optimization again from the starting point
            if stepIdx == 1
                energy_func = @energy_robust_exp;
                energy_func_Dz_est = str2func([func2str(energy_func),'_Dz_est']);
                stepIdx = 2;
                fprintf('Starting step %d.\n',stepIdx);
                learn_rate = 0.1;
                learn_rate_thre = 1e-7;
                val_new = energy_func(Is, zs, delta_ts, szx, szy, netParam, camParams);
                
                fprintf('Initial energy = %f\n',val_new);
                val_history = [val_history,val_new];
                var_history = [var_history,var];
                z_ests_history = [z_ests_history, z_ests];
                learn_rate_history = [learn_rate_history,learn_rate];
                DenDvar_history = [DenDvar_history,DenDvar];
            elseif stepIdx == 2
                break;
            end
        else
            % if the energy decreases we should output and save the update
            fprintf('Epoch %d: energy = %f, learning rate = %f\n', epoch_num, val_new, learn_rate);
            epoch_num = epoch_num + 1;
            
            %% Draw the predicted depths vs. ground truth depths
            for imgIdx = 1:length(zs)/length(textNum):length(zs)
                plot(zs(imgIdx:imgIdx + length(zs)/length(textNum) - 1),...
                    z_ests(imgIdx:imgIdx + length(zs)/length(textNum) - 1),'.');
                hold on;
                plot(zs(imgIdx:imgIdx + length(zs)/length(textNum) - 1),...
                    zs(imgIdx:imgIdx + length(zs)/length(textNum) - 1));
            end
            title(camName);
            hold off;
            
            netParam = netParam_temp;
            val_old = val_temp;
            val_history = [val_history,val_new];
            var_history = [var_history,var];
            z_ests_history = [z_ests_history, z_ests];
            learn_rate_history = [learn_rate_history,learn_rate];
            DenDvar_history = [DenDvar_history,DenDvar];
            % if the energy decreases, we increase the learning rate for
            % faster convergence
            learn_rate = learn_rate * 2;
        end
        
        
        
        % save the file each time it finishes an optimization
        if ~exist(fullfile('opt',func2str(psf_func),camName))
            mkdir(fullfile('opt',func2str(psf_func),camName));
        end
        save(fullfile('opt',func2str(psf_func),camName,[curTime,'.mat']), ...
            'val_history','var_history','netParam','z_ests_history','textNum','zs','learn_rate_history','DenDvar_history','noiseSigma');
    end
end
