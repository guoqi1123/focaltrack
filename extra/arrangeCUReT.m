% Arrange the CUReT dataset so that all textures are arranged from 0 to
% 5611
infolder = 'curetgrey';
outfolder = 'Qi_curetgrey';
samples = dir(infolder);
samples(1:2) = [];
num = 1;
idx = 0;
for i = 1:length(samples)
    images = dir(fullfile(infolder,samples(i).name,'*.png'));
    for j = 1:length(images)
        img = imread(fullfile(infolder,samples(i).name,images(j).name));
        imwrite(img, fullfile(outfolder,sprintf('%04d.png',idx)));
        idx = idx + 1;
    end
    num = num + length(images);
end
