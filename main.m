%{
  ---------------------------------------------------------------------------
    Name       : "main.m"
    
    Description: The parameter initialization, network initilization, Dataset
                 loading, invoking the training and testing, and plotting results are all
                 handled by main.m
  
    Authors: Kalkidan Deme Muleta, Bai-Sun Kong
  
    Paper: "RRAM-Based Spiking Neural Network with Target-Modulated
            Spike-Timing-Dependent Plasticity" published on IEEE Transactions on
            Biomedical Circuits and Systems.
%}

clear
close all

rng(2, 'twister');
%----------------------
%  parameters
H = 28;
W = 28;
M = H*W;
N = 100;

NumClasses = 10;
NumEpoch = 10;

Cx_vth = 0.0*ones(1,N);
Rl_vth = 0.485E6;
% VthTrain = Rl_vth * I_synapse_1T1R(Cx_vth);
VthTrain = ones(1,N);
VthTest = VthTrain;
meanW = 0.8;
stdW = 0.1;

% Display Network 
disp("=============================")
disp("parameters")
disp("-----------------------------")
fprintf("Network size:  %ix%i \n",M,N)
fprintf("Number of Epochs: %i \n", NumEpoch)
fprintf("Training Vth: %i \n", VthTrain(1))
% fprintf("Learning Rate LTP: %i \n", lr_p)
% fprintf("Learning Rate LTD: %i \n", lr_n)
% fprintf("Time Steps: %i \n", tSteps)

%----------------------
% weight initialization
% random initialization
weights   = normrnd(meanW,stdW,[M,N]); % Custom
weights(weights>0.9999) = 0.9999;
weights(weights<0.0001) = 0.0001;

% read init weights
% weights = readmatrix('weights.txt');
% load pretrained weights
% weights_All = load(strcat("Weights_all_proposed_1T1R_"+int2str(M)+"x"+int2str(N)+".mat")).weights_All;
% weights = weights_All(:,:,end);

weights_All = zeros(M,N,NumEpoch+1);
weights_All(:,:,1) = weights;

%----------------------
% %  Load MNIST
[PREtrain, TGTtrain, PREtest, TGTtest] =  GetMNIST(H, W, 0.25);
TGTtrain  = TGTtrain.';
TGTtest = TGTtest.';
%----------------------

Vmp_All = zeros(N,length(PREtrain),NumEpoch);
Vth_All = zeros(N,length(PREtrain),NumEpoch);
SpikeCorrOrNa = zeros(2,NumEpoch);
SpikeHistory = zeros(N,NumEpoch);

%----------------------
% target assignment
assignedTGT = repmat([1:NumClasses],1,N/NumClasses); % multicLAS targer assignment

%----------------------
%testing Prep
% WTA_history = zeros(length(PREtest),NumEpoch);
% overallMatrix = zeros(NumClasses,NumClasses+1,NumEpoch);

Target = TGTtest.';
Tgt = zeros(size(Target,1),1);
for i=1:size(Target,1)
    Tgt(i) = find(Target(i,:));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training and Testing every epoch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[weights, weights_All, WTA_history, overallMatrix, VthTrain, Cx_vth, Vmp_All, SpikeCorrOrNa, SpikeHistory,Vth_All] = ProposedTrainerTester_1T1R(PREtrain, TGTtrain, assignedTGT, PREtest, Tgt, NumEpoch, weights, VthTrain, Cx_vth, Rl_vth, Vmp_All, SpikeCorrOrNa, SpikeHistory, Vth_All, NumClasses);

%----------------------
% Save Trained weights
% save(strcat("Weights_all_proposed_1T1R_"+int2str(M)+"x"+int2str(N)+".mat"),'weights_All')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Plot Result
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Wmatrix = weights;
overallMatrix(:,:,end)
figure("Name", "Trained Weights")
clims = [0 1];
colormap(hot)
for i=1:min(100,N)
    %----- Plot Trained Weights -----------------------------------------------
    subplot(N/10,10,i)
    imagesc(reshape(Wmatrix(:,i),sqrt(M),sqrt(M)),clims)
    daspect([1 1 1])

    axis off
    title(string("Trained W"+ int2str(i) + "(Cf)"),'FontSize',12,'FontWeight','bold','FontName','Cambria')

end
figure("Name", "Trained Weights")
clims = [0 1];
colormap(hot)
t = tiledlayout(min(100,N)/10,10,'TileSpacing','none','Padding','none');
for i=1:min(100,N)
%----- Plot Trained Weights -----------------------------------------------
% subplot(NumPost,NumSteps,i)
nexttile
imagesc(reshape(Wmatrix(:,i),sqrt(M),sqrt(M)),clims)
daspect([1 1 1])
axis off
% title(string("Trained W"+ int2str(i) + "(Cf)"),'FontSize',12,'FontWeight','bold','FontName','Cambria')
end

axes1 = axes('Parent',figure("Name", "Testing Accuracy Progress"));
for ep=1:NumEpoch
    EpochCorrPred(1,ep)  = sum(diag(overallMatrix(:,1:end-1,ep)));
    EpochErrPred(1,ep)   = sum(overallMatrix(:,1:end-1,ep),'All') - sum(diag(overallMatrix(:,1:end-1,ep)));
    EpochAllPred(1,ep)   = sum(overallMatrix(:,1:end-1,ep),'All');
    EpochNoPred(1,ep)    = sum(overallMatrix(:,end,ep));

    EpochAllPredClass(:,ep) = sum(overallMatrix(:,1:end-1,ep),2);
    EpochErrPredClass(:,ep) = sum(overallMatrix(:,1:end-1,ep),2) - diag(overallMatrix(:,1:end-1,ep));
end 
EpochAccuracy   = 100*EpochCorrPred/length(PREtest);
EpochError      = 100*EpochErrPred/length(PREtest);


plot([1:NumEpoch],EpochAccuracy,'-', 'LineWidth',2, 'MarkerSize',6)
box on
xlabel('Epochs', FontSize=14,FontWeight='bold')
ylabel('Accuracy (%)', FontSize=14,FontWeight='bold')
title(" 784x100 Network Performance")
ylim([0,100])
set(axes1,'FontSize',14);

yyaxis right
plot([1:NumEpoch],EpochError,'-', 'LineWidth',2, 'MarkerSize',6)
ylabel('Error (%)', FontSize=14,FontWeight='bold')
ylim([0,100])

% %----- Scatter prediction hit ------------------
% figure()
% scatter([1:length(Pred)],Pred)
% hold on
% scatter([1:length(Pred)],Tgt,'*')
% legend('Predicted','Target')

% %----- Confusion Matrix overall with silent------------------
% figure()
% xvalues = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Silent'};
% yvalues = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
% % xvalues = {'5', '6', '7', '8', '9', 'Silent'};
% % yvalues = {'5', '6', '7', '8', '9'};
% TgtClassCount = [numel(find(Tgt==1)) numel(find(Tgt==2)) numel(find(Tgt==3)) numel(find(Tgt==4)) numel(find(Tgt==5))...
%                  numel(find(Tgt==6)) numel(find(Tgt==7)) numel(find(Tgt==8)) numel(find(Tgt==9)) numel(find(Tgt==10))].';
% h=heatmap(xvalues,yvalues,(overallMatrix(:,:,end)./TgtClassCount)); %,'CellLabelColor','none'
% colormap(flipud(hot))
% h.ColorLimits = [0 1];
% h.Title = 'Confusion Matrix';
% h.XLabel = 'Prediction';
% h.YLabel = 'Target';
% h.FontSize = 14;
% h.FontName = 'Cambria';


%----- Confusion Matrix when not silent------------------
figure()
xvalues = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
yvalues = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
h=heatmap(xvalues,yvalues,(overallMatrix(:,1:end-1,end)./EpochAllPredClass(:,end))); %,'CellLabelColor','none'
colormap(flipud(hot))
h.ColorLimits = [0 1];
% h.Title = 'Confusion Matrix(Active)';
h.XLabel = 'Prediction';
h.YLabel = 'Target';
h.FontSize = 14;
h.FontName = 'Cambria';


disp("============= Individual accuracy=========")
fprintf("-----------------------------------------------\n")
fprintf("\t\t\t Correct \t Silent \t Incorrect\n")
fprintf("------------ -------- \t -------- \t ---------\n")
fprintf("Class 0 :\t %s%% \t %s%% \t %s%% \n", num2str(100*overallMatrix(1,1,end)/numel(find(Tgt==1))),     num2str(100*overallMatrix(1,end,end)/numel(find(Tgt==1))),  num2str(100*sum(EpochErrPredClass(1,end))/numel(find(Tgt==1))))
fprintf("Class 1 :\t %s%% \t %s%% \t %s%% \n", num2str(100*overallMatrix(2,2,end)/numel(find(Tgt==2))),     num2str(100*overallMatrix(2,end,end)/numel(find(Tgt==2))),  num2str(100*sum(EpochErrPredClass(2,end))/numel(find(Tgt==2))))
fprintf("Class 2 :\t %s%% \t %s%% \t %s%% \n", num2str(100*overallMatrix(3,3,end)/numel(find(Tgt==3))),     num2str(100*overallMatrix(3,end,end)/numel(find(Tgt==3))),  num2str(100*sum(EpochErrPredClass(3,end))/numel(find(Tgt==3))))
fprintf("Class 3 :\t %s%% \t %s%% \t %s%% \n", num2str(100*overallMatrix(4,4,end)/numel(find(Tgt==4))),     num2str(100*overallMatrix(4,end,end)/numel(find(Tgt==4))),  num2str(100*sum(EpochErrPredClass(4,end))/numel(find(Tgt==4))))
fprintf("Class 4 :\t %s%% \t %s%% \t %s%% \n", num2str(100*overallMatrix(5,5,end)/numel(find(Tgt==5))),     num2str(100*overallMatrix(5,end,end)/numel(find(Tgt==5))),  num2str(100*sum(EpochErrPredClass(5,end))/numel(find(Tgt==5))))
fprintf("Class 5 :\t %s%% \t %s%% \t %s%% \n", num2str(100*overallMatrix(6,6,end)/numel(find(Tgt==6))),     num2str(100*overallMatrix(6,end,end)/numel(find(Tgt==6))),  num2str(100*sum(EpochErrPredClass(6,end))/numel(find(Tgt==6))))
fprintf("Class 6 :\t %s%% \t %s%% \t %s%% \n", num2str(100*overallMatrix(7,7,end)/numel(find(Tgt==7))),     num2str(100*overallMatrix(7,end,end)/numel(find(Tgt==7))),  num2str(100*sum(EpochErrPredClass(7,end))/numel(find(Tgt==7))))
fprintf("Class 7 :\t %s%% \t %s%% \t %s%% \n", num2str(100*overallMatrix(8,8,end)/numel(find(Tgt==8))),     num2str(100*overallMatrix(8,end,end)/numel(find(Tgt==8))),  num2str(100*sum(EpochErrPredClass(8,end))/numel(find(Tgt==8))))
fprintf("Class 8 :\t %s%% \t %s%% \t %s%% \n", num2str(100*overallMatrix(9,9,end)/numel(find(Tgt==9))),     num2str(100*overallMatrix(9,end,end)/numel(find(Tgt==9))),  num2str(100*sum(EpochErrPredClass(9,end))/numel(find(Tgt==9))))
fprintf("Class 9 :\t %s%% \t %s%% \t %s%% \n", num2str(100*overallMatrix(10,10,end)/numel(find(Tgt==10))),  num2str(100*overallMatrix(10,end,end)/numel(find(Tgt==10))),num2str(100*sum(EpochErrPredClass(10,end))/numel(find(Tgt==10))))
