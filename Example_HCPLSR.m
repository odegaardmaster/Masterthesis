%% Load data

X_path = "../_data/HH_paramter_space_40s.csv";
y_path = "../_data/HH_voltage_40s.csv";


X = csvread(X_path,1,1);
y = csvread(y_path,1,2);

%% Train and test split
yold = y;
y = y(:,1:1200); % Remove 0 variance column

p = .7;      % proportion of rows to select for training
N = size(X,1);  % total number of rows 
tf = false(N,1);    % create logical index vector
tf(1:round(p*N)) = true;     
tf = tf(randperm(N));   % randomise order
X_train = X(tf,:); 
X_test = X(~tf,:);

y_train = y(tf,:);
y_test = y(~tf,:);

%% Choosing setup

options.PCs = 20;
options.autoscaleX=1;
options.autoscaleY=1;
options.CV=1;
options.plot=0; 
options.nrclusters=10;
options.matrix='SX';
options.classification='Fuzzy';
options.residuals=0;
options.reduced_clustering_rank=1;
options.secondorder=0;
options.sin_cos=0;

%% Calibration and prediction


hcplsr_cal = HPLS(y_train,X_train,options);
hcplsr_pred = HPLSpred(y_test,X_test,h);


%% Saving Variables for Sensitivity analysis.


global_train_model.R2 = hcplsr_cal.R2;
global_train_model.B = hcplsr_cal.B;
global_train_model.options = hcplsr_cal.options;
global_train_model.tf = tf;
global_train_model.MSE = hcplsr_cal.MSE;
global_train_model.RMSEP = hcplsr_cal.RMSEP;
global_train_model.pctvar = hcplsr_cal.pctvar;
global_train_model.Ypred = hcplsr_cal.Ypred;
global_train_model.optPCs = hcplsr_cal.optPCs;

local_train_model.cluster_centers = hcplsr_cal.cluster_centers;
local_train_model.Bgr = hcplsr_cal.Bgr;
local_train_model.options = hcplsr_cal.options;
local_train_model.tf = tf;
local_train_model.clusters = hcplsr_cal.clusters;
local_train_model.optPCsgr = hcplsr_cal.optPCsgr;
local_train_model.pctvargr = hcplsr_cal.pctvargr;
local_train_model.Ypredgr = hcplsr_cal.Ypredgr;
local_train_model.MSEgr = hcplsr_cal.MSEgr;
local_train_model.RMSEPgr = hcplsr_cal.RMSEPgr;



train_model_ls.LX = hcplsr_cal.LX;
train_model_ls.SX = hcplsr_cal.SX;
train_model_ls.LY = hcplsr_cal.LY;
train_model_ls.SY = hcplsr_cal.SY;
train_model_ls.W = hcplsr_cal.W;

train_model_ls.LXgr = hcplsr_cal.LXgr;
train_model_ls.SXgr = hcplsr_cal.SXgr;
train_model_ls.LYgr = hcplsr_cal.LYgr;
train_model_ls.SYgr = hcplsr_cal.SYgr;
train_model_ls.Wgr = hcplsr_cal.Wgr;
