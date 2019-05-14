%Copyright (c) 2019, Kristin Tøndel, kristin.tondel@gmail.com and Lars Erik Ødegaard, lars.erik.odegaard@gmail.com
%All rights reserved.
%
%Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
%
%1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
%
%2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the 
%documentation and/or other materials provided with the distribution.
%
%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
%THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS 
%BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE 
%GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
%STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
%OF SUCH DAMAGE.
%
%Publication to cite:
%Tøndel, K., Indahl, U.G., Gjuvsland, A.B., Vik, J.O., Hunter, P.,
%Omholt, S.W., Martens, H. Hierarchical Cluster-based Partial Least Squares 
%Regression is an efficient tool for metamodelling of nonlinear dynamic 
%models, BMC Syst. Biol., 2011, 5, 90.
%
%
% Hierarchical PLS regression Version 2.0
% Note: Changes from the first version:
%           - The input data is now standardized for each  cluster.
%           - The output data is now standardized for each cluster.
%           (The "restandarization" leads to a B0 of the local PLSR models
%           being neglectible.)
%
% Note: Must be used together with HPLSpred Version 2.0.
%
%
%I/O: hmodel=HPLS(X,Y,options,timesteps)
%The input timesteps is used for plotting, and is optional (useful only for
%timeseries data
%Default: 
% options.PCs=10
% options.autoscaleX=1
% options.autoscaleY=1
% options.CV=1 %Cross-validation
% options.plot=1
% options.nrclusters=6
% options.matrix='SX'; (or'SY' or 'T_', or 'Tp')
% options.classification='Fuzzy'; (or 'LDA' or 'QDA' or 'NB_' (NaiveBayes))
% options.residuals=0; (or 1) %Use local modelling only on residuals
% options.reduced_clustering_rank=1 (or 0) %Not use optimal rank in
% clustering (if =1 then only the first three PCs are used)
%options.secondorder=1; (or 0) % Add second order terms in regressor matrix
%options.sin_cos=0; (or 1) %Add sin(X) and cos(X) in regressor matrix

function hmodel=HPLS(X,Y,options,timesteps,hmodel_inverse,ind_CV)

if nargin<4
    timesteps=1:size(Y,2);
end
if nargin<3
    options.PCs=min(20,size(X,2));
    options.autoscaleX=1;
    options.autoscaleY=1;
    options.CV=0;
    options.plot=0; 
    options.nrclusters=6;
    options.matrix='SX';
    options.classification='Fuzzy';
    options.residuals=0;
    options.reduced_clustering_rank=1;
    options.secondorder=1;
    options.sin_cos=0;
end

Xold=X;
mX = mean(X); nX = size(X,1);
%
%Add second order terms, interactions and sinX and cosX
%
if options.secondorder==1
    %Mean-center X
    X = X - repmat(mX,nX,1);
    
    X=x2fx(X,'quadratic');
    if mean(X(:,1),1)==1 %Delete first column if only 1's
        X=X(:,2:end);
    end
end
if options.sin_cos==1
    X=[X sin(X) cos(X)];   
end
if nargin<3
    options.PCs=min(20,size(X,2));
end

%Delete samples with little variation in Y
delsamps=zeros(1,size(Y,1));
 for i=1:size(Y,1)
     if var(Y(i,:))<=0.1*abs(mean(Y(i,:)))
         delsamps(i)=i;
     end
 end

mX2 = mean(X); stdX = ones(size(mX2));

%Scale X-data
if options.autoscaleX==1
    stdX = std(X);
    Xuse = (X - repmat(mX2,nX,1))./repmat(stdX,nX,1);
    
else
    Xuse=X;   
end

%Scale Y-data

if options.autoscaleY==1
    nY = size(Y,1);
    mY = mean(Y); stdY = std(Y);
    Yuse = (Y - repmat(mY,nY,1))./repmat(stdY,nY,1);
    %[Yuse,my,stdy,msg]=auto(Y);
else
    Yuse=Y;
    mY = mean(Y); stdY = ones(size(mY));
end
%Xuse=Xuse./repmat(1:999,nX,1); %NB!!Fjern
%----------------------------------------------------------------------
%
%Make global model
%
%----------------------------------------------------------------------
options_glob=options;
options_glob.plot=0;
[B,Ypred,LX,LY,SX,SY,W,optPCs,MSE,pctvar]=ModelYfromX(Xuse,Yuse,options_glob,timesteps);
optPCs_old=optPCs;
Yresid=(Yuse-Ypred);
if options.residuals==1
    Yglob=Yuse;
    Yuse=Yresid;
end
if options.reduced_clustering_rank==1
    optPCs=3;
end
if nargin>4
    SYcross=x2fx(SY,'quadratic');
    % Model: B=A*C+eps
    % CHat=pinv(A)*B
    %THaT=A*CHat;

    C=pinv(SYcross)*hmodel_inverse.SX(ind_CV,1:options_glob.PCs);
    Tpheno_pred=SYcross*C;
end    
%----------------------------------------------------------------------
%
%Make hierarchical model
%
%----------------------------------------------------------------------
if nargin>4
    class=hmodel_inverse.clusters(ind_CV);
    smallrows=hmodel_inverse.smallrows;
    delclass=hmodel_inverse.delclass;
    rowsOK=1:length(ind_CV);
    fm=hmodel_inverse.cluster_centers;
    fm_old=hmodel_inverse.cluster_centers_old;
    U=hmodel_inverse.class_probabilities(ind_CV,:);
    U_old=hmodel_inverse.class_probabilities_old(ind_CV,:);
    class_old=hmodel_inverse.clusters_all(ind_CV);
    D=hmodel_inverse.distancematrix(ind_CV,:);
    D_old=hmodel_inverse.distancematrix_old(ind_CV,:);
    m=hmodel_inverse.fuzzifier;
else
    %Choose clustering basis
    if options.matrix=='SX'
        matr=SX(:,1:optPCs);
    elseif options.matrix=='SY'
        matr=SY(:,1:optPCs);
    elseif options.matrix=='T_'
        [COEFF,T, VAR] = princomp(X); %PCA on X
        PCs=1;
        for i=2:length(VAR)
        diff=abs(VAR(i)-VAR(i-1));
        if diff>0.01*VAR(1) && PCs<length(VAR)-1
        PCs=i;
        end
        end
        matr=T(:,1:PCs);
    end
    %
    %Separate objects in groups using cluster analysis
    %
    nrclusters=options.nrclusters;
    %Fuzzy clustering
    m=2;
    [U,J,fm,a,D] = fcm(matr,m,nrclusters);

    [Umax,class]=max(U,[],2);

    
    %Assure that all clusters have more than 10 members
    smallrows=[];
    emptyclass=[];
    for gr=1:options.nrclusters
        samps=find(class==gr);
        if length(samps)==0
            emptyclass=[emptyclass gr];
        elseif length(samps)<10
            smallrows=[smallrows; samps];
        end
    end

    D_old=D;
    fm_old=fm;
    class_old=class;
    U_old=U;
    if length(smallrows)==size(X,1)
        rowsOK=[];
        delclass=1:nrclusters;
        class=ones(size(X,1),1);
    elseif length(smallrows)~=0 || length(emptyclass)~=0
        allrows = 1:length(class);
        rowsOK = setdiff(allrows,smallrows);
        delclass=unique(class(smallrows))';
        delclass=[delclass emptyclass];
        fm=fm(:,setdiff(1:size(fm,2),delclass)); %Delete small clusters
        %Update distance matrix with new cluster centers

        D = distancematrix(matr,fm);
        %Update U
        U = oppdater(D,m);
        [Umax,class]=max(U,[],2);

        %Set smallrows to cluster nr. 0
        class(smallrows)=ones(length(smallrows),1)*(max(class)+1);
        [dclass class s] = dummy2(class); 
        class(smallrows)=zeros(length(smallrows),1);
    else
        rowsOK=1:size(X,1);
        delclass=[];
        [dclass class s] = dummy2(class); 
    end
end


nrclusters=max(class);
if nrclusters<2
    disp('Warning: Only one cluster (or only small clusters) found! Only global PLSR model is made.')
end
%Sjekk for outliers vha. distansematrisa D2
class_OK=class(rowsOK);
distances_OK=D(rowsOK,:);
outlier_limits=[];
for cla=1:nrclusters
    scl=find(class_OK==cla);
    min_dist=distances_OK(scl,cla); 
    outlier_limits=[outlier_limits max(min_dist)];
end 
if isempty(rowsOK)==0
    
    if options.classification(1:3)=='NB_'
        NB = NaiveBayes.fit(matr(rowsOK,:),class(rowsOK));
    end
else
    
   if options.classification(1:3)=='NB_'
        NB = NaiveBayes.fit(matr,class);
   end
end
%---------------------------------------------------------------------
%
%Make lokal model for each group
%
%---------------------------------------------------------------------

MYgr =zeros(nrclusters,size(mY,2));
stdYgr = zeros(nrclusters,size(mY,2));
mX2gr = zeros(nrclusters,size(mX2,2));
stdXgr = zeros(nrclusters,size(mX2,2));
Bgr=zeros(nrclusters,size(B,1),size(B,2));
Ypredgr=zeros(size(Ypred));
LXgr=zeros(nrclusters,size(LX,1),size(LX,2));
LYgr=zeros(nrclusters,size(LY,1),size(LY,2));
SXgr=zeros(size(SX));
SYgr=zeros(size(SY));
Wgr=zeros(nrclusters,size(W,1),size(W,2));
optPCsgr=zeros(1,nrclusters);
MSEgr=zeros(nrclusters,2,options.PCs+1);
pctvargr=zeros(nrclusters,2,options.PCs+1);
options_gr=options;
options_gr.plot=0;
if nrclusters>1
    for group=1:nrclusters
        
        samples=find(class==group);
        Xgroupuse = X(samples,:);
        Ygroupuse=Y(samples,:);
        nXgr = size(Xgroupuse,1);
        
        % Restandarize input and output data for each group.
        mX2gr(group,:) = mean(Xgroupuse); stdXgr(group,:) = ones(size(mX2gr(group,:)));
        %Scale X-data
        if options.autoscaleX==1
            stdXgr(group,:) = std(Xgroupuse);
            Xgroup = (Xgroupuse - repmat(mX2gr(group,:),nXgr,1))./repmat(stdXgr(group,:),nXgr,1);
            
        else
            Xgroup =Xgroupuse;   
        end
        
        mYgr(group,:) = mean(Ygroupuse); stdYgr(group,:) = ones(size(mYgr(group,:)));
        if options.autoscaleY==1
            nY = size(Ygroupuse,1);
            Ygroup = (Ygroupuse - repmat(mYgr(group,:),nY,1))./repmat(stdYgr(group,:),nY,1);

        else
            Ygroup=Ygroupuse;
            mYgr(group,:) = mean(Ygroup); stdYgr(group,:) = ones(size(mYgr(group,:)));
        end
        
        

        %Separate cross-effects and maineffects
        if options.secondorder==1
            maineff=Xgroup(:,1:size(Xold,2));
            maineff=[ones(length(samples),1), maineff];
            crosseff=Xgroup(:,size(Xold,2)+1:end);
            %Model: Z=maineff*Bmain_cross+D
            Bmain_cross=pinv(maineff)*crosseff;
            D=crosseff-maineff*Bmain_cross;
            Xgroup=[maineff(:,2:end),D];
        end
        %if mean(var(Xgroup))>0.000001 && mean(var(Ygroup))>0.000001
        options_gr.PCs=min(length(samples)-1,options.PCs); %options_gr.PCs
        [Bgr(group,:,:),Ypredgr(samples,:),lx,ly,sx,sy,w,optPCsgr(group),mse,pctvarg]=ModelYfromX(Xgroup,Ygroup,options_gr,timesteps);

        LXgr(group,1:size(lx,1),1:size(lx,2))=lx;
        LYgr(group,1:size(ly,1),1:size(ly,2))=ly;
        SXgr(samples,1:size(sx,2))=sx;
        SYgr(samples,1:size(sy,2))=sy;
        Wgr(group,1:size(w,1),1:size(w,2))=w;
        MSEgr(group,:,1:size(mse,2))=mse;
        pctvargr(group,:,1:size(pctvarg,2))=pctvarg;

    end
    if options.residuals==1
        Yuse=Yglob;
        Ypredgr=Ypredgr+Ypred;
    else
        Ypredgr(smallrows,:)=Ypred(smallrows,:);
    end
    %Rescale predicted Y-values from local modelling
    if options.autoscaleY==1
        for group=1:nrclusters
            samples=find(class==group);
            nY = length(samples);
            Ypredgr(samples,:) = Ypredgr(samples,:).*repmat(stdYgr(group,:),nY,1)+repmat(mYgr(group,:),nY,1);
        end
    end
    
    %Calculate R2 and RMSEP for local modelling
    r2gr=zeros(1,size(Y,2));
    for i=1:size(Y,2)
        rgr=corrcoef(Ypredgr(:,i),Y(:,i)); 
        r2gr(i)=rgr(1,2)^2;    
    end
    R2gr=mean(r2gr);
    residgr=(Y-Ypredgr).^2;
    RMSEPgr=sqrt(mean(sum(residgr,2)));
else
    R2gr=[];
    residgr=[];
    RMSEPgr=[];  
end
%Rescale predicted Y-values for global model
if options.autoscaleY==1
    nY = size(Ypred,1);
    Ypred = Ypred.*repmat(stdY,nY,1)+repmat(mY,nY,1);
end

%Calculate R2 and RMSEP for global model
r2=zeros(1,size(Y,2));
for i=1:size(Y,2)
    r=corrcoef(Ypred(:,i),Y(:,i)); 
    r2(i)=r(1,2)^2;    
end
R2=mean(r2);
resid=(Y-Ypred).^2;
RMSEP=sqrt(mean(sum(resid,2)));
%------------------------------------------------------
%
%Save results
%
%------------------------------------------------------
if options.matrix=='T_'
    hmodel.PCs=PCs;
    hmodel.T=T;
end
hmodel.smallrows=smallrows; %Clusters not used in calibration due to too few samples
hmodel.delclass=delclass; %Classes corresponding to smallrows
hmodel.rowsOK=rowsOK; %Rows (sampls) used in calibration
hmodel.cluster_centers=fm; %Cluster centres from fuzzy clustering in X- or Y-scores
hmodel.cluster_centers_old=fm_old; %Cluster centres prior to deleting small classes
hmodel.outlier_limits=outlier_limits; %1.5 * largest distance from cluster center for calibration samples
hmodel.class_probabilities=U; %Membership values from fuzzy clustering
hmodel.class_probabilities_old=U_old; %Membership values from fuzzy clustering prior to deleting small classes
hmodel.clusters=class; %Class memberships from fuzzy clustering
hmodel.clusters_all=class_old; %Class memberships from fuzzy clustering prior to deleting small classes
hmodel.distancematrix=D; %Distance matrix from fuzzy clustering
hmodel.distancematrix_old=D_old;%Distance matrix from fuzzy clustering prior to deleting small classes
hmodel.nrclusters=nrclusters; %Final number of clusters after deleting small clusters
hmodel.options=options; %Your input options
hmodel.timesteps=timesteps; %For time series only, used for plotting results
hmodel.fuzzifier=m; %Fuzzifier parameter in fuzzy clustering, usually 2 is used
hmodel.X=Xuse; %X used in regression
hmodel.Xold=Xold; %Input X
hmodel.Y=Y; %Input Y
hmodel.B=B; %Regression coefficients from global model at optimal rank
hmodel.Ypred=Ypred; %Global predictions at optimal rank
hmodel.Yresiduals=Yresid; %Y residuals (global at optimal rank)
hmodel.LX=LX; %X-loadings (global)
hmodel.LY=LY; %Y-loadings (global)
hmodel.SX=SX; %X-scores (global)
hmodel.SY=SY; %Y-scores (global)
hmodel.W=W; %PLS weights (global)
hmodel.optPCs_old=optPCs_old; %Chosen (optimal) rank of global model
hmodel.optPCs=optPCs; %Number of PCs used in clustering (if less than optimal rank)
hmodel.pctvar=pctvar; %Global cross-validated percent explained variance in X and Y (X is row 1, Y is row 2)
hmodel.Bgr=Bgr; %Regression coefficients from local modelling (first mode is cluster number)
hmodel.Ypredgr=Ypredgr; %Locally predicted Y
hmodel.LXgr=LXgr; %X-loadings (local), first mode is cluster number
hmodel.LYgr=LYgr; %Y-loadings (local), first mode is cluster number
hmodel.SXgr=SXgr; %X-scores (local), first mode is cluster number
hmodel.SYgr=SYgr; %Y-scores (local), first mode is cluster number
hmodel.Wgr=Wgr; %PLS weights (local), first mode is cluster number
hmodel.optPCsgr=optPCsgr; %Optimal ranks for the local models
hmodel.R2=R2; %Cross-validated R2 for global model
hmodel.R2gr=R2gr; %Cross-validated R2 for local model (NB! Not cross-model validated)
hmodel.RMSEP=RMSEP; %Cross-validated RMSEP for global model
hmodel.RMSEPgr=RMSEPgr; %Cross-validated RMSEP for local model (NB! Not cross-model validated)
hmodel.resid=resid; %Y-residuals (squared) for global model
hmodel.residgr=residgr; %Y-residuals (squared) for local modelling
hmodel.MSE=MSE; %Global cross-validated MSE in X and Y (X is row 1, Y is row 2)
hmodel.MSEgr=MSEgr; %Local cross-validated MSE in X and Y (X is row 1, Y is row 2)(NB! Not cross-model validated) First mode is cluster number
hmodel.pctvargr=pctvargr;%Local cross-validated percent explained variance in X and Y (X is row 1, Y is row 2)(NB! Not cross-model validated) First mode is cluster number
hmodel.delsamps=unique(delsamps); %dummy
hmodel.mX=mX;%Xmean prior to adding second order terms
hmodel.mX_2=mX2;%Xmean after adding second order terms
hmodel.stdX=stdX;%Weights used for X
hmodel.mY=mY;%Ymean
hmodel.stdY=stdY;%Weights used for Y
hmodel.mX2gr = mX2gr;
hmodel.stdXgr = stdXgr;
hmodel.mYgr = mYgr;
hmodel.stdYgr = stdYgr;


if options.classification(1:3)=='NB_'%Naive Bayes parameters if classification option is Naive Bayes
    hmodel.NB=NB;
end
if nargin>4
    hmodel.C=C; %Regression coefficients for conversion between SY og Tpheno
    hmodel.Tpheno_pred=Tpheno_pred;
end
%------------------------------------------------------
%
%Plot results
%
%------------------------------------------------------
if options.plot==1
    %
    %Plot predicted vs measured
    %
    
    if size(Y,2)==1
        if nrclusters>1
            figure(1);
            plot(Y,Ypredgr,'.')
            ylabel('Ypred, cal')
            xlabel('Yref')
            hold on
            plot(Y,Y,'r')
            title('Predicted and reference Y, calibration, hierarchical model');
            hold off
        end
        figure(2)
        plot(Y,Ypred,'.')
        ylabel('Ypred, cal')
        xlabel('Yref')
        hold on 
        plot(Y,Y,'r')
        title('Predicted and reference Y, calibration, global model');
        hold off
    else
        if nrclusters>1
            figure(1),plot(timesteps(1,:),Ypredgr(1:10,:),'r')
            legend('Ypred, cal');
            hold on
            figure(1),plot(timesteps(1,:),Y(1:10,:),'b')
            title('Predicted and reference Y, calibration, hierarchical model');
            %legend('Yref');
            hold off
        end
        figure(2),plot(timesteps(1,:),Ypred(1:10,:),'y')
        title('Predicted and reference Y, calibration, global model');
        legend('Ypred, cal');
        hold on
        figure(2),plot(timesteps(1,:),Y(1:10,:),'b')
        %legend('Yref');
        hold off
    end    
end


    
%__________________________________________________________________________________________________

function [B,Yhat,LX,LY,SX,SY,W,optPCs,MSE,pctvar]=ModelYfromX(X,Y,options,timesteps)


%----------------------------------------------------------------------
%
%Perform multivariate regression
%
%----------------------------------------------------------------------

if options.CV==1
    %Cross validation
    nsplits=min(size(X,1),10);
    [LX,LY,SX,SY,beta,pctvar,MSE,stats] = plsregress(X,Y,2,'cv',nsplits);  
else
    [LX,LY,SX,SY,beta,pctvar,MSE,stats] = plsregress(X,Y,options.PCs);  
end
optPCs=opt_PCs_MSE(MSE);
W=stats.W;
B = W(:,1:optPCs)*LY(:,1:optPCs)';
B = [mean(Y,1) - (mean(X,1)*B); B];
%Use optimal nr of PCs in prediction 
Yhat=[ones(size(X,1),1) X]*B;


%------------------------------------------------------------------
%Plot predicted vs measured and explained Y-variance
%------------------------------------------------------------------
if options.plot==1
    
    figure(3);
    figure(3),plot(timesteps(1,:),Y(1:5,:),'b')
    hold on
    figure(3),plot(timesteps(1,:),Yhat(1:5,:),'r')
    title('Predicted and measured Y');
    hold off
    
    figure(4);
    figure(4),plot(cumsum(pctvar(2,:)))
    title('Explained Y-variance');

        
end   

 %_____________________________________________________________________________
 function lv=opt_PCs_MSE(MSE)
    [a,b]=size(MSE);
    lv=1;
    for i=2:b
        diff=abs(MSE(2,i)-MSE(2,i-1));
        if diff>0.01*abs(MSE(2,1)) & lv<b-1
            lv=i;
        end
    end
    if lv>1
     lv=lv-1;
    end
%__________________________________________________________________________________
function [dY Yshort s] = dummy2(Y)
% Rutinen konverterer en vektor Y
% av K (uordnede) klasselabler 
% til en N*K indikatormatrise 'dY'.
% 'Yshort' inneholder kompaktomkodede 
% klasselabler. 's' fungerer som 
% transformasjon mellom 'Yshort' og den
% opprinneige kodingen i Y, dvs 
% Y = s(Yshort);
s = sort(unique(Y));
m = length(s);
dY = [];
for i = 1:m
    dY = [dY Y==s(i)];
end
Yshort = dY*(1:m)';

%__________________________________________________________________________

    
function [U,J,fm,a,D2] = fcm(X,m,c,U,limit,iter)

%fcm(X,m,C,...) Fuzzy C-Means using Euclidian distance, can be used for crisp clustering
%
%SYNTAX
%[U,J,fm,a] = fcm(X,m,c,U,limit,iter)
%[U,J,fm] = fcm(X,m,c,U,limit)
%[U,J,fm] = fcm(X,m,c,U)
%[U,J,fm] = fcm(X,m,c)
% or just U = fcm(.)
%
%INPUT
%X: Data to be clustered (n*p, n = no. of samples, p= no. of variables)
%m: Fuzzifier parameter (m>= 1, m = 1 --> crisp clustering)
%c: Number of clusters
%U: optional, initial matrix of membership values
%limit: Optional, can be used to set the limit of convergence, default: limit = 1e-8
%iter: Optional, Maximum number of iterations, default: iter = 100
%
%OUTPUT
%U: Matrix with membership values (n*c)
%J: optional, value of criterion function for each iteration 
%	J = sum_i sum_j (u_ij^m*d_ij^2)
%
%Literature: J. Bezdek 1981: Pattern recognition with fuzzy objective function algorithms
%
%23.02.01 Ingunn Berget, MATFORSK

[n,p] = size(X);

switch nargin
case 3
   U = lag_u(n,c);
   limit = 1e-6;
   iter = 1000;
case 4
   limit = 1e-6;
   iter = 1000;
case 5
   iter = 1000;
otherwise
end

if isempty(U)
   U = lag_u(n,c);
end
if isempty(limit)
   limit = 1e-6;
end

%Initialisation
Jold = 0;
DIFF = 1;
D2 = zeros(n,c);
J = zeros(iter,1);
if m == 1
   U = round(U); 
end

a = 1; %counter

%LOOP FOR OPTIMISATION OF J
%UNTIL DIFF = abs(Jold -Jnew) < limit
while (DIFF > limit) & (a <= iter)
    if mod(a,20) ==1
        fprintf('a:%d, DIFF:%d \n', a, DIFF);
    end
    
   
   %CALCULATE FUZZY MEANS 
   %FM_j = sum_i (u_ij^m*x_i)/sum_i(u_ij^m)
   fm = fuzzymeans(U,X,m);
   
   %CALCULATE SQUARED DISTANCES d_ij^2 = (x_i - FM_j).^2
    D2 = distancematrix(X,fm);   
   
   %UPDATE U
   U = oppdater(D2,m);
   
   %EVALUATE CRITERION
   Jnew = sum(sum(U.^m.*D2));
   DIFF = abs(Jold - Jnew);
   if nargout > 1
      J(a) = Jnew;
   end
   Jold = Jnew;
   a = a + 1;
end

                
%ARRANGE CLUSTERS
%first column of U always has highest membership values on top
%Useful for repetetions with different initialisations
if a < iter
	I = fiks(U); 
   U = U(:,I);
   D2 = D2(:,I);
   fm = fm(:,I);
else
   warning(['THE FCM ALGORITHM DID NOT CONVERGE AFTER ',num2str(iter),' ITERATIONS'])
end
 
 %----------------------------------------------------------------------------------
 function I = fiks(U)

%I = fix(U) Gir en indeks vektor I som angir rekkef�lge p� gruppene
%brukes for at gruppe 1 alltid er f�rst, gir rekkef�lge i U og X
%	IBE 17.01.99

[tmp,i] = sortrows(U');
I = flipud(i);

%-----------------------------------------------------------------------------------
function U = lag_u(N,C)

%U = lag_u(N,C) Lager medlemskaps matrise etter kriteruier for fuzzy c-means
%
%UTGANGSVERDIER FOR U, U_ij = Random.
% NB! Merk at U_ij = 1/C for alle i, j er en _stabil l�sning_!  Derfor st�y.
% sikrer at at sum_j U_ij = 1, samtidig som 0 =< U_ij =< 1:
%20.01.99 Ingunn Berget, basert p� kode fra Bj�rn Helges program

U = rand (N,C);
for i = 1:N
  s = sum(U(i,:)');
  if s > 1				% Skaler ned
    U(i,:) = U(i,:) / s;
  elseif s < 1				% Skaler opp
    U(i,:) = 1 - (1 - U(i,:)) * (C - 1) / (C - s);
  end
end

%----------------------------------------------------------------------------------
function v = fuzzymeans(U,X,m)

%FM_j = sum_i (f(u_ij)*x_i)/sum_i(f(u_ij))

[N,C] = size(U);
[N,P] = size(X);

for c = 1:C
    uc = U(:,c).^m;
    M = uc(:,ones(1,P));
    nominator = sum(M.*X);
    d = sum(uc);
    denominator = d(ones(P,1),:);
    v(:,c) = nominator'./denominator;
end



%--------------------------------------------------------
function D2 = distancematrix(X,v)

[n,p] = size(X);
c = size(v,2);

for i = 1:c
    t = v(:,i)';
    FM = t(ones(n,1),:);
    if size(X,2) == 1 %only one variable
        D2(:,i) = (X-FM).^2;
    else %several variables
        D2(:,i) = sum((X' - FM').^2)';      %diag((X - FM)*(X-FM)'); %sum((X' - FM').^2)';
    end
end


%-------------------------------------------------------
function U = oppdater(D,m)

if m == 1	%CRISP CLUSTERING
   [tmp,I] = min(D');
   [N,C] = size(D);
   tmp = 1:N;
   U = zeros(N,C);
   U(sub2ind(size(D),tmp,I)) = 1;		% u_ij = 1 if j = min_k(D_ik)
else			%FUZZY CLUSTERING
   D = D.^(1/(m - 1));
   DE = D+ eps;
   U = 1./(diag(sum(1./DE'))*DE); %+eps to avoid divide by zero warning
end

%HVIS D_ij = 0 for noen i eller j blir u_ij = NAN,
%retter opp med � sette u_ij = 1
%i = find(isnan(U));
if sum(D(:) == 0)>0
    %keyboard
    C = size(U,2);
    [row,col] = find(D == 0);
    for i = 1:length(row)
        U(row(i),col(i)) = 1;
        U(row(i),setdiff(1:C,col(i))) = 0;
    end
%    U(i) = 1;
end


    
    
    
    
    


 
