%Copyright (c) 2009, Kristin Tøndel, kristin.tondel@gmail.com and Lars Erik Ødegaard, lars.erik.odegaard@gmail.com
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
% Hierarchical PLS regression (prediction model) Version 2.0
% Note: Changes from the first version:
%           - The input data is now standardized for each  cluster.
%           - The output data is now standardized for each cluster.
%           (The "restandarization" leads to a B0 of the local PLSR models
%           being neglectible.)
%
% Note: Must be used together with HPLS Version 2.0.
%
%
%I/O: results_newsamples=HPLSpred(Xnew,Ynew,hmodel)
%hmodel is a result from HPLS
%Ynew can be an empty matrix if no Y-values for the Xnew data are to be 
%used for validation.
%
%

function results_newsamples=HPLSpred(Xnew,Ynew,hmodel)
Xnew_old=Xnew;
nX = size(Xnew,1);
%
%Add second order terms, interactions and sinX and cosX
%
if hmodel.options.secondorder==1
    Xnew = Xnew - repmat(hmodel.mX,nX,1);
    Xnew=x2fx(Xnew,'quadratic');
    if mean(Xnew(:,1),1)==1
        Xnew=Xnew(:,2:end);
    end
end
if hmodel.options.sin_cos==1
    Xnew=[Xnew sin(Xnew) cos(Xnew)];
end
%Xnew=[Xnew Xnew(:,8)./Xnew(:,9)];

%Scale X-data
if hmodel.options.autoscaleX==1
    Xnewuse = (Xnew - repmat(hmodel.mX_2,nX,1))./repmat(hmodel.stdX,nX,1);
end

%Do predictions

if hmodel.options.matrix=='T_'
    [COEFF,Tnew, VAR] = princomp(Xnewuse); %PCA on Xnew
end
idnan=[];
Ypred=[];
Ypredgr=[];
Ypredgr_weighted=[];
Ypred_clust=zeros(hmodel.nrclusters,size(Xnewuse,1),size(hmodel.Ypred,2));

Ypred=[ones(size(Xnewuse,1),1) Xnewuse]*hmodel.B; %From global model

%Center Xnew and Ypred and calculate SXnew and SYnew based on W for global model
if hmodel.options.autoscaleX==0
    Xnew0=Xnewuse - repmat(hmodel.mX_2,size(Xnewuse,1),1);
else
    Xnew0=Xnewuse;
end
if hmodel.options.autoscaleY==0
    Ypred0=Ypred - repmat(hmodel.mY,size(Ypred,1),1);
else
    Ypred0=Ypred;
end

SXnew=Xnew0*hmodel.W;
SYnew=Ypred0*Ypred0'*SXnew;
%Orthogonalize the Y scores w.r.t. the preceding Xscores, i.e. XSCORES'*YSCORES will be lower triangular.
for i=1:size(SYnew,2)
    ui = SYnew(:,i);
        for repeat = 1:2
            for j = 1:i-1
                tj = SXnew(:,j);
                ui = ui - (ui'*tj)*tj;
            end
        end
    SYnew(:,i) = ui;
end

if hmodel.options.matrix=='Tp'
    SYnew_cross=x2fx(SYnew,'quadratic');
    Tpheno_new=SYnew_cross*hmodel.C;
end

if hmodel.options.matrix=='SX'
    matr=SXnew(:,1:hmodel.optPCs);
    compare=hmodel.SX(hmodel.rowsOK,1:hmodel.optPCs);
    compare2=hmodel.SX(hmodel.smallrows,1:hmodel.optPCs);
elseif hmodel.options.matrix=='SY'
    matr=SYnew(:,1:hmodel.optPCs);
    compare=hmodel.SY(hmodel.rowsOK,1:hmodel.optPCs);
    compare2=hmodel.SY(hmodel.smallrows,1:hmodel.optPCs);
elseif hmodel.options.matrix=='T_'
    matr=Tnew(:,1:hmodel.PCs);
    compare=hmodel.T(hmodel.rowsOK,1:hmodel.PCs);
    compare2=hmodel.T(hmodel.smallrows,1:hmodel.PCs);
elseif hmodel.options.matrix=='Tp'
    matr=Tpheno_new(:,1:hmodel.optPCs);
    compare=hmodel.Tpheno_pred(hmodel.rowsOK,1:hmodel.optPCs);
    compare2=hmodel.Tpheno_pred(hmodel.smallrows,1:hmodel.optPCs);       
end


%CALCULATE SQUARED DISTANCES d_ij^2 = (x_i - FM_j).^2

D2 = distancematrix(matr,hmodel.cluster_centers(1:hmodel.optPCs,:));   


if hmodel.options.classification(1:3)=='LDA'
    if isempty(hmodel.rowsOK)==0
        [class,err,pclass,logp,coeff] = classify(matr,compare,hmodel.clusters(hmodel.rowsOK),'linear');
    else
        [class,err,pclass,logp,coeff] = classify(matr,compare2,hmodel.clusters(hmodel.smallrows),'linear');
    end

elseif hmodel.options.classification(1:3)=='QDA'
    if isempty(hmodel.rowsOK)==0
        [class,err,pclass,logp,coeff] = classify(matr,compare,hmodel.clusters(hmodel.rowsOK),'quadratic');
    else
        [class,err,pclass,logp,coeff] = classify(matr,compare2,hmodel.clusters(hmodel.smallrows),'quadratic');
    end
elseif hmodel.options.classification(1:3)=='NB_'
    [pclass,class] = posterior(hmodel.NB,matr);
elseif hmodel.options.classification=='Fuzzy'
    %Calculate U for new objects
    U = oppdater(D2,hmodel.fuzzifier);
    [Umax,class]=max(U,[],2);
    pclass=U;
end
if hmodel.nrclusters>1
    
    %Predict Y for new samples using B for most probable cluster
    for k=1:hmodel.nrclusters
        samp=find(class==k);
        B=squeeze(hmodel.Bgr(k,:,:));
        if size(B,1)==1
             B=B';
        end
        Xsampl=Xnew(samp,:);
        
        %Restandardize X-data for each cluster
        nX = size(Xsampl,1);
        if hmodel.options.autoscaleX==1
            Xsamp = (Xsampl - repmat(hmodel.mX2gr(k,:),nX,1))./repmat(hmodel.stdXgr(k,:),nX,1);
        end
        
        if hmodel.options.secondorder==1
            maineff=Xsamp(:,1:size(Xnew_old,2));
            maineff=[ones(length(samp),1), maineff];
            crosseff=Xsamp(:,size(Xnew_old,2)+1:end);
            %Model: Z=maineff*Bmain_cross+D
            Bmain_cross=pinv(maineff)*crosseff;
            D=crosseff-maineff*Bmain_cross;
            Xsamp=[maineff(:,2:end),D];
        end

        Ypredgr(samp,:)=[ones(size(Xsamp,1),1) Xsamp]*B;
    end

    %Predict Y for new samples in all clusters
    if hmodel.options.secondorder==1
        maineff=Xnewuse(:,1:size(Xnew_old,2));
        maineff=[ones(size(Xnew_old,1),1), maineff];
        crosseff=Xnewuse(:,size(Xnew_old,2)+1:end);
        %Model: Z=maineff*Bmain_cross+D
        Bmain_cross=pinv(maineff)*crosseff;
        D=crosseff-maineff*Bmain_cross;
        Xnewuse=[maineff(:,2:end),D];
    end
    
    
    for i=1:hmodel.nrclusters
        Bcl=squeeze(hmodel.Bgr(i,:,:));
        if size(Bcl,1)==1
             Bcl=Bcl';
        end
        Ypred_clust(i,:,:)=[ones(size(Xnewuse,1),1) Xnewuse]*Bcl;
    end


    if hmodel.options.residuals==1
        Ypredgr=Ypred+Ypredgr;
        Ypredgr_weighted=Ypred+Ypredgr_weighted;
    end
else
    Ypredgr=Ypred;
    Ypredgr_weighted=Ypred;
end
%Rescale predicted Y-values
if hmodel.options.autoscaleY==1
    for k=1:hmodel.nrclusters
        samp=find(class==k);
        nY = size(Ypredgr(samp,:),1);
        Ypredgr(samp,:) = Ypredgr(samp,:).*repmat(hmodel.stdYgr(k,:),nY,1)+repmat(hmodel.mYgr(k,:),nY,1);

    end
    Ypred = Ypred.*repmat(hmodel.stdY,size(Xnewuse,1),1)+repmat(hmodel.mY,size(Xnewuse,1),1);
end
if hmodel.options.autoscaleX==1
    Xnewuse=Xnew_old;
end

if isempty(hmodel.outlier_limits)==0
    %Sjekk for outliers, bruk global modell hvis outliere
    outliers=[];
    for i=1:size(Xnewuse,1)
        mind=D2(i,class(i));
        if mind>1.5*hmodel.outlier_limits(class(i));
            outliers=[outliers; i];
        end
    end

else
    outliers=[];
end
%___________________________________________
%Save results
%___________________________________________
if hmodel.options.matrix=='T_'
    results_newsamples.Tnew=Tnew; %Predicted PCA-scores if using PCA in clustering 
end
results_newsamples.predgroups=class; %Predicted classes 
results_newsamples.Ypredgr=Ypredgr; %Final locally predicted Y (using closest model)
results_newsamples.Ypredgr_weighted=Ypredgr_weighted; %Final locally predicted Y (using weighted average of all local models)
results_newsamples.Ypred=Ypred; %Global prediction
results_newsamples.pclass=pclass; %Cluster membership probabilities 
results_newsamples.outliers=outliers; %Detected outliers based on outlier limits from hmodel
results_newsamples.SXnew=SXnew; %Predicted X-scores for new samples 
results_newsamples.SYnew=SYnew;%Predicted Y-scores for new samples 

if isempty(Ynew)==0
    results_newsamples.Ynew=Ynew;

    %Correlation

    rgr=zeros(1,size(Ynew,2));
    rglob=rgr;
    for i=1:size(Ynew,2)

        rgr_unw=corrcoef(Ypredgr(:,i),Ynew(:,i));
        rgl=corrcoef(Ypred(:,i),Ynew(:,i));
       

        if size(rgr_unw,2)>1
            rgr(i)=rgr_unw(1,2)^2;
        else
            rgr(i)=rgr_unw^2;
        end
        if size(rgl,2)>1
            rglob(i)=rgl(1,2)^2;
        else
            rglob(i)=rgl^2;
        end
    end

    Rgr=rgr;
    R=rglob;

    results_newsamples.R2gr=Rgr;%Test-set R2 from local predictions with closest local model
    results_newsamples.R2=R;%Test-set R2 from global prediction
    
    %RMSEP

    residgr=(Ynew-Ypredgr).^2;
    RMSEPgr=sqrt(mean(sum(residgr,2)));
    results_newsamples.RMSEPgr=RMSEPgr; %Test-set RMSEP from local predictions with closest local model
    
    resid=(Ynew-Ypred).^2;
    RMSEP=sqrt(mean(sum(resid,2)));
    results_newsamples.RMSEP=RMSEP;  %Test-set RMSEP from global prediction
    results_newsamples.idnan=idnan; %Missing data
    
    for k=1:hmodel.nrclusters
        samp=find(class==k);
        if isempty(samp)==0
            R2gr=[];  
            for i=1:size(hmodel.Y,2)
                r2gr=corrcoef(Ypredgr(samp,i),Ynew(samp,i));
                if numel(r2gr)>1
                    r2gr=r2gr(1,2)^2;
                end
                R2gr=[R2gr r2gr]; 
            end
            
       

            R2allgr(k,:)=R2gr;
        else
            R2allgr(k,:)=zeros(1,size(hmodel.Y,2));
        end
        
    end
    results_newsamples.R2allgr=R2allgr;
end
%___________________________________________
%Plot results
%___________________________________________
if hmodel.options.plot==1
   %
    %Plot predicted vs measured
    %
    
    if size(Ynew,2)==1
        if hmodel.nrclusters>1
            figure(5);
            plot(Ynew,Ypredgr,'.')
            ylabel('Ypred, val')
            xlabel('Yref')
            hold on
            plot(Ynew,Ynew,'r')
            title('Predicted and reference Y, testset validation, hierarchical model');
            hold off
        end
        figure(6)
        plot(Ynew,Ypred,'.')
        ylabel('Ypred, val')
        xlabel('Yref')
        hold on 
        plot(Ynew,Ynew,'r')
        title('Predicted and reference Y, testset validation, global model');
        hold off
    else
        if hmodel.nrclusters>1
            figure(5),plot(hmodel.timesteps(1,:),Ypredgr(1:10,:),'r')
            legend('Ypred, val');
            hold on
            figure(5),plot(hmodel.timesteps(1,:),Ynew(1:10,:),'b')
            title('Predicted and reference Y, testset validation, hierarchical model');
            hold off
        end
        figure(6),plot(hmodel.timesteps(1,:),Ypred(1:10,:),'y')
        legend('Ypred, val');
        title('Predicted and reference Y, testset validation, global model');
        hold on
        figure(6),plot(hmodel.timesteps(1,:),Ynew(1:10,:),'b')
        hold off
    end     
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

