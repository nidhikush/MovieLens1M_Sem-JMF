%function [U,V]=jointmf(Traindata,S,dim,alpha,lambda,epsilon,numiter)
% Traindata: The training data of user-item preference. Note that the
% entryies need to be normalized to [0,1].
% S: The item-to-item similarity matrix based on any external knowledge
% about items. Note that the size of S should be the same as the number of
% columns in Traindata 
% dim: Dimensions of latent features/factors
% alpha: the tradeoff parameter for controlling the contribution of S
% lambda: the regularization parameter for penalizing the magnitude of
% latent features
% epsilon: for controlling the convergence
% numiter: the maximal number of iterations
% U: latent user feature matrix, one row per user
% V: latent item feature matrix, one row per item
fprintf('shilpi\n');
Mg = load('MovieXtopic.csv');   % N * K
Traindata = load('UserXmovie.csv');

%[M,N]=size(Traindata);
dim = 48;
M = 6040; %user
N = 3091; %movie
K = 5003; %topic

%csvwrite('UserXmovie.csv', full(Traindata));
fprintf('shilpi1\n');

%disp(S);
fprintf('mansi\n');

fprintf('shilpi3\n');

%Sf = corrcoef(Mg');
%Sf(isnan(Sf)) = 0;
%disp(Traindata);
%csvwrite('MovieXmovie.csv', Sf);
Sf = load('MovieXmovie.csv');
fprintf('shilpi4\n');
t=0;
gamma=0.1;
alpha = 0.01;
lambda = 0.1;
U=0.1*rand(M,dim);
V=0.1*rand(N,dim);
%Vn = Vj;
%Vn=0.1*rand(K,dim);
numiter = 5;
fprintf('shilpi5\n');
E1 = 0.5*sum(sum((U*V'.*(Traindata>0)-Traindata).^2)) + 0.5*alpha*sum(sum((V*V'.*(Sf>0)-Sf).^2)) + 0.5*lambda*(sum(sum(U.^2))+sum(sum(V.^2)));
fprintf('shilpi6\n');
while t < numiter
    gamma=gamma*2;
    nextU=U-gamma*((U*V'.*(Traindata>0)-Traindata)*V+lambda*U);
    %nextVn = Vn;
    nextV=V-gamma*((U*V'.*(Traindata>0)-Traindata)'*U+2*alpha*(V*V'.*(Sf>0)-Sf)*V+lambda*V);
    E2=0.5*sum(sum((nextU*nextV'.*(Traindata>0)-Traindata).^2))+0.5*alpha*sum(sum((nextV*nextV'.*(Sf>0)-Sf).^2))+0.5*lambda*(sum(sum(nextU.^2))+sum(sum(nextV.^2)));
    while E2>=E1
        gamma=gamma/2;
        nextU=U-gamma*((U*V'.*(Traindata>0)-Traindata)*V+lambda*U);
        nextV=V-gamma*((U*V'.*(Traindata>0)-Traindata)'*U+2*alpha*(V*V'.*(Sf>0)-Sf)*V+lambda*V);
        E2=0.5*sum(sum((nextU*nextV'.*(Traindata>0)-Traindata).^2))+0.5*alpha*sum(sum((nextV*nextV'.*(Sf>0)-Sf).^2))+0.5*lambda*(sum(sum(nextU.^2))+sum(sum(nextV.^2)));
    end
    U=nextU;
    V=nextV;
    R=U*V';
    err1=0.0;
eer2=0.0;
err=0.0;
count=0;
      for i=1:M
      for j=1:N
            if (Traindata(i,j)~=0)
               err=Traindata(i,j)-R(i,j);
               err1=err+err*err;
               count=count+1;
            end
      end
      end
    err2=sqrt(err1/count);
    deltaE=(E1-E2)/E1;
    epsilon = 0.0001;
    if deltaE<=epsilon
        break;
    else
        E1=E2;
        t=t+1;
    end
    fprintf(1,'%s%d %s%8.6f %s%8.6f %s%8.6f \n','iteration=',t,'obj=',E1,'deltaE=',deltaE, 'RMSE=',err2);
end 
fprintf('shilpi7\n');
fprintf('%d\n', deltaE);
R = U*V';


%H = [1,2,3;4,5,6;7,8,9];
%csvwrite('updated1.csv', full(R));
fprintf('shilpi8\n');
%disp(R);
%frpintf('RMSE %f', err2);