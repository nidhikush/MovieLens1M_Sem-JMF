
fprintf('testing and training\n');
rand('seed',1);
Rating=load('UserXmovie.csv');

fprintf('matrix loaded\n');
num_points = size(Rating',2);                            %where R= 6040*3091
split = 0.8;
split_point = round(num_points*split);           % create 80% split-ants
seq = randperm (num_points);
rand('seed',1);
R_train = Rating(seq(1:split_point),:);
R_test = Rating(seq(split_point+1:end),:);
fprintf('splitted\n');
%fprintf('r_train\n');
%disp(R_train);

%d = size(R_train);
[K,M] = size(R_train);
[L,M] = size(R_test);
dim = 48;
Sf = load('MovieXmovie.csv');
fprintf('matrix reloaded\n');
%disp(size(Sf));
%fprintf('shilpi4\n');
t=0;

gamma=0.1;
alpha = 0.0;
lambda = 0.1;
U=0.1*rand(K,dim);
V=0.1*rand(M,dim);
rand('seed',1);
%Vn = Vj;
%Vn=0.1*rand(K,dim);
numiter = 10;
%fprintf('shilpi5\n');
E1 = 0.5*sum(sum((U*V'.*(R_train>0)-R_train).^2)) + 0.5*alpha*sum(sum((V*V'.*(Sf>0)-Sf).^2)) + 0.5*lambda*(sum(sum(U.^2))+sum(sum(V.^2)));
  %  fprintf('shilpi6\n');
while t < numiter
        gamma=gamma*2;
        nextU=U-gamma*((U*V'.*(R_train>0)-R_train)*V+lambda*U);
        %nextVn = Vn;
        nextV=V-gamma*((U*V'.*(R_train>0)-R_train)'*U+2*alpha*(V*V'.*(Sf>0)-Sf)*V+lambda*V);
        E2=0.5*sum(sum((nextU*nextV'.*(R_train>0)-R_train).^2))+0.5*alpha*sum(sum((nextV*nextV'.*(Sf>0)-Sf).^2))+0.5*lambda*(sum(sum(nextU.^2))+sum(sum(nextV.^2)));
        while E2>=E1
            gamma=gamma/2;
            nextU=U-gamma*((U*V'.*(R_train>0)-R_train)*V+lambda*U);
            nextV=V-gamma*((U*V'.*(R_train>0)-R_train)'*U+2*alpha*(V*V'.*(Sf>0)-Sf)*V+lambda*V);
            E2=0.5*sum(sum((nextU*nextV'.*(R_train>0)-R_train).^2))+0.5*alpha*sum(sum((nextV*nextV'.*(Sf>0)-Sf).^2))+0.5*lambda*(sum(sum(nextU.^2))+sum(sum(nextV.^2)));
        end
    U=nextU;
    V=nextV;
    epsilon = 0.0001;
 %   R=U*V';
    PredR=U*V';  %calculted approximated R
     err=0.0;
    count=0;
    err1=0.0; err2=0.0;
    deltaE=(E1-E2)/E1; %objective function differenc percentage
        if deltaE<=epsilon
            break;
        else
            E1=E2;
            %t = t+1;
            for i=1:K
              for j=1:M
                    if (Rating(i,j)~= 0)
                       err=R_train(i,j)-PredR(i,j);
                       err1=err1+err*err;
                       count=count+1;
                    end
              end
            end
            err2=sqrt(err1/count);
            t=t+1;
        end
        fprintf('Latent Factor = %d\n', dim);
       fprintf(1,'\n%s%d %s%8.6f\n','iteration=',t,'RMSE = ',err2);
end 
%fprintf('shilpi7\n');
%fprintf('%d\n', deltaE);
R = U*V';
totalp = 0;
N = 10;
relval = 3;
ap = zeros(L,1);
for u=1:L
 
      [val,nb]=sort(full(R(u,:)),'descend');
      count = zeros(N,1);
      rel = zeros(N,1);
      P = zeros(N,1);
  %    fprintf('u = %d\n',u);
   %   fileID = fopen('P@5.csv', 'a');
      %csvwrite(fileID, 'u = %d\n', u);
    %    fprintf(fileID,'u = %d\n', u);
    %  fclose(fileID);
      for j = 1:N
            if(R_test(u, nb(j)) >= relval)
                count(j) = count(j) + 1;
                rel(j) = 1;
            end
             s = 0;
             for k = 1:j
                s = s + count(k); 
             end
             P(j) = s/j;
     %        fprintf('j = %d\nP(j)',j);
      %       disp(P(j));
             if j == 10
                totalp = totalp + P(j); 
             end
              
      end
   %  fileID = fopen('P@5.csv', 'a');
    %          fprintf(fileID,'%d\n', P);
              %csvwrite(fileID, P);
     %         fclose(fileID);
      s = 0;
      for j = 1:N
            s = s + rel(j) * P(j);
      end
      if(sum(rel) == 0)
          ap(u) = 0;
      else
      ap(u) = s / sum(rel);
      end
end
avgp_ten = totalp / L;
fprintf('N = %d\n'  ,N);
fprintf('totalp = %f    avgpten = %f\n'  ,totalp, avgp_ten);
%csvwrite('jmf_semantic_p@10.csv', full(ap));
%csvwrite('average_precision@5.csv', full(ap));
%fprintf('ap == \n');
%disp(ap);
map = sum(ap)/L;
fprintf(1,'%s%8.6f  \n','MAP@10 =',map);
fprintf('numiter = %d\n', numiter);

totalp = 0;
N = 5;
relval = 3;
ap = zeros(L,1);
for u=1:L
 
      [val,nb]=sort(full(R(u,:)),'descend');
      count = zeros(N,1);
      rel = zeros(N,1);
      P = zeros(N,1);
  %    fprintf('u = %d\n',u);
   %   fileID = fopen('P@5.csv', 'a');
      %csvwrite(fileID, 'u = %d\n', u);
    %    fprintf(fileID,'u = %d\n', u);
    %  fclose(fileID);
      for j = 1:N
            if(R_test(u, nb(j)) >= relval)
                count(j) = count(j) + 1;
                rel(j) = 1;
            end
             s = 0;
             for k = 1:j
                s = s + count(k); 
             end
             P(j) = s/j;
             if j == 5
                totalp = totalp + P(j); 
             end
     %        fprintf('j = %d\nP(j)',j);
      %       disp(P(j));
              
      end
   %  fileID = fopen('P@5.csv', 'a');
    %          fprintf(fileID,'%d\n', P);
              %csvwrite(fileID, P);
     %         fclose(fileID);
      s = 0;
      for j = 1:N
            s = s + rel(j) * P(j);
      end
      if(sum(rel) == 0)
          ap(u) = 0;
      else
      ap(u) = s / sum(rel);
      end
end
avgp_five = totalp / L;
fprintf('N = %d\n'  ,N);
fprintf('totalp = %f    avgpten = %f\n'  ,totalp, avgp_five);
%csvwrite('jmf_semantic_p@5.csv', full(ap));
%fprintf('ap == \n');
%disp(ap);
map = sum(ap)/L;
fprintf(1,'%s%8.6f  \n','MAP@5 =',map);
fprintf('numiter = %d\n', numiter);









