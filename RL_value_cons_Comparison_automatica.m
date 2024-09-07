clear all;
close all;

ALGS=["TD_lambda", "TD_0", "EMPH_lambda", "TDC_lambda_1", ];
%ALGS=["GTD2_lambda_1", "GTD2_lambda_2", "GTD2_lambda_3", "GTD2_lambda_4", "TDC_lambda_1","TDC_lambda_2", "TDC_NO_CONS", "GTD_NO_CONS"];
%ALGS=["TDC_NET_NEIGH", "TDC_NET_SINGLE", "TDC_NET_FULLY"];% prebaciti na komplemenatarne agente za ovo poredjenje - bolje se ivdi prednost konsenzusa
%ALGS=["ETD_NET_NEIGH", "ETD_NET_SINGLE", "ETD_NET_FULLY"];

%ALG='TDC_lambda'
%ALG='GTD2_lambda';
% ALG='GTD2';
% ALG='TDC';
%ALG='TD';
%ALG='EMPH';
%ALG='TD_lambda'
%ALG='EMPH_lambda';

broj_MC_trajektorija=50;
broj_epizoda=500;
k_max=250;
%k_max=250;
g=0.85;
node=5;

lambda=0.6;
lambda_td0=0;

lambda1_gtd=0;
lambda2_gtd=0;
lambda3_gtd=0.6;
lambda4_gtd=0.6;
lambda_gtd_no_cons=0.6;

lambda1_tdc=0.6;
lambda2_tdc=0;
lambda_tdc_no_cons=0.4;
lambda_tdc_net_comparison=0.5;
%-----
step_w_TDC1=0.3;
step_t_TDC1=2;

step_w_TDC2=0.3;
step_t_TDC2=2;

step_w_TDC_no_cons=0.3;
step_t_TDC_no_cons=1.5;

step_w_TDC_net_comp=0.3;
step_t_TDC_net_comp=1.5

%-------
step_w_GTD1=0.3;
step_t_GTD1=0.3;

step_w_GTD2=0.3;
step_t_GTD2=2;

step_w_GTD3=0.3;
step_t_GTD3=0.3;

step_w_GTD4=0.3;
step_t_GTD4=2;

step_w_GTD_no_cons=0.3;
step_t_GTD_no_cons=0.3;

mu_w_TD=0.5;
mu_w_EMPH=0.1;
mu_w_EMPH_l=0.1;

n_features=7;
p_t=0.8;

cons_steps=1;
N=10;

%%%%% True value function calculation
V(:,1)=zeros(15,1);
for i=1:100
   for s=1:14
    if mod(s,2)==1   
    V(s,i+1)=p_t*(-4+g*(0.8*V(s+2,i)+0.2*V(s,i)))+(1-p_t)*(-1+g*((1/(1*s))*V(s+1,i)+(1-1/(1*s))*V(s,i)));
    else
    V(s,i+1)=-1+g*((1/(1*s))*V(s+1,i)+(1-1/(1*s))*V(s,i));    
   end; 
    V(15,i+1)=0;
   end;
end
%%%%%%%%%%%%%%%%%%
%multi-agent

% A=[1/2 , 1/2 0;1/3 1/3 , 1/3; 0, 1/2, 1/2];
% A=[1 , 0 ; 0 , 1];
% A=[ 1/3 , 1/3 , 1/3 , 0,    0,     0,    0, 0 , 0, 0, 0, 0 , 0, 0, 0 , 0,   0,    0,   0, 0; 
%     1/4 , 1/4 , 1/4 , 1/4, 0,     0,    0, 0 , 0, 0, 0, 0 , 0, 0, 0 , 0,   0,    0,   0, 0;
%     1/5 , 1/5 , 1/5, 1/5, 1/5, 0,     0,    0, 0 , 0, 0, 0, 0 , 0, 0, 0 , 0,   0,    0,   0;
%     0,    0,    1/5 , 1/5 , 1/5, 0,     0,    0, 0 , 0, 0, 0, 0 , 0, 0, 1/5 , 0,   0,    1/5,   0;
%     0,    0,     0,   1/5 , 1/5, 1/5, 1/5, 1/5, 0,     0,    0, 0 , 0, 0, 0, 0 , 0, 0, 0 , 0;
%     0,    0,     0,    1/6, 1/6 , 1/6, 1/6, 1/6, 1/6 , 0,     0,    0, 0 , 0, 0, 0, 0 , 0, 0, 0;
%     0,    0,     0,    1/7, 1/7 , 1/7, 1/7, 1/7, 1/7 , 1/7,  0,    0, 0 , 0, 0, 0, 0 , 0, 0, 0;
%     0,    0,     0,    0, 1/7 , 1/7, 1/7, 1/7, 1/7 , 1/7, 1/7 ,  0,    0, 0 , 0, 0, 0, 0 , 0, 0;
%     0,    0,     0,    0, 0 , 0, 1/6, 1/6, 1/6 , 1/6, 1/6, 1/6 ,  0,    0, 0 , 0, 0, 0, 0 , 0;
%     0,    0,     0,    0, 0 , 0, 0, 0, 1/5 , 1/5, 1/5, 1/5 , 1/5, 0,  0,    0, 0 , 0, 0, 0;
%     0,    0,     0,    0, 0 , 0, 0, 0, 0 , 0, 1/6, 1/6 , 1/6, 1/6, 1/6, 1/6,  0 , 0, 0, 0;
%     0,    0,     0,    0, 0 , 0, 0, 0, 0 , 1/5, 1/5, 1/5, 1/5,  1/5, 0, 0, 0,0 ,0,0;
%     0,    0,     0,    0, 0 , 0, 0, 0, 0 , 0, 0, 1/6 , 1/6, 1/6, 1/6, 1/6,  1/6, 0, 0, 0;
%     0,    1/5,     0,    0, 0 , 0, 0, 0, 0 , 0, 0, 0 , 1/5, 1/5, 0, 1/5,  1/5, 0, 0, 0;
%     0,    0,     0,    0, 0 , 0, 0, 0, 0 , 0, 0, 0 , 1/5, 1/5, 1/5, 1/5,  1/5, 0, 0, 0;
%     1/5,    0,     1/5,    0, 0 , 0, 0, 0, 0 , 0, 0, 0 , 0,    0,  1/5, 1/5,  1/5, 0, 0, 0;
%     0,    0,     0,    0, 0 , 0, 0, 0, 0 , 0, 0, 0 , 0,   1/5, 1/5, 1/5,  1/5, 1/5, 0, 0;
%     0,    0,     0,    0, 0 , 0, 0, 0, 0 , 0, 0, 0 , 1/6, 1/6, 0,   0,  1/6, 1/6, 1/6, 1/6;
%     0,    0,     0,    0, 0 , 0, 0, 0, 0 , 0, 0, 0 , 0,   0,    0, 1/4,  1/4, 0, 1/4, 1/4;
%     0,    0,     0,    0, 0 , 0, 0, 0, 0 , 0, 0, 0 , 0,   0,    0,   0,  1/4, 1/4, 1/4, 1/4;
%    ];
A=[ 1/4 , 1/4 , 1/4 , 0,    0,     0,    0, 0 , 1/4, 0; 
    1/4 , 1/4 , 1/4 , 1/4, 0,     0,    0, 0 , 0, 0;
    0 , 1/4 , 1/4, 1/4, 1/4, 0,     0,    0, 0 , 0;
    0,    0,    1/3 , 1/3 , 1/3, 0,     0,    0, 0 , 0;
    0,    0,     0,   0 , 1/4, 1/4, 1/4, 1/4, 0,     0;
   1/5,    0,     0,    0, 0 , 1/5, 1/5, 1/5, 1/5 , 0;
    0,    0,     0,   0, 1/4 , 1/4, 1/4, 0, 1/4 , 0;
    0,    0,     0,    0, 0 , 0, 1/4, 1/4, 1/4 , 1/4 ;
     1/4 , 0, 0, 0, 0 , 1/4, 0, 0 ,  1/4,    1/4;
      0 , 0, 0, 0, 0 , 0, 1/4, 1/4 , 1/4, 1/4
    ];

A_neigh=[ 1/4 , 1/4 , 1/4 , 0,    0,     0,    0, 0 , 1/4, 0; 
    1/4 , 1/4 , 1/4 , 1/4, 0,     0,    0, 0 , 0, 0;
    0 , 1/4 , 1/4, 1/4, 1/4, 0,     0,    0, 0 , 0;
    0,    0,    1/3 , 1/3 , 1/3, 0,     0,    0, 0 , 0;
    0,    0,     0,   0 , 1/4, 1/4, 1/4, 1/4, 0,     0;
   1/5,    0,     0,    0, 0 , 1/5, 1/5, 1/5, 1/5 , 0;
    0,    0,     0,   0, 1/4 , 1/4, 1/4, 0, 1/4 , 0;
    0,    0,     0,    0, 0 , 0, 1/4, 1/4, 1/4 , 1/4 ;
     1/4 , 0, 0, 0, 0 , 1/4, 0, 0 ,  1/4,    1/4;
      0 , 0, 0, 0, 0 , 0, 1/4, 1/4 , 1/4, 1/4
    ];
A_fully=(1/N)*ones(N,N);  % fully connected
A_single=eye(N);   % no communications - N single agent algortihms
% A=(A+A')/2;  $ symmetric communication graph
% p_b=ones(N,1)-[0.3596; 0.2412;0.1303;0.1888;0.1512;0.1937;0.2467;0.2757;0.2525;0.3918;0.2727;0.4145;0.3126;0.4192;
%         0.1802; 0.3254; 0.1964; 0.3239;  0.3317;   0.1227];
%     
  state(1:N,1)=   [1; 2; 4; 5; 5; 3;  8; 1;5;6];
 end_state(1:N,1)=[3; 4; 7;15;14;14; 14; 6;10;11];
  p_b=ones(N,1)-[0.3596; 0.2412;0.1303;0.1888;0.1512;0.1937;0.2467;0.2757;0.2525;0.3918];
 % p_b=ones(N,1)*0.5;  
  % state(1:N,1)=    [5;  5;  5; 5;  5; 5; 5;  10;  8; 5];
  % end_state(1:N,1)=[10; 8;  7; 10; 15; 8; 15; 15; 14; 13]; 
  end_state(1:N,1)=ones(1,N)*15; 
     state(1:N,1)=ones(1,N);
 
%  p_b=ones(N,1)-[0.3596; 0.2412;0.1303];
 % state(1:N,1)=[1; 5; 10];
 % end_state(1:N,1)=[7;11;15];

 
MSEs_av_mc=zeros(k_max+1,size(ALGS,2));

figure;
for br_mc=1:broj_MC_trajektorija
    
theta(:,1,:)=zeros(n_features*N,1,size(ALGS,2));
w(:,1,:)=zeros(n_features*N,1,size(ALGS,2));
  Fe(:,1,:)=ones(N,1,size(ALGS,2));
el(:,1,:)=zeros(n_features*N,1,size(ALGS,2));

pp=[];
for i=1:N
pp=[pp ; features(state(i,1))];
end;
phi(:,1)=pp;
r(1:N,1)=0;


e=0;
k=1;
while (e<broj_epizoda) && (k<=k_max)
    
while (sum(state(1:N,k)<end_state)>0) && (k<=k_max) 
k=k+1;
p_ex=0.8;
p_mw=1./(1.*state(:,k-1));
pp=[];
no_update=zeros(N,1);
%generate new state for each agent
for i=1:N
    if state(i,k-1)~=end_state(i)
if (rand(1)<p_b(i)) && (mod(state(i,k-1),2) == 1)
    %exit
    xi(i,k)= p_t/p_b(i); % target/beh odnos verovatnoca za dobijenu akciju
    r(i,k)=-4;
    if rand(1)<p_ex
        %move
        state(i,k)=state(i,k-1)+2;
    else
        %stuck
        state(i,k)=state(i,k-1);
    end;
   
else
    %motorway
    if (mod(state(i,k-1),2) == 0)
    xi(i,k)= 1; % target/beh odnos verovatnoca za dobijenu akciju 
    else
     xi(i,k)= (1-p_t)/(1-p_b(i)); % target/beh odnos verovatnoca za dobijenu akciju   
    end;
    
     r(i,k)=-1;
    if rand(1)<p_mw(i)
        %move
        state(i,k)=state(i,k-1)+1;
    else
        %stuck
        state(i,k)=state(i,k-1);
    end;
end;
    else
     state(i,k)=state(i,k-1);
     no_update(i)=1;
     xi(i,k)= 0;
     r(i,k)=0;
    end;
    
pp=[pp ; features(state(i,k))];

end
phi(:,k)=pp;

theta1(1:N*n_features,k,:)=zeros(N*n_features,1,size(ALGS,2));
w1(1:N*n_features,k,:)=zeros(N*n_features,1,size(ALGS,2));

% algorithm step
for jj=1:size(ALGS,2)
  for i=1:N
      
    if ALGS(jj)=="GTD2_lambda_1"
        
  if no_update(i)==1
        el((1+(i-1)*n_features):(i*n_features),k,jj)=el((1+(i-1)*n_features):(i*n_features),k-1,jj);
    else
    el((1+(i-1)*n_features):(i*n_features),k,jj)=g*lambda1_gtd*xi(i,k-1)*el((1+(i-1)*n_features):(i*n_features),k-1,jj)+phi((1+(i-1)*n_features):(i*n_features),k-1);
  end;  
     
       theta1((1+(i-1)*n_features):(i*n_features),k,jj)= ...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj)+ ...
    step_t_GTD1*(el((1+(i-1)*n_features):(i*n_features),k,jj)*xi(i,k)*(r(i,k)+ ...
    g*phi((1+(i-1)*n_features):(i*n_features),k)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj)- ...
    phi((1+(i-1)*n_features):(i*n_features),k-1)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj))-...
    phi((1+(i-1)*n_features):(i*n_features),k-1)*phi((1+(i-1)*n_features):(i*n_features),k-1)'*...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj));

w1((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k-1,jj)+...
    step_w_GTD1*xi(i,k)*(phi((1+(i-1)*n_features):(i*n_features),k-1)-g*phi((1+(i-1)*n_features):(i*n_features),k))*...
    el((1+(i-1)*n_features):(i*n_features),k,jj)'*theta((1+(i-1)*n_features):(i*n_features),k-1,jj);


    elseif ALGS(jj)=="TDC_lambda_1"
        
  if no_update(i)==1
        el((1+(i-1)*n_features):(i*n_features),k,jj)=el((1+(i-1)*n_features):(i*n_features),k-1,jj);
  else
    el((1+(i-1)*n_features):(i*n_features),k,jj)=g*lambda1_tdc*xi(i,k-1)*el((1+(i-1)*n_features):(i*n_features),k-1,jj)+phi((1+(i-1)*n_features):(i*n_features),k-1);
  end;
  
  theta1((1+(i-1)*n_features):(i*n_features),k,jj)= ...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj)+ ...
    step_t_TDC1*(el((1+(i-1)*n_features):(i*n_features),k,jj)*xi(i,k)*(r(i,k)+ ...
    g*phi((1+(i-1)*n_features):(i*n_features),k)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj)- ...
    phi((1+(i-1)*n_features):(i*n_features),k-1)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj))-...
    phi((1+(i-1)*n_features):(i*n_features),k-1)*phi((1+(i-1)*n_features):(i*n_features),k-1)'*...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj));

w1((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k-1,jj)+...
    step_w_TDC1*(el((1+(i-1)*n_features):(i*n_features),k,jj)*xi(i,k)*(r(i,k)+ ...
    g*phi((1+(i-1)*n_features):(i*n_features),k)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj)- ...
    phi((1+(i-1)*n_features):(i*n_features),k-1)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj))-...
    xi(i,k)*(1-lambda1_tdc)*g*phi((1+(i-1)*n_features):(i*n_features),k)*el((1+(i-1)*n_features):(i*n_features),k,jj)'*...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj));
      
    
elseif ALGS(jj)=="EMPH"
    
    if no_update(i)==1
        Fe(i,k,jj)=Fe(i,k-1,jj);
    else
    Fe(i,k,jj)=g*xi(i,k-1)*Fe(i,k-1,jj)+1;
    end;
    
 w1((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k-1,jj)+...
    mu_w_EMPH*((r(i,k)+(-phi((1+(i-1)*n_features):(i*n_features),k-1)'+g*phi((1+(i-1)*n_features):(i*n_features),k)')*...
    w((1+(i-1)*n_features):(i*n_features),k-1,jj))*phi((1+(i-1)*n_features):(i*n_features),k-1))*xi(i,k)*Fe(i,k,jj);

elseif ALGS(jj)=="TD_lambda"
    
    if no_update(i)==1
        el((1+(i-1)*n_features):(i*n_features),k,jj)=el((1+(i-1)*n_features):(i*n_features),k-1,jj);
    else
    el((1+(i-1)*n_features):(i*n_features),k,jj)=g*lambda*xi(i,k-1)*el((1+(i-1)*n_features):(i*n_features),k-1,jj)+phi((1+(i-1)*n_features):(i*n_features),k-1);
    end;
    
    
         w1((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k-1,jj)+...
    mu_w_TD*((r(i,k)+(-phi((1+(i-1)*n_features):(i*n_features),k-1)'+g*phi((1+(i-1)*n_features):(i*n_features),k)')*...
    w((1+(i-1)*n_features):(i*n_features),k-1,jj))*el((1+(i-1)*n_features):(i*n_features),k,jj))*xi(i,k);

elseif ALGS(jj)=="TD_0"
    
    if no_update(i)==1
        el((1+(i-1)*n_features):(i*n_features),k,jj)=el((1+(i-1)*n_features):(i*n_features),k-1,jj);
    else
    el((1+(i-1)*n_features):(i*n_features),k,jj)=g*lambda_td0*xi(i,k-1)*el((1+(i-1)*n_features):(i*n_features),k-1,jj)+phi((1+(i-1)*n_features):(i*n_features),k-1);
    end;
    
    
         w1((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k-1,jj)+...
    mu_w_TD*((r(i,k)+(-phi((1+(i-1)*n_features):(i*n_features),k-1)'+g*phi((1+(i-1)*n_features):(i*n_features),k)')*...
    w((1+(i-1)*n_features):(i*n_features),k-1,jj))*el((1+(i-1)*n_features):(i*n_features),k,jj))*xi(i,k);

elseif ALGS(jj)=="EMPH_lambda" | ALGS(jj)=="ETD_NET_NEIGH" | ALGS(jj)=="ETD_NET_FULLY" | ALGS(jj)=="ETD_NET_SINGLE"
    % dodati i state interest function
    
     if no_update(i)==1
        Fe(i,k,jj)=Fe(i,k-1,jj);
        el((1+(i-1)*n_features):(i*n_features),k,jj)=el((1+(i-1)*n_features):(i*n_features),k-1,jj);
    else
   Fe(i,k,jj)=g*xi(i,k-1)*Fe(i,k-1,jj)+1;
    
     el((1+(i-1)*n_features):(i*n_features),k,jj)=g*lambda*el((1+(i-1)*n_features):(i*n_features),k-1,jj)+(lambda+(1-lambda)*Fe(i,k,jj))*phi((1+(i-1)*n_features):(i*n_features),k-1);
     
    end;
    
    
      w1((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k-1,jj)+...
    mu_w_EMPH_l*((r(i,k)+(-phi((1+(i-1)*n_features):(i*n_features),k-1)'+g*phi((1+(i-1)*n_features):(i*n_features),k)')*...
    w((1+(i-1)*n_features):(i*n_features),k-1,jj))*el((1+(i-1)*n_features):(i*n_features),k,jj))*xi(i,k);

elseif ALGS(jj)=="GTD2_lambda_2"
    
  if no_update(i)==1
        el((1+(i-1)*n_features):(i*n_features),k,jj)=el((1+(i-1)*n_features):(i*n_features),k-1,jj);
    else
    el((1+(i-1)*n_features):(i*n_features),k,jj)=g*lambda2_gtd*xi(i,k-1)*el((1+(i-1)*n_features):(i*n_features),k-1,jj)+phi((1+(i-1)*n_features):(i*n_features),k-1);
  end;  
     
       theta1((1+(i-1)*n_features):(i*n_features),k,jj)= ...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj)+ ...
    step_t_GTD2*(el((1+(i-1)*n_features):(i*n_features),k,jj)*xi(i,k)*(r(i,k)+ ...
    g*phi((1+(i-1)*n_features):(i*n_features),k)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj)- ...
    phi((1+(i-1)*n_features):(i*n_features),k-1)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj))-...
    phi((1+(i-1)*n_features):(i*n_features),k-1)*phi((1+(i-1)*n_features):(i*n_features),k-1)'*...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj));

w1((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k-1,jj)+...
    step_w_GTD2*xi(i,k)*(phi((1+(i-1)*n_features):(i*n_features),k-1)-g*phi((1+(i-1)*n_features):(i*n_features),k))*...
    el((1+(i-1)*n_features):(i*n_features),k,jj)'*theta((1+(i-1)*n_features):(i*n_features),k-1,jj);

elseif ALGS(jj)=="TDC_lambda_2"
    
  if no_update(i)==1
        el((1+(i-1)*n_features):(i*n_features),k,jj)=el((1+(i-1)*n_features):(i*n_features),k-1,jj);
  else
    el((1+(i-1)*n_features):(i*n_features),k,jj)=g*lambda2_tdc*xi(i,k-1)*el((1+(i-1)*n_features):(i*n_features),k-1,jj)+phi((1+(i-1)*n_features):(i*n_features),k-1);
  end;
  
  theta1((1+(i-1)*n_features):(i*n_features),k,jj)= ...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj)+ ...
    step_t_TDC2*(el((1+(i-1)*n_features):(i*n_features),k,jj)*xi(i,k)*(r(i,k)+ ...
    g*phi((1+(i-1)*n_features):(i*n_features),k)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj)- ...
    phi((1+(i-1)*n_features):(i*n_features),k-1)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj))-...
    phi((1+(i-1)*n_features):(i*n_features),k-1)*phi((1+(i-1)*n_features):(i*n_features),k-1)'*...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj));

w1((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k-1,jj)+...
    step_w_TDC2*(el((1+(i-1)*n_features):(i*n_features),k,jj)*xi(i,k)*(r(i,k)+ ...
    g*phi((1+(i-1)*n_features):(i*n_features),k)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj)- ...
    phi((1+(i-1)*n_features):(i*n_features),k-1)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj))-...
    xi(i,k)*(1-lambda2_tdc)*g*phi((1+(i-1)*n_features):(i*n_features),k)*el((1+(i-1)*n_features):(i*n_features),k,jj)'*...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj));

elseif ALGS(jj)=="GTD2_lambda_3"
    
  if no_update(i)==1
        el((1+(i-1)*n_features):(i*n_features),k,jj)=el((1+(i-1)*n_features):(i*n_features),k-1,jj);
    else
    el((1+(i-1)*n_features):(i*n_features),k,jj)=g*lambda3_gtd*xi(i,k-1)*el((1+(i-1)*n_features):(i*n_features),k-1,jj)+phi((1+(i-1)*n_features):(i*n_features),k-1);
  end;  
     
       theta1((1+(i-1)*n_features):(i*n_features),k,jj)= ...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj)+ ...
    step_t_GTD3*(el((1+(i-1)*n_features):(i*n_features),k,jj)*xi(i,k)*(r(i,k)+ ...
    g*phi((1+(i-1)*n_features):(i*n_features),k)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj)- ...
    phi((1+(i-1)*n_features):(i*n_features),k-1)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj))-...
    phi((1+(i-1)*n_features):(i*n_features),k-1)*phi((1+(i-1)*n_features):(i*n_features),k-1)'*...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj));

w1((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k-1,jj)+...
    step_w_GTD3*xi(i,k)*(phi((1+(i-1)*n_features):(i*n_features),k-1)-g*phi((1+(i-1)*n_features):(i*n_features),k))*...
    el((1+(i-1)*n_features):(i*n_features),k,jj)'*theta((1+(i-1)*n_features):(i*n_features),k-1,jj);

elseif ALGS(jj)=="GTD2_lambda_4"
    
  if no_update(i)==1
        el((1+(i-1)*n_features):(i*n_features),k,jj)=el((1+(i-1)*n_features):(i*n_features),k-1,jj);
    else
    el((1+(i-1)*n_features):(i*n_features),k,jj)=g*lambda4_gtd*xi(i,k-1)*el((1+(i-1)*n_features):(i*n_features),k-1,jj)+phi((1+(i-1)*n_features):(i*n_features),k-1);
  end;  
     
       theta1((1+(i-1)*n_features):(i*n_features),k,jj)= ...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj)+ ...
    step_t_GTD4*(el((1+(i-1)*n_features):(i*n_features),k,jj)*xi(i,k)*(r(i,k)+ ...
    g*phi((1+(i-1)*n_features):(i*n_features),k)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj)- ...
    phi((1+(i-1)*n_features):(i*n_features),k-1)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj))-...
    phi((1+(i-1)*n_features):(i*n_features),k-1)*phi((1+(i-1)*n_features):(i*n_features),k-1)'*...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj));

w1((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k-1,jj)+...
    step_w_GTD4*xi(i,k)*(phi((1+(i-1)*n_features):(i*n_features),k-1)-g*phi((1+(i-1)*n_features):(i*n_features),k))*...
    el((1+(i-1)*n_features):(i*n_features),k,jj)'*theta((1+(i-1)*n_features):(i*n_features),k-1,jj);

elseif ALGS(jj)=="GTD_NO_CONS"
    
  if no_update(i)==1
        el((1+(i-1)*n_features):(i*n_features),k,jj)=el((1+(i-1)*n_features):(i*n_features),k-1,jj);
    else
    el((1+(i-1)*n_features):(i*n_features),k,jj)=g*lambda_gtd_no_cons*xi(i,k-1)*el((1+(i-1)*n_features):(i*n_features),k-1,jj)+phi((1+(i-1)*n_features):(i*n_features),k-1);
  end;  
     
       theta1((1+(i-1)*n_features):(i*n_features),k,jj)= ...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj)+ ...
    step_t_GTD_no_cons*(el((1+(i-1)*n_features):(i*n_features),k,jj)*xi(i,k)*(r(i,k)+ ...
    g*phi((1+(i-1)*n_features):(i*n_features),k)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj)- ...
    phi((1+(i-1)*n_features):(i*n_features),k-1)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj))-...
    phi((1+(i-1)*n_features):(i*n_features),k-1)*phi((1+(i-1)*n_features):(i*n_features),k-1)'*...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj));

w1((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k-1,jj)+...
    step_w_GTD_no_cons*xi(i,k)*(phi((1+(i-1)*n_features):(i*n_features),k-1)-g*phi((1+(i-1)*n_features):(i*n_features),k))*...
    el((1+(i-1)*n_features):(i*n_features),k,jj)'*theta((1+(i-1)*n_features):(i*n_features),k-1,jj);

elseif ALGS(jj)=="TDC_NO_CONS"
    
  if no_update(i)==1
        el((1+(i-1)*n_features):(i*n_features),k,jj)=el((1+(i-1)*n_features):(i*n_features),k-1,jj);
  else
    el((1+(i-1)*n_features):(i*n_features),k,jj)=g*lambda_tdc_no_cons*xi(i,k-1)*el((1+(i-1)*n_features):(i*n_features),k-1,jj)+phi((1+(i-1)*n_features):(i*n_features),k-1);
  end;
  
  theta1((1+(i-1)*n_features):(i*n_features),k,jj)= ...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj)+ ...
    step_t_TDC_no_cons*(el((1+(i-1)*n_features):(i*n_features),k,jj)*xi(i,k)*(r(i,k)+ ...
    g*phi((1+(i-1)*n_features):(i*n_features),k)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj)- ...
    phi((1+(i-1)*n_features):(i*n_features),k-1)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj))-...
    phi((1+(i-1)*n_features):(i*n_features),k-1)*phi((1+(i-1)*n_features):(i*n_features),k-1)'*...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj));

w1((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k-1,jj)+...
    step_w_TDC_no_cons*(el((1+(i-1)*n_features):(i*n_features),k,jj)*xi(i,k)*(r(i,k)+ ...
    g*phi((1+(i-1)*n_features):(i*n_features),k)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj)- ...
    phi((1+(i-1)*n_features):(i*n_features),k-1)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj))-...
    xi(i,k)*(1-lambda_tdc_no_cons)*g*phi((1+(i-1)*n_features):(i*n_features),k)*el((1+(i-1)*n_features):(i*n_features),k,jj)'*...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj));

elseif ALGS(jj)=="TDC_NET_NEIGH"
    
  if no_update(i)==1
        el((1+(i-1)*n_features):(i*n_features),k,jj)=el((1+(i-1)*n_features):(i*n_features),k-1,jj);
  else
    el((1+(i-1)*n_features):(i*n_features),k,jj)=g*lambda_tdc_net_comparison*xi(i,k-1)*el((1+(i-1)*n_features):(i*n_features),k-1,jj)+phi((1+(i-1)*n_features):(i*n_features),k-1);
  end;
  
  theta1((1+(i-1)*n_features):(i*n_features),k,jj)= ...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj)+ ...
    step_t_TDC_net_comp*(el((1+(i-1)*n_features):(i*n_features),k,jj)*xi(i,k)*(r(i,k)+ ...
    g*phi((1+(i-1)*n_features):(i*n_features),k)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj)- ...
    phi((1+(i-1)*n_features):(i*n_features),k-1)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj))-...
    phi((1+(i-1)*n_features):(i*n_features),k-1)*phi((1+(i-1)*n_features):(i*n_features),k-1)'*...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj));

w1((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k-1,jj)+...
    step_w_TDC_net_comp*(el((1+(i-1)*n_features):(i*n_features),k,jj)*xi(i,k)*(r(i,k)+ ...
    g*phi((1+(i-1)*n_features):(i*n_features),k)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj)- ...
    phi((1+(i-1)*n_features):(i*n_features),k-1)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj))-...
    xi(i,k)*(1-lambda_tdc_no_cons)*g*phi((1+(i-1)*n_features):(i*n_features),k)*el((1+(i-1)*n_features):(i*n_features),k,jj)'*...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj));

elseif ALGS(jj)=="TDC_NET_FULLY"
    
  if no_update(i)==1
        el((1+(i-1)*n_features):(i*n_features),k,jj)=el((1+(i-1)*n_features):(i*n_features),k-1,jj);
  else
    el((1+(i-1)*n_features):(i*n_features),k,jj)=g*lambda_tdc_net_comparison*xi(i,k-1)*el((1+(i-1)*n_features):(i*n_features),k-1,jj)+phi((1+(i-1)*n_features):(i*n_features),k-1);
  end;
  
  theta1((1+(i-1)*n_features):(i*n_features),k,jj)= ...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj)+ ...
    step_t_TDC_net_comp*(el((1+(i-1)*n_features):(i*n_features),k,jj)*xi(i,k)*(r(i,k)+ ...
    g*phi((1+(i-1)*n_features):(i*n_features),k)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj)- ...
    phi((1+(i-1)*n_features):(i*n_features),k-1)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj))-...
    phi((1+(i-1)*n_features):(i*n_features),k-1)*phi((1+(i-1)*n_features):(i*n_features),k-1)'*...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj));

w1((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k-1,jj)+...
    step_w_TDC_net_comp*(el((1+(i-1)*n_features):(i*n_features),k,jj)*xi(i,k)*(r(i,k)+ ...
    g*phi((1+(i-1)*n_features):(i*n_features),k)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj)- ...
    phi((1+(i-1)*n_features):(i*n_features),k-1)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj))-...
    xi(i,k)*(1-lambda_tdc_no_cons)*g*phi((1+(i-1)*n_features):(i*n_features),k)*el((1+(i-1)*n_features):(i*n_features),k,jj)'*...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj));

elseif ALGS(jj)=="TDC_NET_SINGLE"
    
  if no_update(i)==1
        el((1+(i-1)*n_features):(i*n_features),k,jj)=el((1+(i-1)*n_features):(i*n_features),k-1,jj);
  else
    el((1+(i-1)*n_features):(i*n_features),k,jj)=g*lambda_tdc_net_comparison*xi(i,k-1)*el((1+(i-1)*n_features):(i*n_features),k-1,jj)+phi((1+(i-1)*n_features):(i*n_features),k-1);
  end;
  
  theta1((1+(i-1)*n_features):(i*n_features),k,jj)= ...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj)+ ...
    step_t_TDC_net_comp*(el((1+(i-1)*n_features):(i*n_features),k,jj)*xi(i,k)*(r(i,k)+ ...
    g*phi((1+(i-1)*n_features):(i*n_features),k)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj)- ...
    phi((1+(i-1)*n_features):(i*n_features),k-1)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj))-...
    phi((1+(i-1)*n_features):(i*n_features),k-1)*phi((1+(i-1)*n_features):(i*n_features),k-1)'*...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj));

w1((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k-1,jj)+...
    step_w_TDC_net_comp*(el((1+(i-1)*n_features):(i*n_features),k,jj)*xi(i,k)*(r(i,k)+ ...
    g*phi((1+(i-1)*n_features):(i*n_features),k)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj)- ...
    phi((1+(i-1)*n_features):(i*n_features),k-1)'*w((1+(i-1)*n_features):(i*n_features),k-1,jj))-...
    xi(i,k)*(1-lambda_tdc_no_cons)*g*phi((1+(i-1)*n_features):(i*n_features),k)*el((1+(i-1)*n_features):(i*n_features),k,jj)'*...
    theta((1+(i-1)*n_features):(i*n_features),k-1,jj));
    end
  end

  
end;

%consensus
 theta(1:N*n_features,k,:)=zeros(N*n_features,1,size(ALGS,2));
 w(1:N*n_features,k,:)=zeros(N*n_features,1,size(ALGS,2));

for jj=1:size(ALGS,2)
for i=1:N
 if (ALGS(jj)=="TDC_NO_CONS") | (ALGS(jj)=="GTD_NO_CONS") 
    theta((1+(i-1)*n_features):(i*n_features),k,jj)=theta1((1+(i-1)*n_features):(i*n_features),k,jj);    
 elseif ALGS(jj)=="TDC_NET_NEIGH"
    for j=1:N
  theta((1+(i-1)*n_features):(i*n_features),k,jj)=theta((1+(i-1)*n_features):(i*n_features),k,jj)+...
    A_neigh(i,j)*theta1((1+(j-1)*n_features):(j*n_features),k,jj);
  end
 elseif ALGS(jj)=="TDC_NET_SINGLE"
    for j=1:N
  theta((1+(i-1)*n_features):(i*n_features),k,jj)=theta((1+(i-1)*n_features):(i*n_features),k,jj)+...
    A_single(i,j)*theta1((1+(j-1)*n_features):(j*n_features),k,jj);
  end
 elseif ALGS(jj)=="TDC_NET_FULLY"
   for j=1:N
  theta((1+(i-1)*n_features):(i*n_features),k,jj)=theta((1+(i-1)*n_features):(i*n_features),k,jj)+...
    A_fully(i,j)*theta1((1+(j-1)*n_features):(j*n_features),k,jj);
  end
 else
  for j=1:N
  theta((1+(i-1)*n_features):(i*n_features),k,jj)=theta((1+(i-1)*n_features):(i*n_features),k,jj)+...
    A(i,j)*theta1((1+(j-1)*n_features):(j*n_features),k,jj);
  end
 end
 
 if ALGS(jj)=="TDC_NET_NEIGH" | ALGS(jj)=="ETD_NET_NEIGH"
    for j=1:N
    w((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k,jj)+...
    A_neigh(i,j)*w1((1+(j-1)*n_features):(j*n_features),k,jj);
    end;
 elseif ALGS(jj)=="TDC_NET_SINGLE" | ALGS(jj)=="ETD_NET_SINGLE"
    for j=1:N
    w((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k,jj)+...
    A_single(i,j)*w1((1+(j-1)*n_features):(j*n_features),k,jj);
    end;
 elseif ALGS(jj)=="TDC_NET_FULLY" | ALGS(jj)=="ETD_NET_FULLY"
    for j=1:N
    w((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k,jj)+...
    A_fully(i,j)*w1((1+(j-1)*n_features):(j*n_features),k,jj);
    end;
 else
     
   for j=1:N
    w((1+(i-1)*n_features):(i*n_features),k,jj)=w((1+(i-1)*n_features):(i*n_features),k,jj)+...
    A(i,j)*w1((1+(j-1)*n_features):(j*n_features),k,jj);
    end;
 end
end;
end

%w(1:N*n_features,k,:)=w1(1:N*n_features,k,:);
%theta(1:N*n_features,k,:)=theta1(1:N*n_features,k,:);
%     

% for c=1:(cons_steps-1)
%     k=k+1;
%     state(:,k)=state(:,k-1);
%      xi(:,k)= xi(:,k-1);
%      r(:,k)=r(:,k-1);
%        Fe(:,k)=Fe(:,k-1);
% el(:,k,:)=el(:,k-1,:);
% 
%      theta(1:N*n_features,k)=zeros(N*n_features,1,:);
% w(1:N*n_features,k)=zeros(N*n_features,1,:);
% for i=1:N
%     for j=1:N
% theta((1+(i-1)*n_features):(i*n_features),k,:)=theta((1+(i-1)*n_features):(i*n_features),k,:)+...
%     A(i,j)*theta((1+(j-1)*n_features):(j*n_features),k-1,:);
% w((1+(i-1)*n_features):(i*n_features),k,:)=w((1+(i-1)*n_features):(i*n_features),k,:)+...
%     A(i,j)*w((1+(j-1)*n_features):(j*n_features),k-1,:);
%     end;
% end;
%     
% end;

end;


    


e=e+1;

state(1:N,k)=state(1:N,1);

pp=[];
for i=1:N
pp=[pp ; features(state(i,k))];
end;
phi(:,k)=pp;
r(1:N,k)=0;
%el(:,k)=phi(:,k);
el(:,1)=zeros(n_features*N,1);

end;

F=[];
for i=1:15 
    F=[F ; features(i)'];
end;
 

for i=1:N
    for jj=1:size(ALGS,2)
        V_est(:,i,jj)=F*w((1+(i-1)*n_features):(i*n_features),k,jj);
    end
end

VV=V(:,size(V,2));

V_nodes_avg=zeros(15,k,size(ALGS,2));
for jj=1:size(ALGS,2)
for i=1:N
  V_nodes_avg(:,:,jj)=V_nodes_avg(:,:,jj)+F*w((1+(i-1)*n_features):(i*n_features),:,jj);  
end
V_nodes_avg(:,:,jj)=V_nodes_avg(:,:,jj)/N;
end


MSEs=squeeze(mean((V_nodes_avg-VV).^2,1));

hold on
MD=30;
%plot(1:k, MSEs(1:k,1),'-x','MarkerIndices',1:MD:k)
%plot(1:k, MSEs(1:k,2),'-o','MarkerIndices',1:MD:k)

plot(1:k, MSEs(1:k,1),'r-v','MarkerIndices',1:MD:k)
plot(1:k, MSEs(1:k,2),'b-o','MarkerIndices',1:MD:k)
plot(1:k, MSEs(1:k,3),'m-*','MarkerIndices',1:MD:k)
plot(1:k, MSEs(1:k,4),'g-s','MarkerIndices',1:MD:k)

MSEs_av_mc=MSEs_av_mc+MSEs;


end % MC trajektorije for petlja

MSEs_av_mc=MSEs_av_mc./broj_MC_trajektorija;

plot(1:k, MSEs_av_mc(1:k,1),'k-v','MarkerSize',10,'MarkerIndices',1:MD:k,'LineWidth',1.7)
plot(1:k, MSEs_av_mc(1:k,2),'k-o','MarkerSize',10,'MarkerIndices',1:MD:k,'LineWidth',1.7)
plot(1:k, MSEs_av_mc(1:k,3),'k-*','MarkerSize',10,'MarkerIndices',1:MD:k,'LineWidth',1.7)
plot(1:k, MSEs_av_mc(1:k,4),'k-s','MarkerSize',10,'MarkerIndices',1:MD:k,'LineWidth',1.7)

hold off
xlabel('Iterations');
ylabel('MSE');
leg=legend;
set(leg,'Interpreter','latex');
legend('D-TD(0.6)', 'D-TD(0)', 'D-ETD(0.6)', 'D-TDC(0.6) [Stankovi{\''{c}} et al., 2021]');
xlim([0,k_max]);
 

mean((MSEs-MSEs_av_mc).^2)



%%%% plotovanje poslednjeg grafika iz Automatica rada:
% V_rand_node=zeros(15,k,size(ALGS,2));
% for jj=1:size(ALGS,2)  
%   V_rand_node(:,:,jj)=F*w((1+(node-1)*n_features):(node*n_features),:,jj);  
% end
% 
% MSEs_node=squeeze(mean((V_rand_node-VV).^2,1));
% 
%  
% % 
% MD=30;
% plot(1:k, MSEs_node(1:k,1),'-d','MarkerIndices',1:MD:k)
% hold on;
% plot(1:k, MSEs_node(1:k,2),'-s','MarkerIndices',1:MD:k)
% plot(1:k, MSEs_node(1:k,3),'-o','MarkerIndices',1:MD:k)
% hold off;
% ylim([0,50]);
% xlabel('Iterations');
% ylabel('MSE');
% %legend;
% 
% legend('Sparse (neighbor-based) connectivity', 'No connectivity (single agent)', ...
%      'Maximal connectivity (fully connected network)');
