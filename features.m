function [ out ] = features(state)

 % out=exp([ -(state-1)^2 ; -(state-5)^2 ; -(state-10)^2 ; -(state-14)^2]./100); 
out=exp([ -(state-1)^2 ; -(state-3)^2 ;-(state-5)^2 ;-(state-7)^2 ;-(state-9)^2 ; -(state-11)^2 ; -(state-13)^2]./4); 
%   out=zeros(15,1);
%   out(state)=1;
end

