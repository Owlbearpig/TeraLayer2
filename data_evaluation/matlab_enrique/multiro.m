function R= multiro(lam,p)

%%% destripa parametros
%l=size(p,2);
%ncapas=(l-4)/2;
%n=p(1:ncapas+2);
es=p;
%a=p(2*ncapas+3);
%thea=p(2*ncapas+4);
parametros
the(1)=thea;
%%%%%%%%%%

nc=sum(size(n))-3; % nc = 6 - 3 = 3
for h=1:sum(size(lam))-1
  for k=1:(nc+1)
    the(k+1)=asin((n(k)*sin(the(k)))/n(k+1));
    if a==1
          ra(k)=((n(k)*cos(the(k+1)))-((n(k+1))*cos(the(k))))/((n(k+1)*cos(the(k)))+(n(k)*cos(the(k+1))));
          rb(k)=((n(k+1)*cos(the(k)))-(n(k)*cos(the(k+1))))/((n(k)*cos(the(k+1)))+(n(k+1)*cos(the(k))));
          ta(k)=(2*n(k)*cos(the(k+1)))/((n(k+1)*cos(the(k)))+(n(k)*cos(the(k+1))));
          tb(k)=(2*n(k+1)*cos(the(k)))/((n(k)*cos(the(k+1)))+(n(k+1)*cos(the(k))));
    else
          ra(k)=((n(k)*cos(the(k)))-(n(k+1)*cos(the(k+1))))/((n(k)*cos(the(k)))+(n(k+1)*cos(the(k+1))));
          rb(k)=((n(k+1)*cos(the(k+1)))-(n(k)*cos(the(k))))/((n(k+1)*cos(the(k+1)))+(n(k)*cos(the(k))));
          ta(k)=(2*n(k)*cos(the(k)))/((n(k)*cos(the(k)))+(n(k+1)*cos(the(k+1))));
          tb(k)=(2*n(k+1)*cos(the(k+1)))/((n(k+1)*cos(the(k+1)))+(n(k)*cos(the(k))));
    end
  end
  M=(1/(tb(1)))*[(ta(1)*tb(1))-(ra(1)*rb(1)),rb(1);-ra(1),1];
  for s=1:nc
    fi(s)=(2*pi*n(s+1)*es(s))/lam(h);
    Q=(1/(tb(s+1)))*[(ta(s+1)*tb(s+1))-(ra(s+1)*rb(s+1)),rb(s+1);-ra(s+1),1];
    P=[exp(-fi(s)*1i),0;0,exp(fi(s)*1i)];
    M=Q*P*M;
  end
  tt=1/(M(2,2));
  rt=M(1,2)*tt;
  R(h)=rt*conj(rt);
end
  
R=R';
