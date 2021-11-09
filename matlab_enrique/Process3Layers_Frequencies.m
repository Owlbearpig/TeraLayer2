clear
clc

r=dlmread('ref_1000x.csv', ',', 1, 0);
b=dlmread('BG_1000.csv', ',', 1, 0);

s=dlmread('Kopf_1x_0001', ',', 1, 0);
s2=dlmread('Kopf_1x_0002', ',', 1, 0);
%s3=dlmread('Kopf_1x_0003', ',', 1, 0);
%s=(s1+s2+s3)/3;
f=r(235:end-1,1)*1e6;
lam=3e8./f;

rr=r(235:end-1,2)-b(235:end-1,2);
ss=s(235:end-1,2)-b(235:end-1,2);
T=ss./rr;
R=T.^2;

ss2=s2(235:end-1,2)-b(235:end-1,2);
T2=ss2./rr;
R2=T2.^2;


ni=400;
nf=600;
nn=40;
enes=[ni:nn:nf];

for k=1:200
    enes=floor(100*rand(6,1)+450);

        di=[0.00006 0.00060 0.00006];
        lb=[0.000001 0.00001 0.000001];
        hb=[0.001   0.001   0.001];
        %options = optimset('TolFun',1e-25,'MaxFunEvals',1e8,'MaxIter',1e4,'TolX',1e-15);
        options=optimoptions(@lsqcurvefit,'Algorithm','levenberg-marquardt', 'MaxFunctionEvaluations',2000000, 'MaxIterations', 200000, 'StepTolerance',1e-15)
        
        [d,resnorm] = lsqcurvefit(@multir,di,lam(enes,1),R(enes,1),lb,hb,options);
        ds(k,:)=d;
        
        [d,resnorm] = lsqcurvefit(@multir,di,lam(enes,1),R2(enes,1),lb,hb,options);
        ds2(k,:)=d;
        

end

niceplotpapernature
% plot(lam/1e-3,R(:,1),'color',[0.9 0.9 0.9])
% hold on
% plot(lam(enes,1)/1e-3,R(enes,1),'o')
% plot(lam/1e-3,multir(d,lam(:,1)),'color',c_azul)
% %semilogy(b(1:end-1,1)/1e6,b(1:end-1,2))
% %semilogy(s(1:end-1,1)/1e6,ss(1:end-1,2))
% parametros
% text(1.5,1.0,['d1 = ' num2str(d(1)*1e6) '{\mu}m, n = ' num2str(n(2))],'HorizontalAlignment','center')
% text(1.5,0.9,['d2 = ' num2str(d(2)*1e6) '{\mu}m, n = ' num2str(n(3))],'HorizontalAlignment','center')
% text(1.5,0.8,['d3 = ' num2str(d(3)*1e6) '{\mu}m, n = ' num2str(n(4))],'HorizontalAlignment','center')
% axis([0 2 0 1.1])
% xlabel('THZ-Wavelenght (mm)')
% ylabel('r^2 (arb. units)')

plot([1:k]',ds)
hold on
plot([1:k]',ds2,'--')
