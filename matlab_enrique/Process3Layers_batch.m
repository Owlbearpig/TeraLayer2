clear
clc
pkg load optim
addpath ('E:/Projects/TeraLayer/matlab_enrique')

r=dlmread('Data/ref_1000x.csv', ',', 1, 0);
b=dlmread('Data/BG_1000.csv', ',', 1, 0);

f=r(235:end-1,1)*1e6;
lam=3e8./f;
rr=r(235:end-1,2)-b(235:end-1,2);
cont=0;
cont2=0;
enes=floor(200*rand(6,1)+400);

for k=0:10
    disp(k)
    s=dlmread(['Data/Kopf_1x/Kopf_1x_0' num2str(k,'%03.f')], ',', 1, 0);

    ss=s(235:end-1,2)-b(235:end-1,2);
    T=ss./rr;

    R=T.^2;

    ni=400;
    nf=600;
    nn=40;

        di=[0.000045 0.00060 0.000045];
        lb=[0.000001 0.00001 0.000001];
        hb=[0.001   0.001   0.001];
        %options = optimset('TolFun',1e-25,'MaxFunEvals',1e8,'MaxIter',1e4,'TolX',1e-15);
        %options=optimoptions(@lsqcurvefit,'Algorithm','levenberg-marquardt', 'MaxFunctionEvaluations',2000000, 'MaxIterations', 200000, 'StepTolerance',1e-15)
        %options=optimset('Algorithm','levenberg-marquardt', 'MaxFunEvals',2000000, 'MaxIter', 200000, 'TolX',1e-15)
        options.bounds=[lb',hb'];
        %options.fract_prec = 1e-15 * ones(size (di))';
        %wt = ones(size(ni:nn:nf)).*(sqrt(@multiro(lam(ni:nn:nf,1), di)).^(-1))'
        %[d,resnorm] = lsqcurvefit(@multir,di,lam(ni:nn:nf,1),R(ni:nn:nf,1),lb,hb,options);
        dp = 0.00000001 * ones(size (di));
        %[f, d, cvg, iter, corp, covp, covr, stdresid, Z, r2] = leasqr(lam(ni:nn:nf,1), R(ni:nn:nf,1), di, @multiro, 1e-15, 200000, [], dp, [], options);
        [f, d, cvg, iter, corp, covp, covr, stdresid, Z, r2] = leasqr(lam(ni:nn:nf,1), R(ni:nn:nf,1), di, @multiro, [], [], [], [], [], options);
        %settings = optimset('bounds', [lb',hb'])
        %disp(f, @multiro(lam(ni:nn:nf,1), d), iter)
        disp(abs(f-@multiro(lam(ni:nn:nf,1), d)))
        ds(k+1,:)=d;
        %disp(@multiro(lam(ni:nn:nf,1), d))
%         if(abs(ds(k+1,1)-45e-6)>=6e-6)
%             di=[0.000040 0.00060 0.000040];
%             lb=[0.000001 0.00001 0.000001];
%             hb=[0.001   0.001   0.001];
%             %options = optimset('TolFun',1e-25,'MaxFunEvals',1e8,'MaxIter',1e4,'TolX',1e-15);
%             options=optimoptions(@lsqcurvefit,'Algorithm','levenberg-marquardt', 'MaxFunctionEvaluations',2000000, 'MaxIterations', 200000, 'StepTolerance',1e-15)
%             [d,resnorm] = lsqcurvefit(@multir,di,lam(ni:nn:nf,1),R(ni:nn:nf,1),lb,hb,options);
%             ds(k+1,:)=d;
%         end
%         
%         if(abs(ds(k+1,1)-45e-6)>=6e-6)
%             di=[0.000030 0.00060 0.000030];
%             lb=[0.000001 0.00001 0.000001];
%             hb=[0.001   0.001   0.001];
%             %options = optimset('TolFun',1e-25,'MaxFunEvals',1e8,'MaxIter',1e4,'TolX',1e-15);
%             options=optimoptions(@lsqcurvefit,'Algorithm','levenberg-marquardt', 'MaxFunctionEvaluations',2000000, 'MaxIterations', 200000, 'StepTolerance',1e-15)
%             [d,resnorm] = lsqcurvefit(@multir,di,lam(ni:nn:nf,1),R(ni:nn:nf,1),lb,hb,options);
%             ds(k+1,:)=d;
%         end

        if false%(abs(ds(k+1,1)-45e-6)>=6e-6)
            disp('no convergio!!!')
            cont2=cont2+1;
            chafa(:,cont2)=R(:,1);
        else
            cont=cont+1;
            dsg(cont,:)=d;
        end       

end

sds=std(ds);
mds=mean(ds);

sdsg=std(dsg);
mdsg=mean(dsg);


%niceplotpapernature
plot([0:k]',ds*1e6)
xlabel('Measurement number')
ylabel('Thickness extracted ({\mu}m)')
text(25,110,['d1 = ' num2str(mds(1)*1e6) ' \pm ' num2str(sds(1)*1e6) '{\mu}m'],'color',[0 0 1],'HorizontalAlignment','center');
text(50,550,['d2 = ' num2str(mds(2)*1e6) ' \pm ' num2str(sds(2)*1e6) '{\mu}m'],'color',[1 0 0],'HorizontalAlignment','center');
text(75,110,['d3 = ' num2str(mds(3)*1e6) ' \pm ' num2str(sds(3)*1e6) '{\mu}m'],'color',	[1 1 0],'HorizontalAlignment','center');

    
% niceplotpapernature
% plot(lam/1e-3,R(:,1),'color',[0.9 0.9 0.9])
% hold on
% plot(lam(ni:nn:nf,1)/1e-3,R(ni:nn:nf,1),'o')
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
