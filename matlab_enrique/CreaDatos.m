clear all
clc
addpath ('E:/Projects/TeraLayer/matlab_enrique')

for w=1:10
for q=1:100
% *** Construct THz wavelenghts ***
    lopt=[800 800.1 800.3 800.7]*1e-9; % Optical Wavelenghts
    fabs=3e8./lopt; % Optical Frequencies
    m=size(fabs,2);
    cont=0;
    for k=1:m-1
        for l=k+1:m
            cont=cont+1;
            dif(cont)=abs(fabs(k)-fabs(l));
        end
    end
    f2=[min(dif):10e9:max(dif)];
    lthz=3e8./dif;


    
lthz2=3e8./f2; % *** NOT IMPORTANT!

% *** Create "experimental" data with noise
    d0=[0.001 0.0003]; % Thickness to be simulated
    r=multir(d0,lthz); % Calculate reflectance (E)
    rr=r+r.*randn(1,cont)/(10*w); % Add amplitude noise
    ewl(w)=1./(10*w);


% *** WORK OUT Thickness, THIS IS THE IMPORTANT PART!!! ***
    di=[0.00085 0.00032];
    lb=[0.0001 0.0001];
    hb=[0.003 0.003];
    %options = optimset('TolFun',1e-25,'MaxFunEvals',1e8,'MaxIter',1e4,'TolX',1e-15);
    options=optimoptions(@lsqcurvefit,'Algorithm','levenberg-marquardt', 'MaxFunctionEvaluations',2000000, 'MaxIterations', 200000, 'StepTolerance',1e-15)
    [d,resnorm] = lsqcurvefit(@multir,di,lthz,rr,lb,hb,options);

%lkasjdas
out(q,:)=[di d d0 (d0-d)]*1000;
end    
delta(w,:)=std(out);
salida(:,:,w)=out;
ddl(w,:)=sum(abs(out(:,7:8)))/q;
ev10pc(:,:,w)=abs(out(:,7:8))>0.01;
ct10pc(w,:)=sum(abs(out(:,7:8))>0.01);
end

%nicepolotpapernature
niceplotpapernature

subplot(2,2,1)
plot(ewl*100,ddl(:,1))
xlabel('Relative error of R (%)')
ylabel('Thickness error (mm)')
axis([0 11 0 0.02])

subplot(2,2,2)
plot(ewl*100,ddl(:,2))
xlabel('Relative error of R (%)')
ylabel('Thickness error (mm)')
axis([0 11 0 0.02])

subplot(2,2,3)
plot(ewl*100,100*ct10pc(:,1)/q)
xlabel('Relative error of R (%)')
ylabel('Error>10{\mu}m (%)')
%axis([0 11 0 0.02])

subplot(2,2,4)
plot(ewl*100,100*ct10pc(:,1)/q)
xlabel('Relative error of R (%)')
ylabel('Error>10{\mu}m (%)')
%axis([0 11 0 0.02])

% disp(['Guess: ' num2str(di*1000) 'mm'])
% disp(['Final: ' num2str(d*1000) 'mm'])
% disp(['Real: ' num2str(d0*1000) 'mm'])
% disp(['Error: ' num2str((d0-d)*1000) 'mm'])

% subplot(1,2,1)
% plot(out(:,3))
% hold on
% plot(out(:,5))
% axis([1 100 0 1.5])
% 
% subplot(1,2,2)
% plot(out(:,4))
% hold on
% plot(out(:,6))
% axis([1 100 0 1.5])



% r2=sqrt(multir(d,lthz2));
% 
% plot(dif,r,'o')
% hold on
% plot(f2,r2)

