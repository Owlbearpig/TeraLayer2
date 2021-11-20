clear
clc

addpath('E:/Projects/TeraLayer/matlab_enrique')

r=dlmread('Data/ref_1000x.csv', ',', 1, 0);
b=dlmread('Data/BG_1000.csv', ',', 1, 0);

s=dlmread('Data/Kopf_1x/Kopf_1x_0001', ',', 1, 0);

f=r(235:end-1,1)*1e6;
lam=3e8./f;

rr=r(235:end-1,2)-b(235:end-1,2);
ss=s(235:end-1,2)-b(235:end-1,2);
T=ss./rr;

R=T.^2;

ni=400;
nf=600;
nn=40;
enes=[ni:nn:nf];

d = [0.0000378283 0.0006273254 0.0000378208];

repeats = 100;
%t0 = clock();
t = cputime;
for _=1:repeats
  multir(d,lam(enes,1));
end
elapsed_time = cputime-t;
%elapsed_time = etime(clock(), t0);
disp([num2str(1000.*elapsed_time./repeats), ' ms'])

%{
plot(lam/1e-3, R(:,1),'color',[0.9 0.9 0.9])
hold on
plot(lam(enes,1)/1e-3, R(enes,1),'o')
plot(lam/1e-3, multir(d,lam(:,1)),'color',	[0 0 1])

parametros
text(1.5,1.0,['d1 = ' num2str(d(1)*1e6) '{\mu}m, n = ' num2str(n(2))],'HorizontalAlignment','center')
text(1.5,0.9,['d2 = ' num2str(d(2)*1e6) '{\mu}m, n = ' num2str(n(3))],'HorizontalAlignment','center')
text(1.5,0.8,['d3 = ' num2str(d(3)*1e6) '{\mu}m, n = ' num2str(n(4))],'HorizontalAlignment','center')
axis([0 2 0 1.1])
xlabel('THZ-Wavelenght (mm)')
ylabel('r^2 (arb. units)')
}%
