%Fitting

% Create experimental data
x=[0:1:10].T; # originally transposed in matlab
y=5+5*x+.2*randn(size(x,1),1);

% Input parameters
lb=[4 4];
ub=[6 6];
max_its=4;
refine=100;


its=0;
a_m=lb(1);
a_M=ub(1);
b_m=lb(2);
b_M=ub(2);

as=([0:refine-1]/(refine-1))*(a_M-a_m) + a_m;
bs=([0:refine-1]/(refine-1))*(b_M-b_m) + b_m;
minerr=1e100;
vals=0;

while(its<=max_its) %this has sligtly differnt sintax in C
    its=its+1;
    for j=1:refine %this has sligtly differnt sintax in C
        for k=1:refine %this has sligtly differnt sintax in C
            yy=as(j) + bs(k) * x; %this line will need extra C programming
            err=sum((y-yy).^2); %this line will need extra C programming
            vals=vals+1;

            if(err<minerr)
                i_a=j;
                i_b=k;
                minerr=err;
            end
        end
    end
    if(i_a==1)
        as=([0:refine-1]/(refine-1))*(as(i_a+1) - as(i_a)) + as(i_a);
    else
        if(i_a==refine)
            as=([0:refine-1]/(refine-1))*(as(i_a) - as(i_a-1)) + as(i_a-1);
        else
            as=([0:refine-1]/(refine-1))*(as(i_a+1) - as(i_a-1)) + as(i_a-1);
        end
    end

    if(i_b==1)
        bs=([0:refine-1]/(refine-1))*(bs(i_b+1) - bs(i_b)) + bs(i_b);
    else
        if(i_b==refine)
            bs=([0:refine-1]/(refine-1))*(bs(i_b) - bs(i_b-1)) + bs(i_b-1);
        else
            bs=([0:refine-1]/(refine-1))*(bs(i_b+1) - bs(i_b-1)) + bs(i_b-1);
        end
    end
    p=[as(i_a) bs(i_b)]

    ps(its,:)=p;
end


plot(x,y)
hold on
plot(x,p(1) + p(2)*x)