%压缩感知：
%对于原始高维信号s，观测矩阵Y将其投影至低维空间（即亚采样）:Y*s
%原始信号s需满足稀疏性才可还原，因此将s在稀疏基矩阵Z上表示为稀疏信号s=Zx,x为稀疏稀疏
%因此压缩过程为YZx，令YZ=A,即Ax；
%令观测结果为b，因此还原原始信号<=>优化|Ax-b|^2，由于x为稀疏向量，因此加入一范数惩罚(Lasso)

%数据
m=128;n=256;
A=randn(m,n);%传感矩阵：
u=sprandn(n,1,0.1);
zs=0.01;%噪声水平
b=A*u+zs*randn(m,1);%采样结果

%优化目标
epsilon=0*ones(n,1);
p=0.2;
F_0=@(x) ((A*x-b)'*(A*x-b))/(b'*b);%得到观测结果相对误差
F=@(x) 0.5*(A*x-b)'*(A*x-b) + p*norm((x.^2+epsilon).^0.5,1);%Lasso
dF=@(x) A'*(A*x-b) + p*x./((x.^2+epsilon).^0.5);%梯度
dF_2=@(x) A'*A + p*diag(epsilon./((x.^2+epsilon).^1.5));%二阶梯度

%---------------------优化---------------------%
x_0=randn(n,1);
e=1e-1;%迭代停止时梯度平均值临界值

%Armijo
a_0=1;%步长初值
r=0.8;%步长回退率
c1=0.1;

%---------梯度下降---------%
% step_G=0;
% x_G=x_0;
% t_G=0;
% tic;
% while(sum(abs(dF(x_G)))/n>=e && step_G<=3000)
%     a=a_0;
%     k=0;
%     while(F(x_G-a*dF(x_G))>F(x_G)-c1*a*(dF(x_G)'*dF(x_G)) && k<=30)
%         a=a*r;
%         k=k+1;
%     end
%     step_G=step_G+1;
%     x_G=x_G-a*dF(x_G);%更新x
%     GD=F(x_G);%优化目标值
%     scatter(step_G,GD,5,'green','filled');
%     hold on;
%     d_GD=dF(x_G);%优化目标梯度值
%     GD_0=F_0(x_G);%(Ax-b)二范数值
% end
% t_G=t_G+toc;


%------------BB------------%
step_B=0;
x_B_0=x_0;
a=a_0;
t_B=0;
tic;
while(sum(abs(dF(x_B_0)))/n>=e && step_B<=3000)  
    k=0;
    while(F(x_B_0-a*dF(x_B_0))>F(x_B_0)-c1*a*(dF(x_B_0)'*dF(x_B_0)) && k<=30)
        a=a*r;
        k=k+1;
    end
    step_B=step_B+1;
    x_B_1=x_B_0-a*dF(x_B_0);
    B=F(x_B_1);%优化目标值
    scatter(step_B,B,5,'blue','filled');
    hold on;
    d_B=dF(x_B_1);%优化目标梯度值
    B_0=F_0(x_B_1);%(Ax-b)二范数值
    
    a=((x_B_1-x_B_0)' * (x_B_1-x_B_0)) / ((x_B_1-x_B_0)' * (dF(x_B_1)-dF(x_B_0)));
    x_B_0=x_B_1;
end
t_B=t_B+toc;
B_F=norm(x_B_0,1)/norm(u,1);%x_n一范数（稀疏度）


%---------牛顿法---------%
% step_N=0;
% x_N=x_0;
% t_N=0;
% tic;
% while(sum(abs(dF(x_N)))/n>=e && step_N<=3000)  
%     step_N=step_N+1;
%     t=-min(eig(dF_2(x_N)))+10;
%     G=dF_2(x_N)+t*eye(n);%修正Hessen矩阵
%     D=-G^(-1)*dF(x_N);%下降方向
%     
%     a=a_0;
%     k=0;
%     while(F(x_N+a*D)>F(x_N)+c1*a*(dF(x_N)'*D) && k<=30)
%         a=a*r;
%         k=k+1;
%     end
%     
%     x_N=x_N+a*D;%更新x
%     N=F(x_N);%优化目标值
%     scatter(step_N,N,5,'red','filled');
%     hold on;
%     d_N=dF(x_N);%优化目标梯度值
%     N_0=F_0(x_N);%(Ax-b)二范数值
% end
% t_N=t_N+toc;
% N_F=norm(x_N,1)/norm(u,1);%x_n一范数（稀疏度）
% err=((x_N-u)'*(x_N-u))/(u'*u);







