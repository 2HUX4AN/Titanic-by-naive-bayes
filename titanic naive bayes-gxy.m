%Load the Titanic data
Train= readtable('titanic\train.csv');
Test = readtable('titanic\test.csv');


% Using the average age replace NaN age
avgAge = nanmean(Train.Age)             % get average age
Train.Age(isnan(Train.Age)) = avgAge;   % replace NaN with the average
Test.Age(isnan(Test.Age)) = avgAge; 

%preprocess

% Del some columns
Train(:,{'Name','Ticket','Fare','Cabin','Embarked'}) = [];
Test(:,{'Name','Ticket','Fare','Cabin','Embarked'}) = [];
%head(Train)

% change Sex to tpye "double"  male-1    female-0
for i = 1 : 891
    if strcmp(Train.Sex{i} ,'male')
        Train.Sex{i}=1;
    else
        Train.Sex{i}=0;
    end
end
 
for i = 1 : 418
    if strcmp(Test.Sex{i} ,'male')
        Test.Sex{i}=1;
    else
        Test.Sex{i}=0;
    end
end
Train.Sex = cell2mat(Train.Sex);
Test.Sex = cell2mat(Test.Sex);

% age 6岁分一档   0-6为1 6-12为2 ... 
	Train_array = table2array(Train);
for i = 1 : 891
    Train_array(i,5)=ceil(Train_array(i,5)/6);
end
 
	Test_array = table2array(Test);
for i = 1 : 418
    Test_array(i,4)=ceil(Test_array(i,4)/6);
end


%计算先验概率
sum_survived=0
for i = 1 : 891
	sum_survived = sum_survived+sum(Train.Survived(i)==1)
end
p_prior_survived = sum_survived/891
%disp(p_prior_survived)


%计算条件概率 Conditional probability

for i = 3:7
	features = unique(Train_array(:,i));    %该特征有几种
	%numclass = height(features);
	%features = table2array(features);
	numclass = length(features);
	%Train_array = table2array(Train);
	for j = 1:numclass
		temp_1=0;
		temp_0=0;
		for k = 1:891
			if Train.Survived(k)==1
				temp_1= temp_1+sum(Train_array(k,i)==features(j,1));
			else
				temp_0= temp_0+sum(Train_array(k,i)==features(j,1));
			end
		end
		%+1避免概率为0
		p_Conditional(j,1+2*(i-3))=(temp_1+1)/(sum_survived+numclass);	%计算存活条件概率		
		p_Conditional(j,2+2*(i-3))=(temp_0+1)/(891-sum_survived+numclass);	%计算死亡条件概率
	end  
end

%计算后验概率
for k = 1:418
	for i = 2:6
		features = unique(Train_array(:,i+1));
		numclass = length(features);
		for j = 1:numclass
			if  Test_array(k,i)==features(j,1)
				p__survived(k,i-1)=p_Conditional(j,1+2*(i-2));
				p__unsurvived(k,i-1)=p_Conditional(j,2+2*(i-2));
			end
		end
		%避免条件概率为0
		if p__survived(k,i-1)==0
			p__survived(k,i-1)=1;
		end
		if p__unsurvived(k,i-1)==0
			p__unsurvived(k,i-1)=1;
		end
	end
end

for k = 1:418
	p__survived(k,6)=1;
	p__unsurvived(k,6)=1;
	for i =1:5
		p__survived(k,6)=p__survived(k,6)*p__survived(k,i);
		p__unsurvived(k,6)=p__unsurvived(k,6)*p__unsurvived(k,i);
	end
	p_posterior_survived(k,1)=p_prior_survived*p__survived(k,6);	%计算存活条件概率		
	p_posterior_unsurvived(k,1)=(1-p_prior_survived)*p__unsurvived(k,6);	%计算死亡条件概率
	if p_posterior_survived(k,1)>=p_posterior_unsurvived(k,1)
		test_predictions(k,1)=1;
	else
		test_predictions(k,1)=0;
	end
end




%{
%Pclass

for j = 1:3
	temp_1=0;
	temp_0=0;
	for k = 1:891
		if Train.Survived(k)==1
			temp_1= temp_1+sum(Train.Pclass(k)==j);
		else
			temp_0= temp_0+sum(Train.Pclass(k)==j);
		end
	end
	p_Conditional_Pclass(j,1)=temp_1/sum_survived;
	p_Conditional_Pclass(j,2)=temp_0/(891-sum_survived);
end  
%sex
%age
%sibsp
%Parch
%}
