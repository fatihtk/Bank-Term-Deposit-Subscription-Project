% Import Data
bank = readtable('bank-additional-full.csv');


% Check and convert Data types

bank.job = categorical(bank.job)
bank.marital = categorical(bank.marital)
bank.education = categorical(bank.education)
bank.default = categorical(bank.default)
bank.housing = categorical(bank.housing)
bank.loan = categorical(bank.loan)
bank.contact = categorical(bank.contact)
bank.month = categorical(bank.month)
bank.day_of_week = categorical(bank.day_of_week)
bank.campaign = categorical(bank.campaign)
bank.poutcome = categorical(bank.poutcome)
bank.y = categorical(bank.y)

% Changing Column Names
bank.Properties.VariableNames = {'age' 'job' 'marital' 'education', 'has_default_credit', 'has_housing_credit', 'has_personal_loan', 'contact_type', 'month', 'day', 'duration', 'campaign_contact_num', 'pdays', 'previous_contact_num', 'previous_campaign_outcome', 'employee_rate', 'price_idx', 'confidence_idx', 'mounth_rate', 'employee_num', 'subscribe'}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    ONE-HOT-ENCODING    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stack variables

idx = find(bank.job == 'unknown');
bank.job(idx) = 'unknown-job'

idx = find(bank.marital == 'unknown');
bank.marital(idx) = 'unknown-marital'

idx = find(bank.education == 'unknown');
bank.education(idx) = 'unknown-education'

idx = find(bank.has_default_credit == 'unknown');
bank.has_default_credit(idx) = 'unknown-has_default_credit'

idx = find(bank.has_housing_credit == 'unknown');
bank.has_housing_credit(idx) = 'unknown-has_housing_credit'

idx = find(bank.has_personal_loan == 'unknown');
bank.has_personal_loan(idx) = 'unknown-has_personal_loan'

idx = find(bank.contact_type == 'unknown');
bank.contact_type(idx) = 'unknown-contact_type'

idx = find(bank.month == 'unknown');
bank.month(idx) = 'unknown-month'

idx = find(bank.day == 'unknown');
bank.day(idx) = 'unknown-day'

idx = find(bank.campaign_contact_num == 'unknown');
bank.campaign_contact_num(idx) = 'unknown-campaign_contact_num'

idx = find(bank.previous_campaign_outcome == 'unknown');
bank.previous_campaign_outcome(idx) = 'unknown-previous_campaign_outcome'

idx = find(bank.subscribe == 'unknown');
bank.subscribe(idx) = 'unknown-subscribe'

idx = find(bank.has_default_credit == 'no');
bank.has_default_credit(idx) = 'no-credit'

idx = find(bank.has_default_credit == 'yes');
bank.has_default_credit(idx) = 'yes-credit'

idx = find(bank.has_housing_credit == 'no');
bank.has_housing_credit(idx) = 'no-housing-credit'

idx = find(bank.has_housing_credit == 'yes');
bank.has_housing_credit(idx) = 'yes-housing-credit'

idx = find(bank.has_personal_loan == 'no');
bank.has_personal_loan(idx) = 'no-personal_loan'

idx = find(bank.has_personal_loan == 'yes');
bank.has_personal_loan(idx) = 'yes-personal_loan'

idx = find(bank.subscribe == 'yes');
bank.subscribe(idx) = 'yes-subscribe'

idx = find(bank.subscribe == 'no');
bank.subscribe(idx) = 'no-subscribe'


Job = categorical(bank.job)
Marital= categorical(bank.marital)
Education= categorical(bank.education)
Has_default_credit= categorical(bank.has_default_credit)
Has_housing_credit= categorical(bank.has_housing_credit)
Has_personal_loan= categorical(bank.has_personal_loan)
Contact_type= categorical(bank.contact_type)
Month= categorical(bank.month)
Day= categorical(bank.day)
Campaign_contact_num= categorical(bank.campaign_contact_num)
Previous_campaign_outcome= categorical(bank.previous_campaign_outcome)
subscribe= categorical(bank.subscribe)


% bank_cat1 = table(Job,Marital,Education,Has_default_credit)
% bank_cat2 = table(Has_housing_credit,Has_personal_loan,Contact_type,Month)
% bank_cat3 = table(Day,Campaign_contact_num,Previous_campaign_outcome,subscribe)


% bank_cat_encode = table();
% 
% for i=1:width(bank_cat1)
%  bank_cat_encode = [bank_cat_encode onehotencode(bank_cat1(:,i))];
%  bank_cat_encode.unknown = [];
% end

% bank_cat_encode


Job = onehotencode(table(Job))
Marital= onehotencode(table(Marital))
Education= onehotencode(table(Education))
Has_default_credit= onehotencode(table(Has_default_credit))
Has_housing_credit= onehotencode(table(Has_housing_credit))
Has_personal_loan= onehotencode(table(Has_personal_loan))
Contact_type= onehotencode(table(Contact_type))
Month= onehotencode(table(Month))
Day= onehotencode(table(Day))
Campaign_contact_num= onehotencode(table(Campaign_contact_num))
Previous_campaign_outcome= onehotencode(table(Previous_campaign_outcome))
subscribe= onehotencode(table(subscribe))


Education.unknown= [];
Marital.unknown= [];
Job.unknown= [];
Has_personal_loan.unknown= [];
Has_housing_credit.unknown= [];
Has_default_credit.unknown= [];
subscribe.yes= [];
subscribe.no= [];
Has_personal_loan.yes= [];
Has_personal_loan.no= [];
Has_housing_credit.yes= [];
Has_housing_credit.no= [];
Has_default_credit.yes= [];
Has_default_credit.no= [];
encData = [Job Marital Education Has_housing_credit Has_default_credit Has_personal_loan Contact_type Month Day Campaign_contact_num Previous_campaign_outcome subscribe]

bank.job = [];
bank.marital = [];
bank.education = [];
bank.has_housing_credit = [];
bank.has_default_credit = [];
bank.has_personal_loan = [];
bank.contact_type = [];
bank.month = [];
bank.day = [];
bank.campaign_contact_num = [];
bank.previous_campaign_outcome= [];
bank.subscribe = [];

bank

Data = [encData bank]


% Splitting Data

% Split Data Test and Train
% Cross varidation (train: 70%, test: 30%)
cv = cvpartition(size(Data,1),'HoldOut',0.2);
idx = cv.test;

% Separate to training and test data
dataTrain = Data(~idx,:);
dataTest  = Data(idx,:);



% İdentify test and train datasets 
X_test = dataTrain(: ,102:103)
y_test = dataTest(: ,102:103)
dataTrain(:,102) = [];
dataTrain(:,103) = [];
dataTest(:,103) = [];
dataTest(:,103) = [];
X_train = dataTrain
y_train =dataTest

% making PCA
X_train = table2array(X_train);
X_test = table2array(X_test);
y_train = table2array(y_train);
y_test = table2array(y_test);


[coeff,scoreTrain,~,~,explained,mu] = pca(X_train);

explained



sum_explained = 0;
idx = 0;
while sum_explained < 95
    idx = idx + 1;
    sum_explained = sum_explained + explained(idx);
end
idx


% De-mean (MATLAB will de-mean inside of PCA, but I want the de-meaned values later)
X_train = X_train - mean(X_train); % Use X = bsxfun(@minus,X,mean(X)) if you have an older version of MATLAB
% Do the PCA
[coeff,score,latent,~,explained] = pca(X_train);
% Calculate eigenvalues and eigenvectors of the covariance matrix
covarianceMatrix = cov(X_train);
[V,D] = eig(covarianceMatrix);
% "coeff" are the principal component vectors. These are the eigenvectors of the covariance matrix. Compare ...
coeff
V
% Multiply the original data by the principal component vectors to get the projections of the original data on the
% principal component vector space. This is also the output "score". Compare ...
dataInPrincipalComponentSpace = X_train*coeff
score
% The columns of X*coeff are orthogonal to each other. This is shown with ...
corrcoef(dataInPrincipalComponentSpace)
% The variances of these vectors are the eigenvalues of the covariance matrix, and are also the output "latent". Compare
% these three outputs
var(dataInPrincipalComponentSpace)'
latent
sort(diag(D),'descend')


% Scree plot
figure
h = plot(explained,'.-');
set(h,'LineWidth',3,'MarkerSize',36)
ylim([0 100])
set(gca,'XTick',1:110)
title('Explained variance by principal components')
xlabel('Principal component number')
ylabel('Fraction of variation explained [%]')

% trainedClassifier = fitcnb(X_train, X_test);
% partitionedModel = crossval(trainedClassifier, 'KFold', 10);
% accuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');


% convert one hot encoded labels to string


x_test = X_test(:,1)
x_test = table(x_test)
x_test=categorical(x_test.x_test)
idx1 = find(x_test=='0');
idx2 = find(x_test=='1');

x_test2 = X_test(:,1)
x_test2 = table(x_test2)
x_test2.x_test2 = categorical(x_test2.x_test2)
x_test2.x_test2(idx1) = 'no'
x_test2.x_test2(idx2) = 'yes'


y_test3 = y_test(:,1)
y_test3 = table(y_test3)
y_test3=categorical(y_test3.y_test3)
idx1 = find(y_test3=='0');
idx2 = find(y_test3=='1');

y_test4 = y_test(:,1)
y_test4 = table(y_test4)
y_test4.y_test4 = categorical(y_test4.y_test4)
y_test4.y_test4(idx1) = 'no'
y_test4.y_test4(idx2) = 'yes'

y_test4 = categorical(y_test4.y_test4)

% Random Forest

x_test2 = tale2array(x_test2)
rng(1); % For reproducibility
Mdl = TreeBagger(50,X_train,x_test2,'OOBPrediction','On',...
    'Method','classification')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
view(Mdl.Trees{1},'Mode','graph')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
oobErrorBaggedEnsemble = oobError(Mdl);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng('default') 
tallrng('default')
tMdl = TreeBagger(20,X_train,x_test2,'Prior','Uniform')

terr = error(tMdl,X_train,x_test2)

avg_terr = mean(terr)






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sum_explained = 0;
idx = 0;
while sum_explained < 95
    idx = idx + 1;
    sum_explained = sum_explained + explained(idx);
end
idx
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
scoreTrain95 = scoreTrain(:,1:idx);
mdl = fitctree(scoreTrain95,x_test2);
scoreTest95 = (y_train-mu)*coeff(:,1:idx);
y_test_predicted = predict(mdl,scoreTest95);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y_test_predicted
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classnames = {'no','yes'}
rng default
Mdl = fitcnb(scoreTrain95,x_test2,'ClassNames',classnames,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

Mdl1 = fitcnb(scoreTrain95,x_test2,...
    'ClassNames',{'no','yes'})
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mdl1.DistributionParameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mdl1.DistributionParameters{1,2}
%%%%%%%%%%%%%%%%%%%%%%%%%%%
isLabels1 = resubPredict(Mdl1);
x_test3 =cellstr(x_test2);
ConfusionMat1 = confusionchart(x_test3,isLabels1);

%%%%%%%%%%%%%%%%%%%%%%%
Mdl2 = fitcnb(scoreTrain95,x_test2,...
    'DistributionNames',{'normal','normal'},...
    'ClassNames',{'no','yes'});
Mdl2.DistributionParameters{1,2}
%%%%%%%%%%%%%%%%
isLabels2 = resubPredict(Mdl2);
ConfusionMat2 = confusionchart(x_test3,isLabels2);

%%%%%%%%%%%%%%%
CVMdl1 = fitcnb(scoreTrain95,x_test2,...
    'ClassNames',{'no','yes'},...
    'CrossVal','on');

t = templateNaiveBayes();
CVMdl2 = fitcecoc(scoreTrain95,x_test2,'CrossVal','on','Learners',t);

classErr1 = kfoldLoss(CVMdl1,'LossFun','ClassifErr')
classErr2 = kfoldLoss(CVMdl2,'LossFun','ClassifErr')  % Mdl2 has a lower generalization error.

resp = strcmp(x_test2,'yes');

[~,score] = resubPredict(mdl);
diffscore = score(:,2) - max(score(:,1),score(:,2));
[X,Y,T,~,OPTROCPT,suby,subnames] = perfcurve(x_test2,diffscore,'yes');

OPTROCPT

suby

subnames

plot(X,Y)
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for Classification by Classification Trees')
hold off

T((X==OPTROCPT(1))&(Y==OPTROCPT(2)))

figure, plot(X,Y)
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for Classification by Classification Trees')
hold off

tc = fitctree(scoreTrain95,x_test2)

rng(1); % For reproducibility
MdlDefault = fitctree(scoreTrain95,x_test2,'CrossVal','on');

numBranches = @(x)sum(x.IsBranch);
mdlDefaultNumSplits = cellfun(numBranches, MdlDefault.Trained);

figure;
histogram(mdlDefaultNumSplits)

view(MdlDefault.Trained{1},'Mode','graph')

Mdl7 = fitctree(scoreTrain95,x_test2,'MaxNumSplits',7,'CrossVal','on');
view(Mdl7.Trained{1},'Mode','graph')

classErrorDefault = kfoldLoss(MdlDefault)

classError7 = kfoldLoss(Mdl7)

%%Optimize Classification Tree

Mdl = fitctree(scoreTrain95,x_test2,'OptimizeHyperparameters','auto')

%%Predictor Importance Estimates

imp = predictorImportance(Mdl);

figure;
bar(imp);
title('Predictor Importance Estimates');
ylabel('Estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel = Mdl.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';