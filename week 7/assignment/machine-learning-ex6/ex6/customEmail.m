function z = customEmail(email_contents)
% gets an custom email from the email inbox and classify whether an email
% is spam or not?
% y is the string that would label the email
% email_contenst is the input email
clc;close all;
load('spamTrain.mat');
fprintf('\n classifu an email ..........\n');
word_indices = processEmail(email_contents);
x = emailFeatures(word_indices);
C = 0.1;
model = svmTrain(X,y,C,@linearKernel);
predict = svmPredict(model,x);
if predict == 0 
    z = ' This email is  not SPAM';
else 
    z = 'This is spam';
end
end

