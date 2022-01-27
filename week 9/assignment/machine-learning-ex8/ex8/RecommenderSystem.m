function  RecommenderSystem()
% build a recommender system for given user i and for movie j
fprintf('\n building a Recommender system \n\n');
movieList = loadMovieList();
my_ratings = zeros(size(movieList,1),1);
my_ratings(1) =5;
my_ratings(73) =4;
my_ratings(44) = 3;
my_ratings(99) =4;
my_ratings(200) =5;
my_ratings(633) =5;
my_ratings(674) =1;
my_ratings(end) =5;
lambda =1.5;
fprintf('\n my ratings \n');
for j = 1: length(my_ratings)
    if my_ratings(j)>0
        fprintf('\n Rated %d for %s\n', my_ratings(j),...
            movieList{j});
    end
end
% training collabrative filtering
fprintf('\n trainibng collabrative filtering \n\n');
load('ex8_movies.mat');
Y = [ my_ratings Y];
R = [(my_ratings ~=0) R];
[Ynorm, Ymean] = normalizeRatings(Y, R);
users = size(Y,2);
movies = size(Y,1);
features = 10;


% learning features 
fprintf('\n learning features\n ');
X = randn(movies,features);
Theta = randn(users,features);
initial_perms = [X(:) ; Theta(:)];

options = optimset('GradObj','on','MaxIter',500);
perms = fmincg(@(t) (cofiCostFunc(t,Ynorm,R,users,movies,features,lambda)),initial_perms,options);

X = reshape(perms(1:movies*features), movies ,features);
Theta = reshape(perms((movies*features)+1 : end), users, features);

fprintf('\n Recommender system learning completed \n\n');

% using prediction to predict ratings

p = X * Theta';
my_predictions = p(:,1) + Ymean;
movieList = loadMovieList();
[r, ix] = sort(my_predictions, 'descend');
fprintf('\nTop recommendations for you:\n');
for i = 1:10
    j = ix(i);
    fprintf('\nPredicted rating %d for movie %s\n', my_predictions(j),...
        movieList{j});
end
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end
end






        



        
        
        
        
        
        
        
        

        




