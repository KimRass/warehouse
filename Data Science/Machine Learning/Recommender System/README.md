# Types of Data
- Source: https://blog.mirumee.com/the-difference-between-implicit-and-explicit-data-for-business-351f70ff3fbf
## Explicit Data
- *Explicit data is much harder to collect.* Let’s look at Spotify. Simply listening to a song is not explicit data in itself. The system does not know for sure that the user likes that song. Actual explicit data is when the user adds a specific tune to a playlist or hits the heart icon to say that they enjoy listening to it. In such cases, there is exponentially more implicit than explicit data being created by user activity.
- *Explicit data can also be shallow.( Users may be asked to give binary reactions: like or dislike, thumbs up or thumbs down. Even when a site like IMDB allows for ratings from 1 to 10, human nature means that people tend to rate in the extremes. Users regularly rate everything as 10 or 1; not many people take the time to leave a 4-out-of-10 rating because they clearly didn’t have a strong opinion in the first place.
## Implicit Data
- *With implicit data, we sometimes need to observe what the user does next.* If someone listens to a single song, we cannot know if they liked that artist. The system needs to store that information and see what happens in future. If the user then purchases an album a few days later, that second action backs up the initial assumption. The system can then learn how that single user or all users interact with the system and make better assumptions as time passes and more data is generated.

# Datasets
## MovieLens
## Last.fm

# Cold Start
- Source: https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)
- *It concerns the issue that the system cannot draw any inferences for users or items about which it has not yet gathered sufficient information.*
- The cold start problem is a well known and well researched problem for recommender systems. *Typically, a recommender system compares the user's profile to some reference characteristics. These characteristics may be related to item characteristics (content-based filtering) or the user's social environment and past behavior (collaborative filtering). Depending on the system, the user can be associated to various kinds of interactions: ratings, bookmarks, purchases, likes, number of page visits etc.*
- Cases of cold start
	- New item
		- The item cold-start problem refers to when items added to the catalogue have either none or very little interactions. ***This constitutes a problem mainly for collaborative filtering algorithms due to the fact that they rely on the item's interactions to make recommendations. If no interactions are available then a pure collaborative algorithm cannot recommend the item. In case only a few interactions are available, although a collaborative algorithm will be able to recommend it, the quality of those recommendations will be poor. This arises another issue, which is not anymore related to new items, but rather to unpopular items. In some cases (e.g. movie recommendations) it might happen that a handful of items receive an extremely high number of interactions, while most of the items only receive a fraction of them. This is referred to as popularity bias.*** (e.g., Number of user interactions associated to each item in a Movielens dataset. Few items have a very high number of interactions, more than 5000, while most of the others have less than 100.)
		- In the context of cold-start items the popularity bias is important because it might happen that many items, even if they have been in the catalogue for months, received only a few interactions. ***This creates a negative loop in which unpopular items will be poorly recommended, therefore will receive much less visibility than popular ones, and will struggle to receive interactions.*** While it is expected that some items will be less popular than others, this issue specifically refers to the fact that the recommender has not enough collaborative information to recommend them in a meaningful and reliable way.
	- New user
		- A new user registers and has not provided any interaction yet, therefore it is not possible to provide personalized recommendations.
	- New community
		- The new community problem, or systemic bootstrapping, refers to the startup of the system, when virtually no information the recommender can rely upon is present. This case presents the disadvantages of both the New user and the New item case, as all items and users are new. Due to this some of the techniques developed to deal with those two cases are not applicable to the system bootstrapping.
		
# Multi-Armed Bandit (Problem)
- Source: https://en.wikipedia.org/wiki/Multi-armed_bandit
- In probability theory and machine learning, *the multi-armed bandit problem is a problem in which a fixed limited set of resources must be allocated between competing (alternative) choices in a way that maximizes their expected gain, when each choice's properties are only partially known at the time of allocation, and may become better understood as time passes or by allocating resources to the choice.* This is a classic reinforcement learning problem that exemplifies the exploration–exploitation tradeoff dilemma. *The name comes from imagining a gambler at a row of slot machines (sometimes known as "one-armed bandits"), who has to decide which machines to play, how many times to play each machine and in which order to play them, and whether to continue with the current machine or try a different machine.*
- *In the problem, each machine provides a random reward from a probability distribution specific to that machine, that is not known a-priori. The objective of the gambler is to maximize the sum of rewards earned through a sequence of lever pulls.* ***The crucial tradeoff the gambler faces at each trial is between "exploitation" of the machine that has the highest expected payoff and "exploration" to get more information about the expected payoffs of the other machines.***
## Thompson Sampling
- Source: https://en.wikipedia.org/wiki/Thompson_sampling
Thompson sampling is a heuristic for choosing actions that addresses the exploration-exploitation dilemma in the multi-armed bandit problem. It consists of choosing the action that maximizes the expected reward with respect to a randomly drawn belief.
- 다양한 추천 시스템 모델을 동시에 사용할 때, 추천을 제공하는 매 순간마다 어떤 모델을 사용하는 것이 적절한지를 알려주는 알고리즘입니다.

# Click-Through Rate (CTR)
- Source: https://en.wikipedia.org/wiki/Click-through_rate
- *Click-through rate (CTR) is the ratio of users who click on a specific link to the number of total users who view a page, email, or advertisement. It is commonly used to measure the success of an online advertising campaign for a particular website as well as the effectiveness of email campaigns.*
- 10번 중 2번 클릭이 발생한 것과 100번 중 20번 클릭이 발생한 것은 둘 다 CTR이 0.2입니다. 그러나 전자보다 후자가 시행횟수가 많으므로 0.2라는 값에 대한 확신이 더 크다고 할 수 있습니다.
```python
# case 1) 사용자에게 10번의 추천을 했을 때 2번 클릭한 경우
# case 2) 사용자에게 100번의 추천을 했을 때 20번 클릭한 경우
# case 3) 사용자에게 1000번의 추천을 했을 때 200번 클릭한 경우
fig, ax = plt.subplots(figsize=(10, 8))
for n_recs, n_clicks in zip([10, 100, 1000], [2, 20, 200]):
    #alpha, beta 대신에 a, b를 사용하겠습니다.
    a = n_clicks + 1
    b = n_recs - n_clicks + 1

    ctr = (a - 1)/(a + b - 2) #mode
    var = (a*b)/((a + b)**2*(a + b + 1)) #var

    xs = np.linspace(0, 1, 200)
    ys = stats.beta.pdf(xs, a, b)
    sb.lineplot(ax=ax, x=xs, y=ys, label=f"n_recs : {n_recs:<5d} n_clicks : {n_clicks:<5d}")
```

# Content-Based Filtering & Collaborative Filtering (CF)
- Source: https://www.mygreatlearning.com/blog/matrix-factorization-explained/
## Content-Based Filtering
- *This approach recommends items based on user preferences. It matches the requirement, considering the past actions of the user, patterns detected, or any explicit feedback provided by the user, and accordingly, makes a recommendation.* Example: If you prefer the chocolate flavor and purchase a chocolate ice cream, the next time you raise a query, the system shall scan for options related to chocolate, and then, recommend you to try a chocolate cake.
- *The model can capture specific interests of a user, and can recommend niche items that very few other users are interested in.*
- *Since the feature representation of the items are hand-engineered to some extent, this technique requires a lot of domain knowledge. Therefore, the model can only be as good as the hand-engineered features.*
- The model can only make recommendations based on existing interests of the user. In other words, the model has limited ability to expand on the users’ existing interests.
- Source: https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)
- ***Content-based filtering algorithms, on the other hand, are in theory much less prone to the new item problem. Since content based recommenders choose which items to recommend based on the feature the items possess, even if no interaction for a new item exist, still its features will allow for a recommendation to be made.*** Consider the case of so-called editorial features (e.g. director, cast, title, year), those are always known when the item, in this case movie, is added to the catalogue. However, other kinds of attributes might not be e.g. features extracted from user reviews and tags. *Content-based algorithms relying on user provided features suffer from the cold-start item problem as well.
## Collaborative Filtering (CF)
- *This approach uses similarities between users and items simultaneously, to provide recommendations. It is the idea of recommending an item or making a prediction, depending on other like-minded individuals.* Example: Suppose Persons A and B both like the chocolate flavor and have them have tried the ice-cream and cake, then if Person A buys chocolate biscuits, the system will recommend chocolate biscuits to Person B.
- In collaborative filtering, *we do not have the rating of individual preferences and car preferences. We only have the overall rating, given by the individuals for each car. As usual, the data is sparse, implying that the person either does not know about the car or is not under the consideration list for buying the car, or has forgotten to give the rating.*
```python
ui = pd.pivot_table(data, index="user", columns="item", values="preferences")
```
- Step 1: Normalization
	- ***Usually while assigning a rating, individuals tend to give either a high rating or a low rating, across all parameters. Normalization usually helps in balancing and evens out such measures. This is done by taking an average of rating available and subtracting it with the individual rating.***
	- ***If you add all the numbers in each row, then it will add up-to zero. Actually, we have centered overall individual rating to zero. From zero, if individual rating for each car is positive, it means he likes the car, and if it is negative it means he does not like the car.***
	```python
	avg_rating = ui.mean().mean()
	# User-Side Popularity-Opportunity Bias
	user_bias = ui.mean(axis=1) - avg_rating
	# Item-Side Popularity-Opportunity Bias
	item_bias = ui.mean(axis=0) - avg_rating
	adj_ui = ui.sub(user_bias, axis=0).sub(item_bias, axis=1)
	```
- Step 2: Similarity measure
	- Cosine similarity
		- Implementation
			```python
			import pandas as pd
			import numpy as np
			
			adj_ui = adj_ui.fillna(0)
			# Vector Normalization
			norm_item = np.linalg.norm(adj_ui, axis=0, ord=2)
			adj_ui_norm = adj_ui.div(norm_item, axis=1)
			titles = [id2title[int(i)] for i in adj_ui.columns]
			cos_sim_item = pd.DataFrame(np.dot(adj_ui_norm.T, adj_ui_norm), index=titles, columns=titles)
			```
		- Using `sklearn.metrics.pairwise.cosine_similarity()`
			```python
			from sklearn.metrics.pairwise import cosine_similarity
			
			titles = [id2title[int(i)] for i in adj_ui.columns]
			cos_sim_item = pd.DataFrame(cosine_similarity(adj_ui.T), index=titles, columns=titles)
			```
	- Eculidean similarity
		- Implementation
			```python
			import pandas as pd
			import numpy as np
			
			adj_ui = adj_ui.fillna(0)
			square = np.array(np.square(adj_ui).sum(axis=0))
			square = np.add.outer(square, square)
			dot = np.dot(adj_ui.T, adj_ui)
			euc_dist_item = np.sqrt(square - 2*dot)
			# titles = [id2title[int(i)] for i in adj_ui.columns]
			euc_sim_item = 1/(1 + euc_dist_item)
			np.fill_diagonal(euc_sim_item, 1)
			euc_sim_item = pd.DataFrame(euc_sim_item, index=titles, columns=titles)
			```
		- Using `sklearn.metrics.pairwise.euclidean_distances()`
			```python
			from sklearn.metrics.pairwise import euclidean_distances
			
			euc_dist_item = euclidean_distances(adj_ui.T)
			euc_sim_item = pd.DataFrame(1/(1 + euc_dist_item), index=titles, columns=titles)
			```
- Step 3: Recommendation
	```python
	title = "Toy Story"
	display(cos_sim_item[title].sort_values(ascending=False)[1:6])
	display(euc_sim_item[title].sort_values(ascending=False)[1:6])
	```
### Memory-Based
#### User-Based
#### Item-Based
### Model-Based
#### Matrix Factorization (MF)
- Source: https://www.mygreatlearning.com/blog/matrix-factorization-explained/
##### BPR (Bayesian Personalizaed Ranking)
- BPR은 implicit data에 사용 가능한 matrix factorization 알고리즘 중 하나입니다.
- 베이지안 개인화 랭킹 알고리즘은 모델을 학습시킬 때 긍정 아이템(예: 유저가 들어본 아티스트)과 부정 아이템(예: 유저가 들어보지 않은 아티스트) 사이의 랭킹을 목적함수로 두고, 그 차이가 커지는 방향으로 임베딩 변수 값을 갱신하는 것이죠.
- BPR의 핵심 아이디어는 바로 고객이 본 영화에 대한 선호도가 보지 않은 영화에 대한 선호도보다 항상 높다는 것입니다.
- Source: https://towardsdatascience.com/recommender-system-bayesian-personalized-ranking-from-implicit-feedback-78684bfcddf6
- *The personalized ranking provides customers with item recommendations of a ranked list of items. The article would focus on recommending customers with a personalized ranked list of items from users’ implicit behavior derived from the past purchase data.*
- ***For the implicit feedback systems, it is able to detect the positive dataset like bought history. For the remaining data, it is a mixture of actually negative and missing values. Nevertheless, machine learning models are unable to learn the missing data.***
- Implementation
	```python
	```
- Using `implicit.bpr.BayesianPersonalizedRanking()`
	```python
	likes["user_id"] = pd.Categorical(likes["user_id"])
	likes["artist_id"] = pd.Categorical(likes["artist_id"])

	vals = likes["plays"]
	rows = likes["artist_id"].cat.codes.values
	cols = likes["user_id"].cat.codes.values

	ui_sparse = csr_matrix((vals, (rows, cols)))

	# Embedding vector에는 마지막에 bias가 하나씩 붙어서 크기가 `factors + 1`이 됩니다.
	model = BPR(factors=60)
	model.fit(ui_sparse)

	user_embs = model.user_factors
	item_embs = model.item_factors

	id2name = {row["artist_id"]:row["artist_name"] for _, row in artists.iterrows()}

	item_embs = pd.DataFrame(item_embs, index=[id2name[i] for i in likes["artist_id"].cat.categories])
	user_embs = pd.DataFrame(user_embs, index=likes["user_id"].cat.categories)
	```
- Using `tensorflow`
	```python
	```
##### ALS (Alternating Least Squares)
##### Logistic Matrix Factorization

# Association Rule Learning (= Association Analysis)
- Source: https://en.wikipedia.org/wiki/Association_rule_learning, https://livebook.manning.com/book/machine-learning-in-action/chapter-11/51
- Association rule learning is a rule-based machine learning method for discovering interesting relations between variables in large databases. It is intended to identify strong rules discovered in databases using some measures of interestingness. In any given transaction with a variety of items, association rules are meant to discover the rules that determine how or why certain items are connected.
- For example, the rule {onions, potatoes} -> {burger} found in the sales data of a supermarket would indicate that if a customer buys onions and potatoes together, they are likely to also buy hamburger meat. Such information can be used as the basis for decisions about marketing activities such as, e.g., promotional pricing or product placements.
- In contrast with sequence mining, *association rule learning typically does not consider the order of items either within a transaction or across transactions.*
- Association rules are made by searching data for frequent if-then patterns and by using a certain criterion under Support and Confidence to define what the most important relationships are. *Support is the evidence of how frequent an item appears in the data given, as Confidence is defined by how many times the if-then statements are found true. However, there is a third criteria that can be used, it is called Lift and it can be used to compare the expected Confidence and the actual Confidence. Lift will show how many times the if-then statement is expected to be found to be true.
- *Looking at items commonly purchased together can give stores an idea of customers’ purchasing behavior.* This knowledge, extracted from the sea of data, can be used for pricing, marketing promotions, inventory management, and so on. Looking for hidden relationships in large datasets is known as association analysis. Brute-force solutions aren’t capable of solving this problem, so a more intelligent approach is required to find frequent itemsets in a reasonable amount of time.
- Association analysis is the task of finding interesting relationships in large datasets. These interesting relationships can take two forms: frequent item sets or association rules. *Frequent item sets are a collection of items that frequently occur together.* The second way to view interesting relationships is association rules. *Association rules suggest that a strong relationship exists between two items.*
- From the dataset we can also find an association rule such as diapers → wine. *This means that if someone buys diapers, there’s a good chance they’ll buy wine.* With the frequent item sets and association rules, retailers have a much better understanding of their customers. Although common examples of association analysis are from the retail industry, it can be applied to a number of other industries, such as website traffic analysis and medicine.
## Support
- (Number of transactions containing the itemset)/(Total number of transactions)
- Support is an indication of how frequently the itemset appears in the dataset.
- When using antecedents and consequents, it allows a data miner to determine the support of multiple items being bought together in comparison to the whole data set. For example, Table 2 shows that if milk is bought, then bread is bought has a support of 0.4 or 40%. This because in 2 out 5 of the transactions, milk as well as bread are bought. In smaller data sets like this example, it is harder to see a strong correlation when there are few samples, but when the data set grows larger, support can be used to find correlation between two or more products in the supermarket example.
- Minimum support thresholds are useful for determining which itemsets are preferred or interesting.
- *Minimum threshold is used to remove samples where there is not a strong enough support or confidence to deem the sample as important or interesting in the dataset.*
## Confidence
- (Number of transactions containing X and Y)/(Number of transactions containing X)
- Confidence is the percentage of all transactions satisfying X that also satisfy Y. The confidence value of an association rule, often denoted as X -> Y, is the ratio of transactions containing both X and Y to the total amount of X values present, where X is the antecedent and Y is the consequent.
## Lift
```python
def lift(x, y):
    return confidence(x, y)/support(y)
```
- ***If the rule had a lift of 1, it would imply that the probability of occurrence of the antecedent and that of the consequent are independent of each other. When two events are independent of each other, no rule can be drawn involving those two events.***
- ***If the lift is > 1, that lets us know the degree to which those two occurrences are dependent on one another, and makes those rules potentially useful for predicting the consequent in future data sets. 전체 사용자들 중 아이템 y를 좋아하는 비율(Support)보다 아이템 x를 좋아하는 사용자들 중 아이템 y도 좋아하는 비율(Confidence)가 매우 크다면 아이템 x와 y는 매우 연관 관계가 높을 것입니다.***
- ***If the lift is < 1, that lets us know the items are substitute to each other. This means that presence of one item has negative effect on presence of other item and vice versa. 전체 사용자들 중 아이템 y를 좋아하는 비율(support)보다 아이템 x를 좋아하는 사용자들 중 아이템 y도 좋아하는 비율(confidence)이 매우 작다면 아이템 x와 y는 매우 연관 관계가 낮을 것입니다.***
- 즉 어떤 사용자가 아이템 x를 좋아한다는 사실이 아이템 y를 좋아할 가능성이 높인다는 의미입니다.
### Apriori
- Many algorithms for generating association rules have been proposed. *Some well-known algorithms are Apriori, Eclat and FP-Growth, but they only do half the job, since they are algorithms for mining frequent itemsets (Lists of items that commonly appear together). Another step needs to be done after to generate rules from frequent itemsets found in a database.*
- Apriori is for frequent itemset mining and association rule learning. *It proceeds by identifying the frequent individual items in the database and extending them to larger and larger item sets as long as those itemsets appear sufficiently often.*
- Apriori uses a "bottom up" approach, where *frequent subsets are extended one item at a time (a step known as candidate generation), and groups of candidates are tested against the data. The algorithm terminates when no further successful extensions are found.* Apriori uses breadth-first search and a Hash tree structure to count candidate item sets efficiently. It generates candidate item sets of length `max_len`. Then it prunes the candidates which have an infrequent sub pattern.
- Source: http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
- Apriori is a popular algorithm [1] for extracting frequent itemsets with applications in association rule learning. The apriori algorithm has been designed to operate on databases containing transactions, such as purchases by customers of a store. An itemset is considered as "frequent" if it meets a user-specified support threshold. For instance, if the support threshold is set to 0.5 (50%), a frequent itemset is defined as a set of items that occur together in at least 50% of all transactions in the database.
- Support와 Confidence의 최소 기준을 정하여 그 기준에 미달하는 연관 관계를 제거합니다. Support와 Confidence가 너무 작으면 Lift의 의미가 왜곡되어 해석될 수 있기 때문입니다.
- Using `mlxtend.frequent_patterns.apriori()`
	```python
	import os
	import pickle as ps
	import mlxtend
	from mlxtend.preprocessing import TransactionEncoder
	from mlxtend.frequent_patterns import apriori, association_rules
	
	n_users = likes["user_id"].nunique()
	likes = ratings[(ratings["rating"]>=4.0)]
	user_likes = likes.groupby("user_id")["movie_id"].apply(set)

	# The apriori function expects data in a one-hot encoded pandas DataFrame.
	enc = TransactionEncoder()
	user_likes_ohe = enc.fit_transform(user_likes)
	user_likes_ohe = pd.DataFrame(user_likes_enc, index=user_likes.index, columns=enc.columns_)
	
	# The entries in the `itemsets` column are of type `frozenset`
	file = "./frequent_sets.pkl"
	if os.path.exists(file):
		with open(file, "rb") as f:
			freq_sets = pk.load(f)
	else:
		with open(file, "wb") as f:
			# `max_len`: (default `None`) Maximum length of the itemsets generated.
			# `low_memory`: Note that while `True` should only be used for large dataset if memory resources are limited, because this implementation is approx. 3-6x slower than the default.
			freq_sets = apriori(user_likes_ohe, min_support=0.01, max_len=2, use_colnames=True, verbose=1)
			pk.dump(freq_sets, f)
			
	# `df`: DataFrame of frequent itemsets with columns `['support', 'itemsets']`.
	# `metric` (`"support"`, `"confidence"`, `"lift"`) Metric to evaluate if a rule is of interest.
	asso_rules = association_rules(freq_sets, metric="confidence", min_threshold=0.01)
	# Lift를 기준으로 추천합니다.
	asso_rules = asso_rules.sort_values("lift", ascending=False)
	asso_rules = asso_rules.drop(["leverage", "conviction"], axis=1)
	```
### FP-Growth (Frequent Pattern-Growth)
# Factorization Machine
# DeepFM

# K-Core Pruning
- For Last.fm
	```python
	# `plays`가 전체 `plays`의 하위 10% 초과인 레코드만 남깁니다. 이 경우에만 각 사용자가 각 아시트를 좋아한다고 판단하는 것입니다.
	likes = plays[plays["plays"]>plays["plays"].quantile(0.1)]

	thr = 5
	len_prev = -1
	len_next = -2
	while len_prev != len_next:
		len_prev = len(likes)
		print(f"len(likes): {len(likes):,}")
		
		# `thr`명보다 많은 수의 아티스트의 음악을 들은 사용자
		user_n_artists = likes["user_id"].value_counts()
		users_ = user_n_artists[user_n_artists>thr].index
		
		# `thr`명보다 많은 수의 사용자가 음악을 들은 아티스트
		artist_n_users = likes["artist_id"].value_counts()
		artists_ = artist_n_users[artist_n_users>thr].index

		likes = likes[(likes["user_id"].isin(users_)) & (likes["artist_id"].isin(artists_))]
		len_next = len(likes)
	print("Finished!")
	```
- For MovieLens 100k
	```python
	thr = 5
	len_prev = -1
	len_next = -2
	while len_prev != len_next:
		len_prev = len(ratings)
		print(f"len(ratings): {len(ratings):,}")
		
		user_n_ratings = ratings["user_id"].value_counts()
		users_ = user_n_ratings[user_n_ratings>thr].index
		
		item_n_ratings = ratings["item_id"].value_counts()
		items_ = item_n_ratings[item_n_ratings>thr].index

		ratings = ratings[(ratings["user_id"].isin(users_)) & (ratings["item_id"].isin(items_))]
		len_next = len(ratings)
	print("Finished!")
	```

# Evaluation Metrices
## Hit Rate
- Source: https://towardsdatascience.com/evaluating-a-real-life-recommender-system-error-based-and-ranking-based-84708e3285b
- Let’s see how good our top-10 recommendations are. To evaluate top-10, we use hit rate, that is, ***if a user rated one of the top-10 we recommended, we consider it is a "hit"***.
- The process of compute hit rate for a single user:
	1. Find all items in this user’s history in the training data.
	2. Intentionally remove one of these items (Leave-One-Out CV).
	3. Use all other items to feed the recommender and ask for top 10 recommendations.
	4. If the removed item appear in the top 10 recommendations, it is a hit. If not, it’s not a hit.
- ***The whole hit rate of the system is the count of hits, divided by the test user count.***
## HR (Hit Ratio)
- Source: https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54
- ***The number of users for which the correct answer is included in the top L recommendation list, divided by the total number of users in the test dataset.***
## MAP(Mean Average Precision)
- 하지만 사용자에 따라서 실제로 소비한 아이템의 수가 천 개, 2천개까지 늘어날 수 있습니다. 이 경우 recall이 1이 되는 지점까지 고려하는 것은 무리이므로 최대 n개까지만 고려하여 mAP를 계산하며, 이 경우 mAP@n 으로 표기합니다.
- Source: https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54
## nDCG(normalized Discounted Cumulative Gain)
- Source: https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54

# 1. 신규 고객(고객에 대한 정보 없음) : 
## 1-1. 비개인화 추천(Non-personalized Recommendation)
### (1) 신규 아이템 추천
- ex. 2020년에 개봉한 영화 추천
### (2) 평이 좋은 아이템 추천
- explicit data가 있을 때만 사용 가능한 방법
    - 점수, 후기 등 사용자가 자신의 선호를 직접적으로 데이터
- ex. 평점의 평균이 높은 영화 추천(이때 평점의 갯수가 너무 적은 것은 통계적 신뢰성이 낮으므로 제외해야 함)
### (3) 인기 아이템 추천
- implicit data에도 사용 가능한 방법
    - 장바구니 기록, 시청 횟수 등 사용자가 자신의 선호를 직접적으로 드러내지 않은 데이터
    - 따라서 고객은 자신이 선호하는 아이템에 대해서 주로 로그를 발생시킨다고 가정하여 접근(ex. 사람들은 좋아하는 음악을 많이 듣지, 좋아하지 않는 음악은 거의 듣지 않음. 따라서 주로 긍정적 반응이 기록됨)
    - 사용자의 행동에 따라 자동으로 기록되므로 explicit data보다 밀하다(dense)(평점은 고객이 작성해야 데이터화되지만 로그는 자동으로 기록됨)
    - 그러나 로그 횟수를 별점으로 해석하면 안 됨. 즉 10번 들은 곡보다 1000번 들은 곡을 100배 좋아하는 것은 아님. 또한 로그는 오래된 곡에 대한 것일수록 많을 수밖에 없음.
- ex. 쿠팡에서 고객들의 클릭 횟수가 가장 많은 제품 추천
### 1-2. 연관 분석(Association Analysis)
- 대용량의 거래 데이터로부터 "X를 구매했으면, Y를 구매할 것이다" 형식의 아이템 간 연관 관계를 분석하는 방법
- 보통 장바구니 분석(Market Basket Analysis)로 불리기도 합니다. 즉, 고객의 장바구니에 어떤 아이템이 동시 담겼는지 패턴을 파악하여 상품을 추천하는 방법입니다.
- 전체 사용자들 중 아이템 y를 좋아하는 비율(support)보다 아이템 x를 좋아하는 사용자들 중 아이템 y도 좋아하는 비율(confidence)가 매우 크다면 아이템 x와 y는 매우 연관 관계가 높을 것입니다.(lift = confidence/support가 클수록 연관 관계가 높음. lift가 얼마 이상이어야 연관 관계가 높은지에 대한 판단은 정해진 것이 없음)
- 반면에 전체 사용자들 중 아이템 y를 좋아하는 비율(support)보다 아이템 x를 좋아하는 사용자들 중 아이템 y도 좋아하는 비율(confidence)이 매우 작다면 아이템 x와 y는 매우 연관 관계가 낮을 것입니다.
- "lift"의 의미는 어떤 사용자가 아이템 x를 좋아한다는 사실이 아이템 y를 좋아할 가능성을 높인다는 의미입니다.
- (평점을 4점 이상 준 것을 해당 영화를 좋아한다는 표시로 볼 때)

# 2. 기존 고객 재방문(고객에 대한 약간의 정보 있음)
## 2-1. Content-based Filtering
- 고객 또는 아이템의 세부 정보를 바탕으로 추천하는 것(ex. 영화 "One Day"에 높은 평점을 준 고객에게 동일한 장르의 영화를 추천)
- 그러나 추천한 영화가 "One Day"와 장르는 동일하지만 세부적인 내용이 완전히 달라 선호하지 않을 수 있음. 또한 고객이 좋아하는 영화 중 "One Day"와 장르가 다른 영화는 추천할 수 없음
-  가 동일하지만 세부적인 내용은 상당히 다를 수 있습니다. 또한 장르는 다르지만 해당 사용자가 좋아할 만한 영화도 있을 수 있습니다.
- 이렇게 장르와 같이 한정적인 정보만으로는 정교한 추천이 어려우며 영화별로 출연 배우나 감독 등 더 세부적인 정보를 수집하는 것 또한 많은 시간과 노력을 필요로 함
## 2-2. Collaborative Filtering(CF)
- 고객과 아이템 간 상호작용 데이터 활용
- User-Item Matrix 만들기
고객 79,044명, 영화 3,413개
### (1) Memory-based
- 별도의 특성 추출 과정 없음
#### User-based CF
-  어떤 사용자와 유사한 취향을 가진 사용자가 좋아하는 아이템을 추천
#### Item-based CF
- 어떤 아이템과 유사한 아이템을 추천
- Item Similarity Matrix 구하기
- 한계 : 연산량이 지나치게 많아 실시간으로 반영하기 어려움
### (2) Model-based CF
- 특성을 추출하여 embedding vectors 생성
#### 행렬 분해(Matrix Factorization)
- 아래 예시는 2차원이지만 실제로는 100차원 이상을 사용하기도 합니다.
- embedding vectors를 생성했다면 평점을 예측할 수 있습니다.
- 고객과 고객 또는 아이템과 아이템 간의 유사도를 계산할 수 있습니다.

# 3. 단골
- Factorization Machine(FM)

# Association Rules
- Source: 강의 자료
- 연관분석(Association Analysis)은 대용량의 거래(transaction) 데이터로부터 "X를 구매했으면, Y를 구매할 것이다" 형식의 아이템 간 연관 관계를 분석하는 방법입니다.
보통 장바구니 분석(Market Basket Analysis)로 불리기도 합니다. 즉, 고객의 장바구니에 어떤 아이템이 동시 담겼는지 패턴을 파악하여 상품을 추천하는 방법입니다.
## 2. 연관 분석의 주요 지표
- 연관분석에서는 크게 지지도, 신뢰도, 리프트라는 세 가지 지표를 통해 아이템 간의 관계를 표현합니다. 각각의 의미를 알아봅시다.
### (1) Support
- 스타워즈2를 재미있게 본 유저가 있다고 합시다. 이 유저에게 어떤 영화를 추천하는 것이 좋을까요? 가장 단순한 방법은 각각의 영화를 전체 유저 중에 얼마나 되는 사람이 좋아하는지 알아보고, 많은 인기(혹은 지지)를 받은 영화를 찾아서 추천하는 것입니다. 예를 들면 대부분 사람들이 타이타닉을 선호하는 만큼 해당 유저도 타이타닉을 선호할 거으로 보는 것이죠.
이 확률값을 '지지도(Support)'라고 부릅니다. 전체 유저 중에 스타워즈 3, 스타트렉, 러브액츄얼리, 타이타닉을 선호하는 유저의 수를 각각 구하면 알 수 있습니다.
### (2) Confidence
- "유저가 스타워즈2를 재미있게 보았다"는 정보를 이용해서 영화 스타워즈3를 좋아할 확률을 보다 정확하게 알 수는 없을까요? 스타워즈2를 좋아하는 유저 중에는 대상 영화를 좋아하는 유저가 얼마나 되는지 알아볼 수 있을 것입니다. 스타워즈2를 좋아하는 유저들 중에 대상 영화를 좋아했던 유저가 많다면, 이 유저 역시 대상 영화를 좋아할 확률이 높다고 보는 것이죠. 이 확률값을 '신뢰도(Confidence)'라고 하며, 영화X를 좋아하는 유저 중에 영화Y를 좋아하는 유저(즉, 영화X와 영화Y를 모두 좋아한 유저)의 비율로 계산합니다.
### (3) Lift
- 그렇다면 스타워즈2를 선호했다는 사실이 대상 영화 대한 선호를 파악하는데 얼마나 중요했을까요? 유저 전반적으로 대상 영화 Y를 좋아할 확률(지지도)보다 스타워즈2라는 영화를 좋아하는 사람 중에 대상 영화 Y를 좋아할 확률(신뢰도)이 더 크다면, 스타워즈2를 선호한다는 사실이 대상 영화Y를 선호할 것으로 예상하는 데에 대한 확신을 높여줄 것입니다.
confidence(StarWars2→StarWars3)>support(StarWars3)
- 반면에, 전반적으로 타이타닉을 좋아할 확률이 스타워즈2를 좋아하는 사람 중에 타이타닉을 좋아할 확률(신뢰도)이 더 높다면, 타이타닉과 스타워즈2의 연관관계는 높지 않을 것입니다.
confidence(StarWars2→Titanic)<support(Titanic)
- 이처럼 지지도와 신뢰도를 이용해 아이템의 관계를 파악하는 지표가 바로 리프트(Lift)입니다. 리프트는 어떻게 구할까요?
lift(StarWars2→Y)=confidence(StarWars2→Y)support(Y)
- 리프트가 1보다 크면 전자의 상황을, 1보다 작으면 후자의 상황을 뜻하는 것이죠. 스타워즈2를 재미있게 보았다는 정보를 얻고 나니 대상 영화Y를 재미있게 볼 확률이 기본 확률값(지지도)에 비해 높아졌는지, 낮아졌는지 확인하는 것이죠. '리프트'라는 지표의 이름은 "어떤 증거가 신뢰도를 높여주는가?"라는 의미에서 나온 것입니다.
- Source: https://yamalab.tistory.com/86?category=747907
- Association Rule은 고객들의 상품 묶음 정보를 규칙으로 표현하는 가장 기본적인 알고리즘이다. 흔히 장바구니 분석이라고도 불린다. 데이터마이닝 같은 수업을 들었다면 한번 쯤 들어봤을 법한 알고리즘이다. 이 알고리즘은 기초적인 확률론에 기반한 방법으로, 전체 상품중에 고객이 함께 주문한 내역을 살펴본 뒤 상품간의 연관성을 수치화하여 나타내는 알고리즘이다. 매우 직관적이고 구현하기도 쉽지만, 그렇다고 현재로서 성능이 매우 떨어지는 알고리즘도 아니다. 추천 시스템에서 여전히 가장 중요한 알고리즘으로 분류되며 Association Rule에서 파생된 다양한 알고리즘들이 존재한다. 

# Redis
- sources: https://medium.com/@jyejye9201/%EB%A0%88%EB%94%94%EC%8A%A4-redis-%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80-2b7af75fa818, https://hwigyeom.ntils.com/entry/Windows-%EC%97%90-Redis-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0-1
- REDIS(REmote Dictionary Server)는 메모리 기반의 “키-값” 구조 데이터 관리 시스템이며, 모든 데이터를 메모리에 저장하고 조회하기에 빠른 Read, Write 속도를 보장하는 비 관계형 데이터베이스이다.
- 레디스는 크게 5가지< String, Set, Sorted Set, Hash, List >의 데이터 형식을 지원한다.
- Redis는 빠른 오픈 소스 인 메모리 키-값 데이터 구조 스토어이며, 다양한 인 메모리 데이터 구조 집합을 제공하므로 사용자 정의 애플리케이션을 손쉽게 생성할 수 있다.

# Frequent Set
- 빈번하게 등장한 아이템의 쌍을 빈발집합이라고 부릅니다. 앞서 연관분석의 세 가지 주요 지표의 수식을 떠올려보면, 모두 빈도(Freq())를 이용해 만들어졌음을 알 수 있습니다. 연관분석을 실제 구매 데이터에 적용한다면, 각각의 아이템의 쌍이 얼마나 등장했는지를 세어야 할 것입니다.하지만 아이템의 가짓수가 늘어나고, 확인해야 할 바스켓의 수가 커지면, 이에 대한 계산은 기하급수적으로 늘어나게 됩니다.
이 문제를 해결하기 위해 고안된 것이 자주 등장하는 아이템의 쌍만을 빠르게 추려 계산하는 빈발집합 탐색 알고리즘입니다. 대표적인 빈발집합 탐색 알고리즘으로는 Apriori 알고리즘과 FP-Growth 알고리즘이 있습니다. 둘 다 데이터 셋 내에서 빈발집합을 찾아내고, 몇 번이나 등장했는지를 세어주는 알고리즘으로, 두 알고리즘의 결과는 동일합니다. 코드의 최적화 수준에 따라 조금씩 달라지지만, 일반적으로 FP-Growth 알고리즘이 Apriori 알고리즘보다 빠릅니다.
이번에는 Apriori 알고리즘을 사용하겠습니다. Apriori 알고리즘은 모든 가능한 조합의 개수를 줄이는 전략을 사용합니다.아래 이미지를 보면, 5가지 아이템이 있다고 할 때, 이 5가지를 이용해 나올 수 있는 가능한 조합은 총  25−1=31 개 입니다.아이템 수가 늘어날수록 아이템 조합 역시 급격하게 늘어날 것입니다.
Apriori는 각 조합의 지지도를 구하면서 조합의 아이템 수를 늘리며 내려가면서 어떤 조합의 지지도가 일정 기준 이하로 떨어지면, 그 아래로 내려가도(즉, 조합의 아이템 수를 늘리더라도) 빈발집합이라고 볼 수 없다 판단하여 더 이상 가지를 따라 내려가지 않고 쳐내는 식으로 빈발집합을 탐색합니다.
- <img src="https://i.imgur.com/pZ75IjW.png">

# Factorization Machine
- sources: https://zzaebok.github.io/machine_learning/factorization_machines/, https://greeksharifa.github.io/machine_learning/2019/12/21/FM/

# DeepFM
- source: https://greeksharifa.github.io/machine_learning/2020/04/07/DeepFM/
- 모델에 대해 설명할 것이다. 이 DeepFM이라는 모델은 FM과 딥러닝을 결합한 것이다. 최근(2017년 기준) 구글에서 발표한 Wide & Deep model에 비해 피쳐 엔지니어링이 필요 없고, wide하고 deep한 부분에서 공통된 Input을 가진다는 점이 특징적이다.
- DeepFM은 피쳐 엔지니어링 없이 end-to-end 학습을 진행할 수 있다. 저차원의 interaction들은 FM 구조를 통해 모델화하고, 고차원의 interaction들은 DNN을 통해 모델화한다.
- DeepFM은 같은 Input과 Embedding 벡터를 공유하기 때문에 효과적으로 학습을 진행할 수 있다.
- source: https://orill.tistory.com/category/RecSys
- 최근 구글에서 발표한 Wide & Deep Model과 비교해보면, DeepFM은 Wide 부분과 Deep 부분이 공통된 입력(shared input)을 받고 Feature Engineering이 필요하지 않다.
- 낮은 차원, 높은 차원의 상호 작용 둘 다를 모델링하기 위해 [Cheng et al., 2016]은 linear("wide") 모델과 deep 모델을 결합한 흥미로운 혼합 네트워크 구조(Wide & Deep)를 제시했다. 이 모델에서는 두 개의 다른 inputs이 wide 부분과 deep 부분을 위해 각각 필요하다. "wide part"의 입력은 여전히 전문가의 feature engineering이 필요하다.
- 이 모델은 FM을 통해 낮은 차원의 피쳐 상호작용을 학습하고 DNN을 통해서는 높은 차원의 피쳐 상호작용을 학습한다. Wide & Deep 모델과는 다르게 DeepFM은 end-to-end로 feature engineering이 필요 없이 학습할 수 있다.
- DeepFM은 Wide & Deep과는 다르게 같은 input과 embedding layer를 공유하기 때문에 효율적으로 학습할 수 있다. Wide & Deep 에서는 input vector가 직접 고안한 pairwise 피쳐 상호작용을 wide part에 포함하기 때문에 입력 vector가 굉장히 커질 수 있고 이는 복잡도를 굉장히 증가시킨다.
- 1) 다른 field의 input vector의 길이는 다를 수 있지만 embedding은 같은 크기(k)이다. (역자 예시: gender field는 보통 length가 남, 여 2인 반면 국적이나 나이 field의 길이는 더 길다. 하지만 embedding시에는 똑같이 k=5차원 벡터로 임베딩 된다.)

# Evaluation Metrics
## MAP(Mean Average Precision)
- 하지만 사용자에 따라서 실제로 소비한 아이템의 수가 천 개, 2천개까지 늘어날 수 있습니다. 이 경우 recall이 1이 되는 지점까지 고려하는 것은 무리이므로 최대 n개까지만 고려하여 mAP를 계산하며, 이 경우 mAP@n 으로 표기합니다.
## nDCG(normalized Discounted Cumulative Gain)
- 추천 엔진은 기본적으로 각 아이템에 대해서 사용자가 얼마나 선호할 지를 평가하며, 이 스코어 값을 relevance score라고 부릅니다. 그리고 이 relevance score 값들의 총 합을 Cumulative Gain(CG)라고 부릅니다. 먼저 위치한 relavance score가 CG에 더 많은 영향을 줄 수 있도록 할인의 개념을 도입한 것이 Discounted Cumulative Gain(DCG)입니다.
- <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https://blog.kakaocdn.net/dn/dikjW1/btqDvgFUh3K/GdjWcm9XS4zpqECsQx9Nu1/img.png" height="80px">
- 하루에 100개의 동영상을 소비하는 사용자와 10개의 동영상을 소비하는 사용자에게 제공되는 추천 아이템의 개수는 다를 수 밖에 없습니다. 이 경우 추천 아이템의 개수를 딱 정해놓고 DCG를 구하여 비교할 경우 제대로 된 성능 평가를 진행할 수 없습니다. 때문에 DCG에 정규화를 적용한 NDCG(normalized discounted cumulative gain)이 제안됩니다. NDCG를 구하기 위해서는 먼저 DCG와 함께 추가적으로 iDCG를 구해주어야 합니다. iDCG의 i는 ideal을 의미하며 가장 이상적으로 relavace score를 구한 것을 말합니다. NDCG는 DCG를 iDCG로 나누어 준 값으로 0에서 1 사이의 값을 가지게 됩니다.
- <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https://blog.kakaocdn.net/dn/r9q0s/btqDxJzAlCa/Pccibd37tTjavu1QBeMZeK/img.png" height="200px">
## Entropy Diversity
- 그런데 아직 전체 아이템에서 얼마나 다양하게 추천을 진행했는지는 평가하지 못했습니다. Entropy Diversity는 추천 결과가 얼마나 분산 되어 있느냐를 평가하는 지표입니다.
- Entropy는 섀넌의 정보 이론에서 등장한 개념으로 머신러닝에서도 많이 사용됩니다. 간략하게 설명하면 잘 일어나지 않는 사건의 정보량은 잘 일어나는 사건의 정보량보다 많다는 것입니다. 이를 사건이 일어날 확률에 로그를 씌워서 정보량을 표현하며 로그의 밑의 경우 자연 상수를 취해줍니다. (다른 상수도 가능하긴 합니다.)
- <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https://blog.kakaocdn.net/dn/bOwzN3/btqDxt4UBcd/s0Kx6WWXyITuoKXaSZMAsk/img.png" height="250px">
- entropy란 발생할 수 있는 모든 사건들의 정보량의 기대 값입니다.
- <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https://blog.kakaocdn.net/dn/sXcph/btqDwdoeBsv/IjVthd4jsDV0jt3p6MCfT0/img.png" height="70px">
- Entropy Diversity란 이러한 엔트로피의 개념을 추천 결과에 적용한 것입니다. 모든 사용자들에게 비슷한 종류의 상품을 추천할 경우 해당 상품 추천은 자주 발생하므로 정보량이 낮습니다. 반면 개인에게 맞춤화 된 추천은 발생 횟수가 적으므로 정보량이 높아집니다. 이들의 기대값을 구한 것이 바로 Entropy Diversity입니다.
- 그러나 Entropy Diversity 만으로 추천 엔진이 더 정확하다고 평가할 수 는 없습니다. 어디까지나 추천 결과의 다양성을 측정하는 지표이므로 mAP나 NDCG처럼 정확도를 측정할 수 있는 지표와 함께 사용하는 것이 바람직해 보입니다.
---------------------------------------------------
# 협업 필터링(Collaborative Filtering, CF)
## 1) 사용자 기반 추천 (User-based Recommendation)
- 나와 비슷한 성향을 지닌 사용자를 기반으로, 그 사람이 구매한 상품을 추천하는 방식입니다.
예를 들어 한 사용자가 온라인 몰에서 피자와 샐러드, 그리고 콜라를 함께 구매하고, 또 다른 사용자는 피자와 샐러드를 구매했다고 가정해보겠습니다. 알고리즘은 구매 목록이 겹치는 이 둘을 유사하다고 인식하고, 두 번째 사용자에게 콜라를 추천합니다.
## 2) 아이템 기반 추천 (Item-based Recommendation)
- 내가 이전에 구매했던 아이템을 기반으로, 그 상품과 유사한 다른 상품을 추천하는 방식입니다. 
상품 간 유사도는 함께 구매되는 경우의 빈도를 분석하여 측정합니다. 예를 들어 콜라와 사이다가 함께 구매되는 경우가 많다면 콜라를 구매한 사용자에게 사이다를 추천하는 것입니다.
## 한계
### 1) 콜드 스타트(Cold Start)
- 협업 필터링 알고리즘을 사용하기 위해 필수적인 요소를 눈치채셨나요? 바로 기존 데이터입니다. 
사용자 기반 추천방식만으로는 아무런 행동이 기록되지 않은 신규 사용자에게 어떠한 아이템도 추천할 수 없을 것입니다. 아이템 기반 추천방식에서도, 새로운 아이템이 출시되더라도 이를 추천할 수 있는 정보가 쌓일 때까지 추천이 어려워지겠죠.
콜드 스타트란 이러한 상황을 일컫는 말입니다. ‘새로 시작할 때의 곤란함’ 정도로 해석할 수 있습니다. 시스템이 아직 충분한 정보를 모으지 못한 사용자에 대한 추론을 이끌어낼 수 없는 문제입니다.
### 3) 롱테일(Long Tail), sparsity problem
- 롱테일은 파레토 법칙(전체 결과의 80%가 전체 원인의 20%에서 일어나는 현상)을 그래프로 나타내었을 때 꼬리처럼 긴 부분을 형성하는 80%의 부분을 일컫는 말입니다. 이 현상을 협업 필터링에 적용하면, 사용자들이 관심을 많이 보이는 소수의 콘텐츠가 전체 추천 콘텐츠로 보이는 비율이 높은 ‘비대칭적 쏠림 현상'이 발생한다는 의미입니다. 
아이템이나 콘텐츠의 수가 많다고 하더라도 사용자들은 소수의 인기 있는 항목에만 관심을 보이기 마련입니다. 따라서 사용자들의 관심이 저조한 항목은 정보가 부족하여 추천되지 못하는 문제점이 발생할 수 있습니다.
- <img src="https://t1.daumcdn.net/thumb/R1280x0.fpng/?fname=http://t1.daumcdn.net/brunch/service/user/16yJ/image/JKDg--QU_l-Byu2qXjutdGaZccU.png" height="360px">
## 알고리즘 분류
### memory based
- Memory based 알고리즘은 neighborhood model의 user based(user-oriented neighborhood model), Item based(item-oriented neighborhood model)로 나뉨.
- 메모리 기반 알고리즘(Neighborhood model 기준)은 유저와 아이템에 대한 matrix를 만든 뒤, 유저 기반 혹은 아이템 기반으로 유사한 객체를 찾은 뒤 빈공간을 추론하는 알고리즘이다.
- 먼저 Neighborhood model은 주어진 평점 데이터를 가지고 서로 비슷한 유저 혹은 아이템을 찾습니다. 이 때 유사도는 주로 Pearson 상관계수를 통해 구합니다.
### model based
- model based 알고리즘은 latent factor model 가운데 matrix factorization(MF), RBM, 베이지안 등으로 나뉨.
- Implicit Dataset이 주어질 경우, Latent Factor Model이 Neighborhood model 보다 뛰어난 성능을 보입니다.
- 반대로 item based CF의 경우, 어벤져스에 대해 내린 유저의 평가와 가장 유사한 레디플레이어원을 유사도 계산으로 찾아내어 빈공간을 메우는 것에 이용된다. 두 방법 모두 콜드 스타트 문제점을 안고있지만, item based CF가 Hybrid 방식을 적용하여 콜드 스타트 문제를 해결하기에 조금 더 능동적이다.
- 출처 : https://yamalab.tistory.com/69?category=747907
- 다음으로 모델 기반의 알고리즘에 대해 알아보자. 모델 기반의 알고리즘 중, 가장 널리 사용되는 MF만을 살펴보겠다. MF는 유저나 상품의 정보를 나타내는 벡터를 PCA나 SVD같은 알고리즘으로 분해하거나 축소하는 방법이다. 즉 Matrix Factorization은, 유저를 행으로 하고 상품에 대한 평가를 열로 하는 matrix가 있다고 할 때 이를 두 개의 행렬로 분해하는 방법으로, 유저에 대한 latent와 상품에 대한 latent를 추출해내는 것에 그 목적이 있다고 하겠다.
latent는 각각의 유저에 대한 특징을 나타내는 vector로, 머신이 이해하는 방법과 개수대로 생성해낸 것이다. latent vector 간의 distance를 이용하여 유사한 유저나 상품을 추천하는 것에 활용할 수 있다. (latent의 rank를 학습하는 알고리즘과 사용자 지정해주는 알고리즘으로 나뉜다) PCA의 에이겐 벡터 + 에이겐 벨류와 비슷한 개념이라고 생각하면 된다. 아래의 그림(Andrew Ng의 강의 슬라이드 발췌)을 보면 직관적으로 이해가 될 것이다.
MF의 가장 대표적인 방법은 SVD(Singular Value Decomposition)이다. 특이값 분해라고 하는데, 고유값 분해처럼 행렬을 대각화하여 분해하는 방법 중 하나이다. 고유값 분해와 다른 점은 nXn의 정방행렬이 아니어도 분해가 가능하다는 것이고, 이는 Sparse한 특성을 가지는 추천 시스템에서의 Matrix를 분해하는 것에 안성맞춤이다.
U는 AAT를 고유값분해해서 얻어진 직교행렬로 U의 열벡터들을 A의 left singular vector라 부르고, V 의 열벡터들을 A의 right singular vector라 부른다. 또한 Σ는 AAT, ATA를 고유값분해해서 나오는 고유값들의 square root를 대각원소로 하는 m x n 직사각 대각행렬로, 그 대각원소들을 A의 특이값(singular value)이라 부른다. 즉 U, V는 특이 벡터를 나타낸 행렬이고 Σ 는 특이값에 대한 행렬이라고 할 수 있다.
Σ의 크기를 지정해 줌으로써 latent(의미 부여 벡터)의 크기를 지정해 줄 수도 있다. 이후 decomposition 된 행렬들을 이용하여 원래의 행렬 A와 사이즈가 동일한 행렬을 만들어내면, 비어있는 공간들을 유추해내는 새로운 행렬이 나타나는 것이다. 이를 Matrix Completion의 관점에서 보면, A 행렬에서 rating이 존재하는 데이터를 이용해 미지의 U, Σ, V를 학습하여 Completed 된 행렬 A`를 만들어내는 것이다.
SVD를 비롯한 MF에서 목적함수는, Predicted rating을 구하는 Matrix Completion의 경우, 최적의 해(Rating – Predicted Rating의 최소)가 없이 근사값을 추론하는 문제이다. 따라서 Gradient Descent 알고리즘, ALS(Alternating Least Square) 알고리즘 등으로, global minimum에 근접하는 thresh를 선정하여 이를 objective로 삼아 구하는 문제로 볼 수 있다. 일반적으로는 GD가 우수하지만, ALS는 병렬 처리 환경에서 좋은 성능을 보인다고 알려져 있다.
참고 : 학습이 완료된 후 user나 item에 대한 입력값의 행렬 연산 결과를 prediction을 할 때, 예상을 해야하는 결측값(Matrix에서 *으로 표기된 부분)의 초기값은 binary로 보정하거나 평균 혹은 중앙값으로 보정하기도 한다.
그래서 일반적으로 CF 기반의 추천 시스템을 구축할 때, 가장 많이 사용하는 알고리즘 스택은 SVD, 혹은 ALS를 기반으로 한 Hybrid 방법이 많다
# contents-based filtering
- 콘텐츠 기반 필터링은 말 그대로 콘텐츠에 대한 분석을 기반으로 추천하는 방식입니다.
영화 콘텐츠의 경우라면 스토리나 등장인물을, 상품이라면 상세 페이지의 상품 설명을 분석합니다. 
콘텐츠 기반 필터링의 장점은 많은 양의 사용자 행동 정보가 필요하지 않아 콜드 스타트 문제점이 없다는 것입니다. 아이템과 사용자 간의 행동을 분석하는 것이 아니라 콘텐츠 자체를 분석하기 때문입니다.
##  '메타 정보의 한정성'입니다.
- 상품의 프로파일을 모두 함축하는 데에 한계가 있다는 점입니다.
쉬운 설명을 위해 BTS의 또 다른 팬 T군이 있다고 가정해보겠습니다. T군은 BTS 중에서도 특히 정국이라는 멤버를 좋아하여, 그와 관련 있는 신문 기사를 주로 찾아보았습니다. 
하지만 기사 내용만으로 콘텐츠를 분류해야 하는 신문사 알고리즘 입장에서는, T군의 성향을 세부적으로 파악하기가 어렵습니다. BTS 전체의 기사 텍스트와 BTS 중 한 명의 멤버에 대한 기사 텍스트에서 큰 차이가 없기 때문입니다. 여기서 콘텐츠 기반 필터링의 정밀성이 떨어지는 문제가 발생합니다.
# hybrid recommender systems
- 하이브리드 추천 시스템은 협업 필터링과 콘텐츠 기반 필터링을 조합하여 상호 보완적으로 개발된 알고리즘입니다. 협업 필터링의 콜드 스타트 문제 해결을 위해 신규 콘텐츠는 콘텐츠 기반 필터링 기술로 분석하여 추천하고, 충분한 데이터가 쌓인 후부터는 협업 필터링으로 추천의 정확성을 높이는 방식입니다.
# machine learning recommender systems
- 머신러닝의 학습으로 추천하는 방식도 많이 개발되고 있습니다. 사용자에게 추천할 후보군을 먼저 보여주고 기계가 그에 대한 사용자 반응을 학습하며 점점 더 정교한 결과를 도출해내는 방식입니다.
## information utilization problem
- 다음으로, 영화 추천이나 맛집 추천 서비스가 아닌 대부분의 추천 서비스에서 나타나는 문제점인 Information Utilization Problem이 있다. 이는 추천 시스템 구축에 활용하기 위한 데이터, 정보들을 올바르게 활용하기 위한 고민에서 나온 문제점이라고 할 수 있다. 이를 이해하기 위해 e-commere에서의 추천 시스템에서 고객의 행동을 생각해보자. 대부분의 고객들은 상품을 눌러보고, 다른 상품도 살펴보고, 본인 기준에 마음에 든다면 장바구니에 넣어뒀다가 이를 구매하기도 한다. 이런 정보들을 Implicit Score(암묵 점수)라고 한다. 
왓챠나 넷플릭스 같은 영화 추천 서비스에서처럼, 사용자들은 아이템에 대한 명확한 평가를 내리지 않는다. 단지 상품을 눌러보고, 관심 표시를 하거나 구매 하는 정도이다. 상품에 대한 리뷰를 작성하거나 별점을 주는 고객은 극소수에 가깝다. 이러한 로그 데이터 속에 숨어있는 정보를 이용하는 고민이 필요하다. 하지만 이것이 쉽지 않기 때문에 Information Utilization Problem 이라고 부르는 것이다. 만약 고객의 구매 목록 데이터가 있을 때, 구매가 완료되었다고 과연 이 데이터가 상품에 대한 호감을 나타내는 데이터라고 할 수 있을까? 환불이나 교환이 일어났다면? 이 모든 것을 고려하여 데이터를 활용하기 용이한 Explicit Score(명시 점수 : 영화 평점에 대한 rating 같은 점수)처럼 데이터를 Utilization 하는 과정이 필요할 것이다.
그렇다고 Explicit Score가 항상 좋은 데이터인 것은 아니다. 대부분의 잘 정리된 명시 점수의 경우 Sparsity Problem을 심각하게 겪을 것이다.
## 분류
- 추천의 타입은 크게 3가지로 분류된다. 먼저 유저의 정보에 기반하여 자동으로 아이템 리스트를 추려주는 Personalized recommender(개인화 추천), 그리고 rating 기반의 인기 상품이나 현재 상품 기준 AR(Association Rule. 이하 AR) 순위 상품을 추천해주는 Non-personalized recommender 방법이 있다. 이 방법은 주로 Cold Start Problem(개인화 추천 모델링을 위한 유저정보 혹은 아이템 정보가 부족한 상황)이 발생하는 상황이나 개인화추천이 잘 적용되지 않는 추천 영역에 사용된다. 그리고 마지막으로 Attribute-based recommender 방법이 있다. 아이템 자체가 가지고 있는 정보, 즉 Contents 정보를 활용하여 추천하는 방법으로 Cold Start 문제를 해결하는 조금 더 세련된 방법이라고 할 수 있다. 뒤에 설명할 CF(Collaborative Filtering. 이하 CF)와 상호 보완적인 알고리즘인 Content-based approach 라고도 불린다.
위의 세 가지 타입에 매칭되는 대표적인 알고리즘은 Personalized recommender - CF, Non-personalized recommender - AR, Attribute-based recommender - Content based approach 라고 할 수 있다. 

# `implicit`
```python
conda install -c conda-forge implicit
```

# `annoy`
- Source: https://www.lfd.uci.edu/~gohlke/pythonlibs/#annoy
```python
!pip install "D:/annoy-1.17.0-cp38-cp38-win_amd64.whl"
```
- Source: https://github.com/spotify/annoy
- Tree Building
	```python
	from annoy import AnnoyIndex

	dim = 61
	# `metric`: (`"angular"`, `"euclidean"`, `"manhattan"`, `"hamming"`, `"dot"`)
	for i, value in enumerate(item_embs.values):
		tree.add_item(i, value)
	# Builds a forest of `n_trees` trees. More trees gives higher precision when querying. After calling `build()`, no more items can be added. `n_jobs` specifies the number of threads used to build the trees. `n_jobs=-1` uses all available CPU cores.
	tree.build(n_trees=20)
	```
- Item-Item Similarity Measure
	```python
	artist = "beyoncé"
	item_vec = item_embs.loc[artist].values

	res = tree.get_nns_by_vector(vector=item_vec, n=11, include_distances=True)
	display(pd.Series(res[1][1:], index=[id2name[i] for i in res[0][1:]]))
	```
- User-Item Similarity Measure
	```python
	user = 209
	user_vec = user_embs.loc[user].values

	res = tree.get_nns_by_vector(vector=user_vec, n=10, include_distances=True)
	display(pd.Series(res[1], index=[id2name[i] for i in res[0]]))
	```