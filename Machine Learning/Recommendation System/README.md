# Datasets
## MovieLens
## MovieLens 100k
```python
plt.hist(sorted(ratings.groupby(["user_id"]).size().values), bins=50);
```
## Last.fm
## Frappe
- Reference: https://arxiv.org/pdf/1505.03014v1.pdf
- 2015년 5월에 컨텍스트 인지 모바일 앱 추천 시스템을 위해 공개된 현실 데이터입니다. 유저의 앱에 대한 사용 카운트가 96,200여개의 로그로 구성되어 있습니다.
- `daytime`: `['morning', 'afternoon', 'evening', 'sunset', 'night', 'sunrise', 'noon']`
- `weekday`: `['sunday', 'friday', 'tuesday', 'wednesday', 'saturday', 'monday', 'thursday']`
- `isweekend`: `["workday", "weekend"]`
- `homework`: `['unknown', 'home', 'work']`
- `cost`: `['free', 'paid']`
- `weather`: `['sunny', 'cloudy', 'drizzle', 'rainy', 'unknown', 'foggy', 'stormy', 'snowy', 'sleet']`

# Categories of Recommendation System
- Content-Based Filtering
- Collaborative Filtering (CF)
	- Memory-Based
		- Association Rule Learning (= Association Analysis)
	- Model-Based
		- Matrix Factorization (MF)
			- BPR (Bayesian Personalizaed Ranking)
			- ALS (Alternating Least Squares)
			- Logistic Matrix Factorization
		- Factorization Machine (FM)
		- Context-Aware Recommender System (CARS)
			- DeepFM
			
# Memory-Based & Model-Based
## Memory-Based
- *Memory-based techniques rely heavily on simple similarity measures (Cosine similarity, Pearson correlation, Jaccard coefficient… etc) to match similar people or items together. If we have a huge matrix with users on one dimension and items on the other, with the cells containing votes or likes, then memory-based techniques use similarity measures on two vectors (rows or columns) of such a matrix to generate a number representing similarity.*
## Model-Based
- *Model-based techniques on the other hand try to further fill out this matrix. They tackle the task of "guessing" how much a user will like an item that they did not encounter before. For that they utilize several machine learning algorithms to train on the vector of items for a specific user, then they can build a model that can predict the user’s rating for a new item that has just been added to the system.*

# Explicit & Implicit Data
- Source: https://blog.mirumee.com/the-difference-between-implicit-and-explicit-data-for-business-351f70ff3fbf
## Explicit Data
- *Explicit data is much harder to collect.* Let’s look at Spotify. Simply listening to a song is not explicit data in itself. The system does not know for sure that the user likes that song. Actual explicit data is when the user adds a specific tune to a playlist or hits the heart icon to say that they enjoy listening to it. In such cases, there is exponentially more implicit than explicit data being created by user activity.
- *Explicit data can also be shallow.( Users may be asked to give binary reactions: like or dislike, thumbs up or thumbs down. Even when a site like IMDB allows for ratings from 1 to 10, human nature means that people tend to rate in the extremes. Users regularly rate everything as 10 or 1; not many people take the time to leave a 4-out-of-10 rating because they clearly didn’t have a strong opinion in the first place.
## Implicit Data
- *With implicit data, we sometimes need to observe what the user does next.* If someone listens to a single song, we cannot know if they liked that artist. The system needs to store that information and see what happens in future. If the user then purchases an album a few days later, that second action backs up the initial assumption. The system can then learn how that single user or all users interact with the system and make better assumptions as time passes and more data is generated.

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
	
# Negative Sampling
- Source: https://medium.com/mlearning-ai/overview-negative-sampling-on-recommendation-systems-230a051c6cd7#:~:text=In%20the%20negative%20sampling%20of,interacted%20are%20all%20negative%20examples.
```python
def negative_sampling(df, ratio=3):
    neg_sampling = pd.concat([df]*ratio)
    neg_sampling["item"] = df["item"].sample(n=len(neg_sampling), replace=True).values
    # "네거티브 데이터"는 카운트가 존재하지 않음
    neg_sampling["cnt"] = 0

    sampling = pd.concat([df, neg_sampling])
    sampling = sampling.drop_duplicates(sampling.columns.drop("cnt"))
    return sampling
```
		
# Association Rule Learning (= Association Analysis)
- Source: https://en.wikipedia.org/wiki/Association_rule_learning, https://livebook.manning.com/book/machine-learning-in-action/chapter-11/51
- Association rule learning is a rule-based machine learning method for discovering interesting relations between variables in large databases. It is intended to identify strong rules discovered in databases using some measures of interestingness. In any given transaction with a variety of items, association rules are meant to discover the rules that determine how or why certain items are connected.
- For example, the rule {onions, potatoes} -> {burger} found in the sales data of a supermarket would indicate that if a customer buys onions and potatoes together, they are likely to also buy hamburger meat. Such information can be used as the basis for decisions about marketing activities such as, e.g., promotional pricing or product placements.
- In contrast with sequence mining, *association rule learning typically does not consider the order of items either within a transaction or across transactions.*
- Association rules are made by searching data for frequent if-then patterns and by using a certain criterion under Support and Confidence to define what the most important relationships are. *Support is the evidence of how frequent an item appears in the data given, as Confidence is defined by how many times the if-then statements are found true. However, there is a third criteria that can be used, it is called Lift and it can be used to compare the expected Confidence and the actual Confidence. Lift will show how many times the if-then statement is expected to be found to be true.
- *Looking at items commonly purchased together can give stores an idea of customers’ purchasing behavior.* This knowledge, extracted from the sea of data, can be used for pricing, marketing promotions, inventory management, and so on. Looking for hidden relationships in large datasets is known as association analysis. Brute-force solutions aren’t capable of solving this problem, so a more intelligent approach is required to find frequent itemsets in a reasonable amount of time.
- Association analysis is the task of finding interesting relationships in large datasets. These interesting relationships can take two forms: frequent item sets or association rules. *Frequent item sets are a collection of items that frequently occur together.* The second way to view interesting relationships is association rules. *Association rules suggest that a strong relationship exists between two items.*
- From the dataset we can also find an association rule such as diapers → wine. *This means that if someone buys diapers, there’s a good chance they’ll buy wine.* With the frequent item sets and association rules, retailers have a much better understanding of their customers. Although common examples of association analysis are from the retail industry, it can be applied to a number of other industries, such as website traffic analysis and medicine.
- Support
	- (Number of transactions containing the itemset)/(Total number of transactions)
	- Support is an indication of how frequently the itemset appears in the dataset.
	- When using antecedents and consequents, it allows a data miner to determine the support of multiple items being bought together in comparison to the whole data set. For example, Table 2 shows that if milk is bought, then bread is bought has a support of 0.4 or 40%. This because in 2 out 5 of the transactions, milk as well as bread are bought. In smaller data sets like this example, it is harder to see a strong correlation when there are few samples, but when the data set grows larger, support can be used to find correlation between two or more products in the supermarket example.
	- Minimum support thresholds are useful for determining which itemsets are preferred or interesting.
	- *Minimum threshold is used to remove samples where there is not a strong enough support or confidence to deem the sample as important or interesting in the dataset.*
- Confidence
	- (Number of transactions containing X and Y)/(Number of transactions containing X)
	- Confidence is the percentage of all transactions satisfying X that also satisfy Y. The confidence value of an association rule, often denoted as X -> Y, is the ratio of transactions containing both X and Y to the total amount of X values present, where X is the antecedent and Y is the consequent.
- Lift
	```python
	def lift(x, y):
		return confidence(x, y)/support(y)
	```
	- ***If the rule had a lift of 1, it would imply that the probability of occurrence of the antecedent and that of the consequent are independent of each other. When two events are independent of each other, no rule can be drawn involving those two events.***
	- ***If the lift is > 1, that lets us know the degree to which those two occurrences are dependent on one another, and makes those rules potentially useful for predicting the consequent in future data sets. 전체 사용자들 중 아이템 y를 좋아하는 비율(Support)보다 아이템 x를 좋아하는 사용자들 중 아이템 y도 좋아하는 비율(Confidence)가 매우 크다면 아이템 x와 y는 매우 연관 관계가 높을 것입니다.***
	- ***If the lift is < 1, that lets us know the items are substitute to each other. This means that presence of one item has negative effect on presence of other item and vice versa. 전체 사용자들 중 아이템 y를 좋아하는 비율(support)보다 아이템 x를 좋아하는 사용자들 중 아이템 y도 좋아하는 비율(confidence)이 매우 작다면 아이템 x와 y는 매우 연관 관계가 낮을 것입니다.***
	- 즉 어떤 사용자가 아이템 x를 좋아한다는 사실이 아이템 y를 좋아할 가능성이 높인다는 의미입니다.
## Apriori Algorithm
- Many algorithms for generating association rules have been proposed. *Some well-known algorithms are Apriori, Eclat and FP-Growth, but they only do half the job, since they are algorithms for mining frequent itemsets (Lists of items that commonly appear together). Another step needs to be done after to generate rules from frequent itemsets found in a database.*
- Apriori is for frequent itemset mining and association rule learning. *It proceeds by identifying the frequent individual items in the database and extending them to larger and larger item sets as long as those itemsets appear sufficiently often.*
- Apriori uses a "bottom up" approach, where *frequent subsets are extended one item at a time (a step known as candidate generation), and groups of candidates are tested against the data. The algorithm terminates when no further successful extensions are found.* Apriori uses breadth-first search and a Hash tree structure to count candidate item sets efficiently. It generates candidate item sets of length `max_len`. Then it prunes the candidates which have an infrequent sub pattern.
- Source: http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
- Apriori is a popular algorithm for extracting frequent itemsets with applications in association rule learning. The apriori algorithm has been designed to operate on databases containing transactions, such as purchases by customers of a store. An itemset is considered as "frequent" if it meets a user-specified support threshold. For instance, if the support threshold is set to 0.5 (50%), a frequent itemset is defined as a set of items that occur together in at least 50% of all transactions in the database.
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
	# `max_len`: (default `None`) Maximum length of the itemsets generated.
	# [`low_memory`]: Note that while `True` should only be used for large dataset if memory resources are limited, because this implementation is approx. 3-6x slower than the default.
	freq_sets = apriori(user_likes_ohe, min_support=0.01, max_len=2, use_colnames=True, verbose=1)
			
	# `df`: DataFrame of frequent itemsets with columns `['support', 'itemsets']`.
	# `metric` (`"support"`, `"confidence"`, `"lift"`) Metric to evaluate if a rule is of interest.
	asso_rules = association_rules(freq_sets, metric="confidence", min_threshold=0.01)
	# Lift를 기준으로 추천합니다.
	asso_rules = asso_rules.sort_values("lift", ascending=False)
	asso_rules = asso_rules.drop(["leverage", "conviction"], axis=1)
	```

# Content-Based Filtering
- Source: https://www.mygreatlearning.com/blog/matrix-factorization-explained/, https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)
- *This approach recommends items based on user preferences. It matches the requirement, considering the past actions of the user, patterns detected, or any explicit feedback provided by the user, and accordingly, makes a recommendation.* Example: If you prefer the chocolate flavor and purchase a chocolate ice cream, the next time you raise a query, the system shall scan for options related to chocolate, and then, recommend you to try a chocolate cake.
- *The model can capture specific interests of a user, and can recommend niche items that very few other users are interested in.*
- *Since the feature representation of the items are hand-engineered to some extent, this technique requires a lot of domain knowledge. Therefore, the model can only be as good as the hand-engineered features.*
- The model can only make recommendations based on existing interests of the user. In other words, the model has limited ability to expand on the users’ existing interests.
- ***Content-based filtering algorithms, on the other hand, are in theory much less prone to the new item problem. Since content based recommenders choose which items to recommend based on the feature the items possess, even if no interaction for a new item exist, still its features will allow for a recommendation to be made.*** Consider the case of so-called editorial features (e.g. director, cast, title, year), those are always known when the item, in this case movie, is added to the catalogue. However, other kinds of attributes might not be e.g. features extracted from user reviews and tags. *Content-based algorithms relying on user provided features suffer from the cold-start item problem as well.

# Collaborative Filtering (CF)
- Source: https://www.mygreatlearning.com/blog/matrix-factorization-explained/, https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)
- *This approach uses similarities between users and items simultaneously, to provide recommendations. It is the idea of recommending an item or making a prediction, depending on other like-minded individuals.* Example: Suppose Persons A and B both like the chocolate flavor and have tried the ice-cream and cake, then if Person A buys chocolate biscuits, the system will recommend chocolate biscuits to Person B.
- In collaborative filtering, *we do not have the rating of individual preferences and item preferences. We only have the overall rating, given by the individuals for each item. As usual, the data is sparse, implying that the person either does not know about the item or is not under the consideration list for buying the item, or has forgotten to give the rating.*
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

# BPR (Bayesian Personalizaed Ranking)
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
	input_user = Input(shape=(), name="Input_user")
	input_pos = Input(shape=(), name="Input_pos")
	input_neg = Input(shape=(), name="Input_neg")
	inputs = [input_user, input_pos, input_neg]

	n_users = ratings["user_id"].max() + 1
	n_items = ratings["movie_id"].max() + 1
	dim = 64
	embedding_user = Embedding(input_dim=n_users, output_dim=dim, name="Embedding_user")
	embedding_item = Embedding(input_dim=n_items, output_dim=dim, name="Embedding_item")

	z1 = embedding_user(input_user)
	z2 = embedding_item(input_pos)
	z3 = embedding_item(input_neg)

	pos_score = Dot(axes=(1, 1))([z1, z2])
	neg_score = Dot(axes=(1, 1))([z1, z3])
	diff = pos_score - neg_score
	outputs = sigmoid(diff)

	model = Model(inputs=inputs, outputs=outputs)

	model.compile(optimizer=Adagrad(), loss="binary_crossentropy", metrics=["binary_accuracy"])

	X = [likes_tar_tr["user_id"].values, likes_tar_tr["movie_id"].values, likes_tar_tr["neg_movie_id"].values]
	y = np.ones(shape=(len(likes_tar_tr), 1))

	es = EarlyStopping(monitor="val_loss", mode="auto", verbose=1, patience=2)
	model_path = "./movielens_bpr_tf.h5"
	mc = ModelCheckpoint(filepath=model_path, monitor="val_binary_accuracy", mode="auto", verbose=1, save_best_only=True)
	model.fit(x=X, y=y, validation_split=0.2, batch_size=128, epochs=32, verbose=2, callbacks=[es, mc])
	```

# Factorization Machine (FM)
- Sources: https://zzaebok.github.io/machine_learning/factorization_machines/, https://greeksharifa.github.io/machine_learning/2019/12/21/FM/
- 반면 Factorization Machine은 이에 인구통계학적 정보(Demographic data)나 과거 이력 등 user의 특성 데이터와 상품에 대한 설명, 이미지, 인지도 등 상품의 특성 데이터를 모두 활용하여 추천 시스템을 만드는 것이 가능합니다.
- matrix factorization에서 row index는 각 user를, column index는 각 item을 나타내고 각 value rating을 의미했습니다.
- 그러나 factorization machine에서는 일반적인 regression과 본질적으로 같은 방식으로 row index는 각 sample을, columns index는 각 feature를 나타내며 rating을 나타내는 별도의 column이 존재합니다.
- 따라서 rating 값을 target으로 하는 regression이기 때문에 metadata와 같은 여러 feature를 추가할 수 있다는 장점이 있습니다.
- 나이라는 특성과 성별이라는 특성의 상호 작용을 고려하여, 나이가 20대 후반일때의 학생과 50대 초반일때의 학생의 예상 평점을 예측해보면 다를 수 있다는 것입니다.
- Dimension of embedding vectors를 매우 큰 값으로 잡으면 FM 모델은 가능한 모든 상호작용 관계에 대한 가중치를 표현하는 임베딩 행렬을 만들 것입니다. 하지만 복잡한 상호작용 관계를 추정하기에 데이터가 충분히 크지 않다면 작은 수준으로 제한해줘야 보다 일반화가 가능한 안정적인 FM모델을 만들 수 있습니다.
- 각 피쳐의 임베딩을 FM모델과 DeepLearning 모델이 공유하고 있다.
- 피쳐 엔지니어링에 들어가는 수고를 많이 덜어낼 수 있습니다.
- 미리 학습(Pre-trained)하지 않고 한번의 Input으로 End-to-end로 진행한다.
- 다른 모델의 경우 FM으로 미리 한번 학습된 Embedding으로 Deeplearning모델을 학습하는 경우가 있습니다. 한번에 모델을 학습시킬 수 있으므로, 학습의 어려움이 적습니다.
```python
k = 64
inputs = list()
embedding_mlr_list = list()
embedding_fm_list = list()
for col in cols:
    input_ = Input(shape=(), name=f"Input_{col}")
    inputs.append(input_)
    
    input_dim = data[col].nunique()
    z1 = Embedding(input_dim=input_dim, output_dim=1, name=f"Embedding_1_{col}")(input_)
    embedding_mlr_list.append(z1)

    logits_mlr
    z2 = Embedding(input_dim=input_dim, output_dim=k, name=f"Embedding_{k}_{col}")(input_)
    embedding_fm_list.append(z2)
# Multivariate Linear Regression Part
logits_mlr = tf.Variable((0.,)) + tf.math.add_n(embedding_mlr_list)

# Factorization Machine Part
z = tf.stack(embedding_fm_list, axis=1)
logits_fm = 1/2*tf.math.reduce_sum(tf.math.square(tf.math.reduce_sum(z, axis=1)) - tf.math.reduce_sum(tf.math.square(z), axis=1), axis=1, keepdims=True)

model = Model(inputs=inputs, outputs=logits_mlr + logits_fm, name="movielens100k_fm")

model.compile(optimizer=Adagrad(lr=0.05), loss="mse", metrics=["mse"])
model.summary()
```

# Context-Aware Recommender System (CARS)
- Source: https://www.sigapp.org/sac/sac2016/T6.pdf
- Context is defined as "any information that can be used to characterize the situation of an entity".
- *Context-aware recommender system
(CARS) is one RS trying to adapt their recommendations to users' specific contextual situations, since users usually make different decisions in different situations. For example, users may choose a romantic movie to watch with partner, but probably a cartoon if he or she is going to watch
it with kids. Companion, either partner or kid, in this example, is one influential context factor. Other examples of the contexts could be time, location, weather, and so forth. Due to that users' preferences and decisions vary from situations to situations, it is necessary to take context into consideration when providing recommendations to the end users.*
- There are two typical recommendation tasks involved when context is taken into account: one is context-aware recommendation (CAR) and another one is context recommendation (CR). *The topics in CAR are focused on how to build context-aware recommendation algorithms to recommend items to users in a specific situations. For example, which restaurant I should choose if I am going to have a formal business dinner with a company director. By contrast, context recommendation is a novel research direction emerged in recent years, where it aims to suggest appropriate contexts for the users to consume the item. For example, which could be the best contexts for me to watch the movie "Titanic"? Potential answers could be seeing it in a theater with your partner at weekend.*

# DeepFM
- Paper: https://arxiv.org/pdf/1703.04247.pdf
- Source: https://greeksharifa.github.io/machine_learning/2020/04/07/DeepFM/
- 모델에 대해 설명할 것이다. 이 DeepFM이라는 모델은 FM과 딥러닝을 결합한 것이다. 최근(2017년 기준) 구글에서 발표한 Wide & Deep model에 비해 피쳐 엔지니어링이 필요 없고, wide하고 deep한 부분에서 공통된 Input을 가진다는 점이 특징적이다.
- DeepFM은 피쳐 엔지니어링 없이 end-to-end 학습을 진행할 수 있다. 저차원의 interaction들은 FM 구조를 통해 모델화하고, 고차원의 interaction들은 DNN을 통해 모델화한다.
- DeepFM은 같은 Input과 Embedding 벡터를 공유하기 때문에 효과적으로 학습을 진행할 수 있다.
- Source: https://orill.tistory.com/category/RecSys
- 최근 구글에서 발표한 Wide & Deep Model과 비교해보면, DeepFM은 Wide 부분과 Deep 부분이 공통된 입력(shared input)을 받고 Feature Engineering이 필요하지 않다.
- 낮은 차원, 높은 차원의 상호 작용 둘 다를 모델링하기 위해 [Cheng et al., 2016]은 linear("wide") 모델과 deep 모델을 결합한 흥미로운 혼합 네트워크 구조(Wide & Deep)를 제시했다. 이 모델에서는 두 개의 다른 inputs이 wide 부분과 deep 부분을 위해 각각 필요하다. "wide part"의 입력은 여전히 전문가의 feature engineering이 필요하다.
- 이 모델은 FM을 통해 낮은 차원의 피쳐 상호작용을 학습하고 DNN을 통해서는 높은 차원의 피쳐 상호작용을 학습한다. Wide & Deep 모델과는 다르게 DeepFM은 end-to-end로 feature engineering이 필요 없이 학습할 수 있다.
- DeepFM은 Wide & Deep과는 다르게 같은 input과 embedding layer를 공유하기 때문에 효율적으로 학습할 수 있다. Wide & Deep 에서는 input vector가 직접 고안한 pairwise 피쳐 상호작용을 wide part에 포함하기 때문에 입력 vector가 굉장히 커질 수 있고 이는 복잡도를 굉장히 증가시킨다.
- 1) 다른 field의 input vector의 길이는 다를 수 있지만 embedding은 같은 크기(k)이다. (역자 예시: gender field는 보통 length가 남, 여 2인 반면 국적이나 나이 field의 길이는 더 길다. 하지만 embedding시에는 똑같이 k=5차원 벡터로 임베딩 된다.)
```python
inputs = dict()
embedding_mlr_list = list()
embedding_fm_list = list()
for col in data.columns:
    if col != "cnt":
        input_dim = data[col].nunique()
        z = Input(shape=(), name=f"Input_{col}")
        inputs[col] = z

        # Order-1 Part
        z1 = Embedding(input_dim=input_dim, output_dim=1, name=f"Embedding_mlr_{col}")(z)
        embedding_mlr_list.append(z1)
        
        # Order-2 Part
        output_dim = 8
        z2 = Embedding(input_dim=input_dim, output_dim=output_dim, name=f"Embedding_fm_{col}")(z)
        embedding_fm_list.append(z2)
logits_mlr = tf.Variable((0.,)) + tf.math.add_n(embedding_mlr_list)

# Order-2 Part
z = tf.stack(embedding_fm_list, axis=1)
logits_fm = 1/2*tf.math.reduce_sum(tf.math.square(tf.math.reduce_sum(z, axis=1)) - tf.math.reduce_sum(tf.math.square(z), axis=1), axis=1, keepdims=True)

# Order-3 Part
z = Concatenate(axis=1)(embedding_fm_list)
z = Dense(units=100, activation="relu")(z)
z = Dropout(rate=0.5)(z)
z = Dense(units=100, activation="relu")(z)
z = Dropout(rate=0.5)(z)
z = Dense(units=100, activation="relu")(z)
z = Dropout(rate=0.5)(z)
logits_mlp = Dense(units=1)(z)

z = Add()([logits_mlr, logits_fm, logits_mlp])
logits = sigmoid(z)
model = Model(inputs=inputs, outputs=logits, name="deepfm")
```

# Multi-Armed Bandit (Problem)
- Source: https://en.wikipedia.org/wiki/Multi-armed_bandit
- In probability theory and machine learning, *the multi-armed bandit problem is a problem in which a fixed limited set of resources must be allocated between competing (alternative) choices in a way that maximizes their expected gain, when each choice's properties are only partially known at the time of allocation, and may become better understood as time passes or by allocating resources to the choice.* This is a classic reinforcement learning problem that exemplifies the exploration–exploitation tradeoff dilemma. *The name comes from imagining a gambler at a row of slot machines (sometimes known as "one-armed bandits"), who has to decide which machines to play, how many times to play each machine and in which order to play them, and whether to continue with the current machine or try a different machine.*
- *In the problem, each machine provides a random reward from a probability distribution specific to that machine, that is not known a-priori. The objective of the gambler is to maximize the sum of rewards earned through a sequence of lever pulls.* ***The crucial tradeoff the gambler faces at each trial is between "exploitation" of the machine that has the highest expected payoff and "exploration" to get more information about the expected payoffs of the other machines.***
## Thompson Sampling
- Source: https://en.wikipedia.org/wiki/Thompson_sampling
Thompson sampling is a heuristic for choosing actions that addresses the exploration-exploitation dilemma in the multi-armed bandit problem. It consists of choosing the action that maximizes the expected reward with respect to a randomly drawn belief.
- 다양한 추천 시스템 모델을 동시에 사용할 때, 추천을 제공하는 매 순간마다 어떤 모델을 사용하는 것이 적절한지를 알려주는 알고리즘입니다.

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
## MAP (Mean Average Precision)
- Source: https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54
## nDCG (normalized Discounted Cumulative Gain)
- Source: https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54
## Entropy Diversity

# Redis
- sources: https://medium.com/@jyejye9201/%EB%A0%88%EB%94%94%EC%8A%A4-redis-%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80-2b7af75fa818, https://hwigyeom.ntils.com/entry/Windows-%EC%97%90-Redis-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0-1
- REDIS(REmote Dictionary Server)는 메모리 기반의 “키-값” 구조 데이터 관리 시스템이며, 모든 데이터를 메모리에 저장하고 조회하기에 빠른 Read, Write 속도를 보장하는 비 관계형 데이터베이스이다.
- 레디스는 크게 5가지< String, Set, Sorted Set, Hash, List >의 데이터 형식을 지원한다.
- Redis는 빠른 오픈 소스 인 메모리 키-값 데이터 구조 스토어이며, 다양한 인 메모리 데이터 구조 집합을 제공하므로 사용자 정의 애플리케이션을 손쉽게 생성할 수 있다.

# Install `implicit`
```python
conda install -c conda-forge implicit
```