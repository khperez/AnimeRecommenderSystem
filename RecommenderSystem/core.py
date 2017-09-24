import pandas as pd
import csv

t_cols = ['user_id', 'work_id', 'rating']
training_set = pd.read_csv('../../data/train.csv', sep=',', names=t_cols, encoding='latin-1', header=0)
train_df = pd.DataFrame(training_set)
test_set = pd.read_csv('../../data/test.csv', sep=',', names=['user_id', 'work_id'], header=0)
test_df = pd.DataFrame(test_set)

class User:
	def __init__(self, id_num):
		self.user_id = id_num
		self.work_df = self.get_work_df()
		self.likes = self.get_like_count()
		self.dislikes = self.get_dislike_count()

	def get_work_df(self):
		work_df = train_df.copy()
		return work_df.loc[train_df['user_id'].isin([self.user_id])]

	def get_like_count(self):
		like_df = train_df.loc[lambda df: df.user_id == self.user_id, :]
		like_df = like_df.loc[lambda df: df.rating == 1, :]
		return len(like_df.index)

	def get_dislike_count(self):
		dislike_df = train_df.loc[lambda df: df.user_id == self.user_id, :]
		dislike_df = dislike_df.loc[lambda df: df.rating == 0, :]
		return len(dislike_df.index)	

	def similarity_with(self, user):
		df = pd.merge(self.work_df, user.work_df, how='inner', on='work_id')
		similar_df = df.loc[lambda df: df.rating_x == df.rating_y, :]
		dissimilar_df = df.loc[lambda df: df.rating_x != df.rating_y, :]
		if not similar_df.empty:
			agreements = len(similar_df.index)
		else:
			agreements = 0.0
		if not dissimilar_df.empty:
			disagreements = len(dissimilar_df.index)
		else:
			disagreements = 0.0
		
		total = (self.likes + self.dislikes) + (user.likes + user.dislikes)
	
		return (agreements - disagreements) / total

	def recommend(self, work):
		hive_mind_sum = 0.0
		rated_by = work.ratings_count

		for user in work.liked_by:
			hive_mind_sum += self.similarity_with(User(user))
		for user in work.disliked_by:
			hive_mind_sum -= self.similarity_with(User(user))		
		# Normalize hive mind sum
		hive_mind_sum = (hive_mind_sum + 1) / 2 
		print("Recommendation {} / Rated By {}".format(hive_mind_sum, rated_by))
		if rated_by:		
			recommendation = hive_mind_sum / rated_by
		else:
			recommendation = hive_mind_sum
		return recommendation			
		
class Work:
	def __init__(self, id_num):
		self.work_id = id_num
		self.like_count = self.get_like_count()
		self.dislike_count = self.get_dislike_count()
		self.ratings_count = self.like_count + self.dislike_count
		self.liked_by = self.get_users_liked()
		self.disliked_by = self.get_users_disliked()

	def get_like_count(self):
		like_df = train_df.loc[lambda df: (df.work_id == self.work_id), :]
		like_df = like_df.loc[lambda df: (df.rating == 1), :]
		return len(like_df.index)

	def get_dislike_count(self):
		dislike_df = train_df.loc[lambda df: (df.work_id == self.work_id), :]
		dislike_df = dislike_df.loc[lambda df: (df.rating == 0), :]
		return len(dislike_df.index)
	
	def get_users_liked(self):
		like_df = train_df.loc[lambda df: (df.work_id == self.work_id), :]
		like_df = like_df.loc[lambda df: (df.rating == 1), :]
		return like_df.loc[:, 'user_id'].as_matrix()

	def get_users_disliked(self):
		dislike_df = train_df.loc[lambda df: (df.work_id == self.work_id), :]
		dislike_df = dislike_df.loc[lambda df: (df.rating == 0), :]
		return dislike_df.loc[:, 'user_id'].as_matrix()

def main():
	result_list = []
	for index, row in test_df.iterrows():
		user = User(row['user_id'])
		work = Work(row['work_id'])
		recommendation = user.recommend(work)
		result_list.append(recommendation)
	results = pd.DataFrame({'prob_willsee': result_list})
	result_set = test_set.join(results)
	result_set.to_csv('submission_test.csv', index=False)

if __name__ == '__main__':
	main()
