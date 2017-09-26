import pandas as pd
import csv
from scipy.stats import pearsonr
from sklearn.metrics import jaccard_similarity_score
import math

t_cols = ['user_id', 'work_id', 'rating']
training_set = pd.read_csv('../../data/train.csv', sep=',', names=t_cols, encoding='latin-1', header=0)
train_df = pd.DataFrame(training_set)
test_set = pd.read_csv('../../data/test.csv', sep=',', names=['user_id', 'work_id'], header=0)
test_df = pd.DataFrame(test_set)
watched_set = pd.read_csv('../../data/watched.csv', sep=',', names=t_cols, header=0)
watched_df = pd.DataFrame(watched_set)

class User:
        def __init__(self, id_num):
                self.user_id = id_num
                self.work_df = self.get_work_df()
                self.likes = self.get_like_count()
                self.dislikes = self.get_dislike_count()
                self.loves = self.get_love_count()
                self.neutral = self.get_neutral_count()
                self.watched_df = self.get_watched_df()

        def get_work_df(self):
                return train_df.loc[train_df['user_id'].isin([self.user_id])]

        def get_watched_df(self):
                return watched_df.loc[watched_df['user_id'].isin([self.user_id])]

        def get_like_count(self):
                like_df = watched_df.loc[lambda df: df.user_id == self.user_id, :]
                like_df = like_df.loc[lambda df: df.rating == 'like', :]
                return len(like_df.index)

        def get_dislike_count(self):
                dislike_df = watched_df.loc[lambda df: df.user_id == self.user_id, :]
                dislike_df = dislike_df.loc[lambda df: df.rating == 'dislike', :]
                return len(dislike_df.index)    

        def get_neutral_count(self):
                neutral_df = watched_df.loc[lambda df: df.user_id == self.user_id, :]
                neutral_df = neutral_df.loc[lambda df: df.rating == 'neutral', :]
                return len(neutral_df.index)

        def get_love_count(self):
                love_df = watched_df.loc[lambda df: df.user_id == self.user_id, :]
                love_df = love_df.loc[lambda df: df.rating == 'love', :]
                return len(love_df.index)

        def similarity_with(self, user):
                df = pd.merge(self.watched_df, user.watched_df, how='inner', on='work_id')
                df = df.replace(to_replace={'rating_x': {'dislike': 0, 'neutral': 1, 'like': 2, 'love': 3}})
                df = df.replace(to_replace={'rating_y': {'dislike': 0, 'neutral': 1, 'like': 2, 'love': 3}})
                #print("Pearson: {}".format(pearsonr(df.rating_x.values, df.rating_y.values)))
                #print("Jaccard: {}".format(jaccard_similarity_score(df.rating_x.values, df.rating_y.values)))
                if not df.rating_x.values.all():
                    similarity_score = jaccard_similarity_score(df.rating_x.values, df.rating_y.values)
                else:
                    similarity_score = 0.0
                return similarity_score 

        def recommend(self, work):
                hive_mind_sum = 0.0
                rated_by = work.ratings_count

                for user in work.liked_by:
                        hive_mind_sum += self.similarity_with(User(user))
                for user in work.loved_by:
                        hive_mind_sum += self.similarity_with(User(user))
                for user in work.disliked_by:
                        hive_mind_sum -= self.similarity_with(User(user))
                for user in work.neutral_by:
                        hive_mind_sum -= self.similarity_with(User(user))
                if rated_by:            
                        recommendation = hive_mind_sum / rated_by 
                else:
                        recommendation = hive_mind_sum
                recommendation = (recommendation + 1) / 2
                print("Recommendation {} / Rated By {}".format(recommendation, rated_by))
                return recommendation                   
                
class Work:
        def __init__(self, id_num):
                self.work_id = id_num
                self.like_count = self.get_like_count()
                self.dislike_count = self.get_dislike_count()
                self.neutral_count = self.get_neutral_count()
                self.love_count = self.get_love_count()
                self.ratings_count = self.like_count + self.dislike_count + self.love_count + self.neutral_count
                self.liked_by = self.get_users_liked()
                self.disliked_by = self.get_users_disliked()
                self.neutral_by = self.get_users_neutral()
                self.loved_by = self.get_users_loved()

        def get_like_count(self):
                like_df = watched_df.loc[lambda df: (df.work_id == self.work_id), :]
                like_df = like_df.loc[lambda df: (df.rating == 'like'), :]
                return len(like_df.index)

        def get_dislike_count(self):
                dislike_df = watched_df.loc[lambda df: (df.work_id == self.work_id), :]
                dislike_df = dislike_df.loc[lambda df: (df.rating == 'dislike'), :]
                return len(dislike_df.index)
            
        def get_neutral_count(self):
                neutral_df = watched_df.loc[lambda df: (df.work_id == self.work_id), :]
                neutral_df = neutral_df.loc[lambda df: (df.rating == 'neutral'), :]
                return len(neutral_df.index)

        def get_love_count(self):
                love_df = watched_df.loc[lambda df: (df.work_id == self.work_id), :]
                love_df = love_df.loc[lambda df: (df.rating == 'love'), :]
                return len(love_df.index)
        
        def get_users_liked(self):
                like_df = watched_df.loc[lambda df: (df.work_id == self.work_id), :]
                like_df = like_df.loc[lambda df: (df.rating == 'like'), :]
                return like_df.loc[:, 'user_id'].as_matrix()

        def get_users_disliked(self):
                dislike_df = watched_df.loc[lambda df: (df.work_id == self.work_id), :]
                dislike_df = dislike_df.loc[lambda df: (df.rating == 'dislike'), :]
                return dislike_df.loc[:, 'user_id'].as_matrix()
        
        def get_users_neutral(self):
                neutral_df = watched_df.loc[lambda df: (df.work_id == self.work_id), :]
                neutral_df = neutral_df.loc[lambda df: (df.rating == 'neutral'), :]
                return neutral_df.loc[:, 'user_id'].as_matrix()

        def get_users_loved(self):
                loved_df = watched_df.loc[lambda df: (df.work_id == self.work_id), :]
                loved_df = loved_df.loc[lambda df: (df.rating == 'love'), :]
                return loved_df.loc[:, 'user_id'].as_matrix()


def main():
        result_list = []
        for index, row in test_df.iterrows():
                user = User(row['user_id'])
                work = Work(row['work_id'])
                print("Reccomendation for {} on work {}".format(user.user_id, work.work_id))
                recommendation = user.recommend(work)
                result_list.append(recommendation)
        results = pd.DataFrame({'prob_willsee': result_list})
        result_set = test_set.join(results)
        result_set.to_csv('submission_test.csv', index=False)

if __name__ == '__main__':
        main()
