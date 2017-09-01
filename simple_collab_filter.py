# Experimentation with user-user based collaborative filtering
# No machine learning libraries
# Data set from Mangaki challenge

import pandas as pd
import random

def get_user_work_list(df, user_id):
	work_df = df.copy()
	return work_df.loc[df['user_id'].isin(user_id)]

def choose_two_rand_users(df):
	max_num_users = len(df.index)
	user_A_index = random.randrange(0, max_num_users)
	user_B_index = random.randrange(0, max_num_users)
	user_A = df.loc[user_A_index, 'user_id']
	user_B = df.loc[user_B_index, 'user_id']
	return [user_A, user_B]	

# Get user "will watch" list
def get_user_will_watch(df, user):	
	user_work_df = get_user_work_list(df, user)
	return user_work_df[user_work_df['rating'].isin([1])]

def get_user_will_not_watch(df, user):
	user_work_df = get_user_work_list(df, user)
	return user_work_df[user_work_df['rating'].isin([0])]

def get_common_works_list(userA_df, userB_df):
	return pd.merge(userA_df, userB_df, how='inner')


def main():
	t_cols = ['user_id', 'work_id', 'rating']
	training_set = pd.read_csv('./data/train.csv', sep=',', names=t_cols, encoding='latin-1', header=0)
	train_df = pd.DataFrame(training_set)
	rand_users = choose_two_rand_users(train_df)
	userA = rand_users[0]
	userB = rand_users[1]
	userA_works_list = get_user_work_list(train_df, [userA])
	userB_works_list = get_user_work_list(train_df, [userB])
	print(get_common_works_list(userA_works_list, userB_works_list))	
			

if __name__ == '__main__':
	main()
