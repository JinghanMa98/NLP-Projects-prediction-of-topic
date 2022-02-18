#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 19:52:15 2020
@author: pathou
"""

import praw
import pandas as pd
import pytz
from datetime import datetime
from my_functions import *
the_path = "/Users/majinghan/Desktop/Columbia/2021Fall/5067/HW/hw2/"

subreddit_channel = 'politics'##can change the channel

reddit = praw.Reddit(
     client_id="Waou7Kh3TjzC123PVWb2BA",
     client_secret="o0BU_YVHYKgHa-tET8S2w2H9gv6tIg",
     user_agent="testscript by u/fakebot3",
     username="JinghanMa",
     password="78901234A",
     check_for_async=False
 )

print(reddit.read_only)

def conv_time(var):
    tmp_df = pd.DataFrame()
    tmp_df = tmp_df.append(
        {'created_at': var},ignore_index=True)
    tmp_df.created_at = pd.to_datetime(
        tmp_df.created_at, unit='s').dt.tz_localize(
            'utc').dt.tz_convert('US/Eastern') 
    return datetime.fromtimestamp(var).astimezone(pytz.utc)

def get_reddit_data(var_in):
    import pandas as pd
    tmp_dict = pd.DataFrame()
    tmp_time = None
    try:
        tmp_dict = tmp_dict.append({"created_at": conv_time(
                                        var_in.created_utc)},
                                    ignore_index=True)
        tmp_time = tmp_dict.created_at[0] 
    except:
        print ("ERROR")
        pass
    tmp_dict = {'msg_id': str(var_in.id),
                'author': str(var_in.author),
                'body': var_in.body, 'datetime': tmp_time}
    return tmp_dict
    


    
def my_func(var, the_path_i):
    import numpy as np
    from my_functions import clean_text, stem_fun, rem_sw
    import pickle 
    #Step 1 call; up clean_text function
    clean_text = clean_text(var)
    clean_text = rem_sw(clean_text)
    clean_text = stem_fun(clean_text)
    pred, score = score_text(clean_text, the_path_i)
    return pred, score

for comment in reddit.subreddit(subreddit_channel).stream.comments():
    tmp_df = get_reddit_data(comment)
    the_pred, the_pred_proba = my_func(tmp_df["body"], the_path)
    print (the_pred[0], the_pred_proba)

