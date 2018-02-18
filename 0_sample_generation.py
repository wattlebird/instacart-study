# -*- coding: utf-8 -*-
"""
Generating sample data for fast testing
"""
import pandas as pd
import numpy as np
from setting import *

priors = pd.read_csv(DATA + '/order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8}, engine='c')

orders = pd.read_csv(DATA + '/orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32}, engine='c')

small_user_id = np.random.choice(orders.user_id.unique(), 10000, False)
small_orders = orders[orders.user_id.isin(small_user_id)].reset_index(drop=True)
small_orders.to_csv(DATA+"/orders.small.csv", index=False)

small_prior_log=priors[priors.order_id.isin(small_orders.order_id)].reset_index(drop=True)
small_prior_log.to_csv(DATA+'/log.prior.small.csv', index=False)

train=pd.read_csv(DATA + '/order_products__train.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8}, engine='c')
small_train_log=train[train.order_id.isin(small_orders.order_id)].reset_index(drop=True)
small_train_log.to_csv(DATA+'/log.train.small.csv', index=False)

test=pd.read_csv(DATA + '/sample_submission.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16}, engine='c')
test[test.order_id.isin(small_orders.order_id)].reset_index(drop=True)
test.to_csv(DATA+'/sample_submission.small.csv', index=False)