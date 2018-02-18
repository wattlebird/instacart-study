# -*- coding: utf-8 -*-
"""
Generating labeled data for training/testing process.
"""
import pandas as pd
import numpy as np
from setting import *

orders = pd.read_csv(DATA + '/orders.small.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32}, usecols=['order_id', 'user_id', 'eval_set'], engine='c')

log=pd.concat([
        pd.read_csv(DATA + '/log.train.small.csv', dtype={
                'order_id': np.int32,
                'product_id': np.uint16,
                'add_to_cart_order': np.int16,
                'reordered': np.int8}, usecols=['order_id', 'product_id', 'reordered'], engine='c'),
        pd.read_csv(DATA + '/log.prior.small.csv', dtype={
                'order_id': np.int32,
                'product_id': np.uint16,
                'add_to_cart_order': np.int16,
                'reordered': np.int8}, usecols=['order_id', 'product_id', 'reordered'], engine='c')],
    ignore_index=True)
log.sort_values(by=['order_id', 'product_id'], inplace=True)
log = log.reset_index(drop=True)

train_order = orders[orders.eval_set=='train'][['user_id', 'order_id']]
train_prior_order = orders[(orders.eval_set=='prior') & (orders.user_id.isin(train_order.user_id))][['user_id', 'order_id']]
train_products = pd.merge(log, train_prior_order, on='order_id')[['user_id', 'product_id']].drop_duplicates().reset_index(drop=True)
print(train_products.shape[0])
train_basic = pd.merge(train_order, train_products, on='user_id')
print(train_basic.shape[0])
train = train_basic.merge(log[(log.reordered==1)][['order_id', 'product_id', 'reordered']], how='left', on=['order_id', 'product_id'])
train.head()
train.fillna(0).to_csv(DATA+"/train.label.small.csv")

test_order = orders[orders.eval_set=='test'][['user_id', 'order_id']]
test_prior_order = orders[(orders.eval_set=='prior') & (orders.user_id.isin(test_order.user_id))][['user_id', 'order_id']]
test_products = pd.merge(log, test_prior_order, on='order_id')[['user_id', 'product_id']].drop_duplicates().reset_index(drop=True)
print(test_products.shape[0])
test_basic = pd.merge(test_order, test_products, on='user_id')
print(test_basic.shape[0])
test_basic.to_csv(DATA+'/test.label.small.csv')