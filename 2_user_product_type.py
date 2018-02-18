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
        'days_since_prior_order': np.float32}, engine='c').\
        rename(columns = {'order_dow': 'order_week', 'order_hour_of_day': 'order_hour', 'days_since_prior_order': 'order_gap'})

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

product = pd.read_csv(DATA+'/products.csv', dtype={
    'product_id': np.uint16,
    'product_name': str,
    'aisle_id': np.uint8,
    'department_id': np.uint8
}, engine='c')

product.set_index('product_id', inplace=True)
product['product_organic'] = product.product_name.apply(lambda x: 0 if('organic' not in x.lower()) else 1)
product['product_glutenfree'] = product.product_name.apply(lambda x: 0 if('gluten' not in x.lower() or 'free' not in x.lower()) else 1)
product['product_asian'] = product.product_name.apply(lambda x: 0 if('asian' not in x.lower()) else 1)

order_prior = orders[orders.eval_set=='prior'].order_id
log_product = log[log.order_id.isin(order_prior)].\
merge(product[['product_organic', 'product_glutenfree', 'product_asian']], 
     left_on='product_id', right_index=True, how='left')
log_product = log_product.merge(orders[['order_id', 'user_id']], on='order_id')

u3 = pd.concat([
    log_product.groupby(by='user_id')['product_organic'].sum().rename('user_organic_cnt'),
    log_product.groupby(by='user_id')['product_glutenfree'].sum().rename('user_glutenfree_cnt'),
    log_product.groupby(by='user_id')['product_asian'].sum().rename('user_asian_cnt'),
    log_product.groupby(by='user_id')['product_id'].count().rename('user_total_item')
], axis=1)
u3['user_organic_rate'] = u3.user_organic_cnt/u3.user_total_item
u3['user_glutenfree_rate'] = u3.user_glutenfree_cnt/u3.user_total_item
u3['user_asian_rate'] = u3.user_asian_cnt/u3.user_total_item
u3.to_csv(DATA+'/user_product_type.csv')