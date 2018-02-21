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

log=pd.read_csv(DATA + '/log.prior.small.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8}, usecols=['order_id', 'product_id', 'reordered'], engine='c' )
log.sort_values(by=['order_id', 'product_id'], inplace=True)
log = log.reset_index(drop=True)

product_week = pd.merge(log[['order_id', 'product_id']], 
         orders[['order_id', 'order_week']],
         on='order_id', how='inner').drop(columns='order_id')
product_week = product_week.merge(product_week.order_week.value_counts(normalize=True).rename('order_week_distribution').to_frame(),
                   left_on='order_week', right_index=True).sort_values(['product_id', 'order_week']).reset_index(drop=True)
product_week = product_week.set_index(['product_id', 'order_week']).merge(
    product_week.groupby(['product_id', 'order_week'])['order_week_distribution'].count().rename('item_week_cnt').to_frame(),
    left_index=True, right_index=True
).reset_index().set_index('product_id').merge(
    product_week.groupby('product_id')['order_week_distribution'].count().rename('item_cnt').to_frame(),
    left_index=True, right_index=True
).reset_index().drop_duplicates()
product_week['item_week_prob'] = product_week['item_week_cnt'] / product_week['item_cnt']
product_week['item_week_prior_to_posterior'] = product_week['order_week_distribution'] - product_week['item_week_prob']
product_week[['product_id', 'order_week', 'item_week_prior_to_posterior']].to_csv(DATA+'/item_week_order_prob_posterior.csv', index=False)

