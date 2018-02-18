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

orders['order_id_t-1'] = orders.groupby('user_id').order_id.shift(1)
orders['order_id_t-2'] = orders.groupby('user_id').order_id.shift(2)
orders['order_id_t-3'] = orders.groupby('user_id').order_id.shift(3)

order_hour = orders[['order_id', 'order_hour']]
order_prevhour = orders.merge(order_hour.add_suffix('_t-1'), on='order_id_t-1', how='left').\
                        merge(order_hour.add_suffix('_t-2'), on='order_id_t-2', how='left').\
                        merge(order_hour.add_suffix('_t-3'), on='order_id_t-3', how='left')
order_prevhour['order_delta_hour_t-1'] = order_prevhour.order_hour-order_prevhour['order_hour_t-1']
order_prevhour['order_delta_hour_t-2'] = order_prevhour.order_hour-order_prevhour['order_hour_t-2']
order_prevhour['order_delta_hour_t-3'] = order_prevhour.order_hour-order_prevhour['order_hour_t-3']

order_prevhour = order_prevhour.drop(columns=['user_id', 'eval_set', 'order_number', 'order_week', 'order_hour', 
                             'order_gap', 'order_id_t-1', 'order_id_t-2', 'order_id_t-3',
                             'order_hour_t-1', 'order_hour_t-2', 'order_hour_t-3']).\
               set_index('order_id')

order_prevhour.to_csv('order_prev_hour.csv')