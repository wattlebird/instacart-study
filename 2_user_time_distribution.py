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

orders['order_timeslot'] = orders['order_hour']//6
orders['order_week_slot'] = orders['order_timeslot']+4*orders['order_week']
orders_time = orders[orders.eval_set=='prior']

u2 = pd.concat([
    pd.crosstab(orders_time.user_id, orders_time.order_timeslot).add_prefix('user_timeslot_'),
    pd.crosstab(orders_time.user_id, orders_time.order_timeslot, normalize='index').add_prefix('user_timeslotN_'),
    pd.crosstab(orders_time.user_id, orders_time.order_week).add_prefix('user_week_'),
    pd.crosstab(orders_time.user_id, orders_time.order_week, normalize='index').add_prefix('user_weekN_'),
    pd.crosstab(orders_time.user_id, orders_time.order_week_slot).add_prefix('user_week_slot_'),
    pd.crosstab(orders_time.user_id, orders_time.order_week_slot, normalize='index').add_prefix('user_week_slotN_')
], axis=1)

u2.to_csv(DATA+'/user_time_distribution.csv')