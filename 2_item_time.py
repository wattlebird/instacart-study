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
log_extended = log.merge(orders, on='order_id').drop(columns=['reordered', 'eval_set', 'order_number', 'order_gap'])
log_extended['order_timeslot'] = log_extended.order_hour//6
log_extended['order_timeslot_week'] = 4*log_extended.order_week+log_extended.order_timeslot

i1 = log_extended.groupby(by=['product_id', 'order_hour']).size().rename('item_hour_cnt').reset_index().set_index('product_id')
i1['item_hour_prob'] = i1.item_hour_cnt / i1.groupby('product_id')['item_hour_cnt'].sum()
i2 = log_extended.drop_duplicates(['user_id', 'product_id', 'order_hour']).\
groupby(by=['product_id', 'order_hour']).size().\
rename('itemperuser_hour_cnt').reset_index().set_index('product_id')
i2['itemperuser_hour_prob'] = i2.itemperuser_hour_cnt / i2.groupby('product_id')['itemperuser_hour_cnt'].sum()
i3 = log_extended.groupby(by=['product_id', 'order_week']).size().rename('item_week_cnt').\
reset_index().set_index('product_id')
i3['item_week_prob'] = i3.item_week_cnt / i3.groupby('product_id')['item_week_cnt'].sum()
i4 = log_extended.drop_duplicates(['user_id', 'product_id', 'order_week']).\
groupby(by=['product_id', 'order_week']).size().\
rename('itemperuser_week_cnt').reset_index().set_index('product_id')
i4['itemperuser_week_prob'] = i4.itemperuser_week_cnt / \
                              i4.groupby('product_id')['itemperuser_week_cnt'].sum()
i5 = log_extended.groupby(by=['product_id', 'order_timeslot']).size().rename('item_timeslot_cnt').\
reset_index().set_index('product_id')
i5['item_timeslot_prob'] = i5.item_timeslot_cnt / i5.groupby('product_id')['item_timeslot_cnt'].sum()
i6 = log_extended.drop_duplicates(['user_id', 'product_id', 'order_timeslot']).\
groupby(by=['product_id', 'order_timeslot']).size().\
rename('itemperuser_timeslot_cnt').reset_index().set_index('product_id')
i6['itemperuser_timeslot_prob'] = i6.itemperuser_timeslot_cnt / \
                              i6.groupby('product_id')['itemperuser_timeslot_cnt'].sum()
i7 = log_extended.groupby(by=['product_id', 'order_timeslot_week']).size().rename('item_timeslot_week_cnt').\
reset_index().set_index('product_id')
i7['item_timeslot_week_prob'] = i7.item_timeslot_week_cnt / i7.groupby('product_id')['item_timeslot_week_cnt'].sum()
i8 = log_extended.drop_duplicates(['user_id', 'product_id', 'order_timeslot_week']).\
groupby(by=['product_id', 'order_timeslot_week']).size().\
rename('itemperuser_timeslot_week_cnt').reset_index().set_index('product_id')
i8['itemperuser_timeslot_week_prob'] = i8.itemperuser_timeslot_week_cnt / \
                              i8.groupby('product_id')['itemperuser_timeslot_week_cnt'].sum()
i9 = pd.crosstab(log_extended.product_id, log_extended.order_timeslot_week, normalize='index').add_prefix('item_timeslot_week_prob_').reset_index()

pd.merge(i1.reset_index(), i2.reset_index(), on=['product_id', 'order_hour']).to_csv(DATA+'/item_hour.csv', index=False)
pd.merge(i3.reset_index(), i4.reset_index(), on=['product_id', 'order_week']).to_csv(DATA+'/item_week.csv', index=False)
pd.merge(i5.reset_index(), i6.reset_index(), on=['product_id', 'order_timeslot']).to_csv(DATA+'/item_timeslot.csv', index=False)
pd.merge(i7.reset_index(), i8.reset_index(), on=['product_id', 'order_timeslot_week']).to_csv(DATA+'/item_timeslot_week.csv', index=False)
i9.to_csv(DATA+'/item_timeslot_week_dist.csv', index=False)