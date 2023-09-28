from fastai.text.all import *
import pandas as pd

# check default device
print(default_device())

path = untar_data(URLs.YELP_REVIEWS)
print('Data are lay down:', path.ls())

train_path = 'train.csv'
test_path = 'test.csv'
readme = 'readme.txt'

data = pd.read_csv(
    f"{path}/{train_path}",
    nrows=150000
)
print(data.head())
print(data.size)
print(data.columns.tolist())

label_column, text_column = data.columns.tolist()
dls = TextDataLoaders.from_df(
    df=data,
    text_col=text_column,
    label_col=label_column
)
# dls = load_data('default.pkl')
print(dls.device)

torch.save(dls, 'default.pkl')

print(dls.show_batch())

learn = text_classifier_learner(
    dls=dls,
    arch=AWD_LSTM,
    drop_mult=0.5,
    metrics=accuracy
)

# using CPU on ~30%, MPS ~< 90%,
# ??? - using CPU on > 100%, MPS only ~< 20
# requires so much RAM (after 30 min of fine_tune, its eat RAM 30GB+)
epochs = 4
for i in range(0, epochs):
    learn.fine_tune(1, 1e-2)
    learn.save('model')
    print(learn.show_results())

learn.predict("I really liked doctor")