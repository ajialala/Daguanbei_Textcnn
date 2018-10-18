import pandas as pd

file = pd.read_csv('train_set.csv')
del file['id']
del file['article']
file = file.loc[:,['class','word_seg']]
# file.to_csv('train_set_cnn',index=0,sep='\t',header=0)

fenge = [0.9,0.05,0.05]
file.iloc[0:int(len(file)*fenge[0]),].to_csv('train_data_w',index=0,header=0,sep='\t')
file.iloc[int(len(file)*fenge[0]):int(len(file)*(fenge[0]+fenge[1])),].to_csv('val_data_w',index=0,header=0,sep='\t')
file.iloc[int(len(file)*(fenge[0]+fenge[1])):len(file),].to_csv('test_data_w',index=0,header=0,sep='\t')