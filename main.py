import argparse
import os
from pickle import HIGHEST_PROTOCOL

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import dill
import ITSG_LSTM
from TSErrors import FindErrors

def _main(args):
    docs = sorted([file for file in os.listdir(args.p_doc_dir)])
    ttd_embed_list = []
    for p, doc in enumerate(docs, 1):
        data = pd.read_csv(args.p_doc_dir+doc ,engine='python', index_col=0, encoding = 'utf-8')
        for l in range(1,args.num_level+1):
            ttd_embedding = generate_ttd_embedding(data, p, l, args)
            ttd_embed_list.append(ttd_embedding)
    
    c_ttd_embedding = pd.concat(ttd_embed_list, axis = 1).reset_index()
    
    c_ttd_embedding['month'] = c_ttd_embedding.iloc[:,0].apply(lambda x: x.split('-')[0])+'-'+c_ttd_embedding.iloc[:,0].apply(lambda x: x.split('-')[1])
    c_ttd_embedding['country'] = c_ttd_embedding.iloc[:,0].apply(lambda x: x.split('-')[-1])
    
    
    
    
    
def main(args):
    tt_im_data, tt_ex_data, node_feature, cat_list = processing_bl(args.cargo, args.bl_dir, args.bl_file)
    
    ttd_embed_list = []
    
    for l in range(1,args.num_level+1):
        ttd_embedding = generate_ttd_embedding(tt_im_data, 1, l, args)
        ttd_embed_list.append(ttd_embedding)
    for l in range(1,args.num_level+1):
        ttd_embedding = generate_ttd_embedding(tt_ex_data, 2, l, args)
        ttd_embed_list.append(ttd_embedding)
    
    c_ttd_embedding = pd.concat(ttd_embed_list, axis = 1).reset_index()
    c_ttd_embedding['month'] = c_ttd_embedding.iloc[:,0].apply(lambda x: x.split('-')[0])+'-'+c_ttd_embedding.iloc[:,0].apply(lambda x: x.split('-')[1])
    c_ttd_embedding['country'] = c_ttd_embedding.iloc[:,0].apply(lambda x: x.split('-')[-1])
    
    train_G, test_G, trainY, testY, minmax_scaler, minmax_scaler_y, G_list = train_test_data(node_feature, c_ttd_embedding, args.split_ratio, cat_list, args)
    
#     os.chdir(args.dill_dir+'{}_dill_includecat'.format(args.cargo))
#     os.chdir(args.dill_dir+'{}_dill_exceptcat'.format(args.cargo))

#     dill_file = '{}_total_total_all_{}_{}_graph.pkl'.format(args.target, args.cut_off, args.seq_length)
#     dill_file = 'nocat_{}_total_total_all_{}_{}_graph.pkl'.format(args.target, args.cut_off, args.seq_length)
#     print(dill_file)
#     dill.load_session(dill_file)

    torch.manual_seed(1234)

    model = ITSG_LSTM.GCNLSTM(input_dim = args.input_dim, gcn_dim = args.gcn_dim, seq_length = args.seq_length, 
    lstm_dim = args.lstm_dim, output_dim = args.output_dim, num_layer = args.num_layer)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)

    history = {0: np.inf}

    for epoch in range(1,args.epochs+1):
        # model.train()
        
        pred = model(train_G)
        loss = loss_fn(pred, trainY)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tr_forecasted = minmax_scaler_y.inverse_transform(model(train_G).detach().numpy())
        tr_truth = minmax_scaler_y.inverse_transform(trainY)    
        te_forecasted = minmax_scaler_y.inverse_transform(model(test_G).detach().numpy())
        te_truth = minmax_scaler_y.inverse_transform(testY)
        
        tr_er = FindErrors(tr_truth,tr_forecasted)
        te_er = FindErrors(te_truth,te_forecasted)

        if epoch % 100 == 0:
            # model.eval()

            print("epoch: {} loss: {}".format(epoch,loss))
            print('trainMAE: {:.2f} | trainRMSE: {:.2f} | trainMAPE: {:.2f} | trainSMAPE: {:.2f}'.format(tr_er.mae(), tr_er.rmse(), tr_er.mape(), tr_er.smape()))
            print("testMAE: {:.2f} | testRMSE: {:.2f} | testMAPE: {:.2f} | testSMAPE: {:.2f}".format(te_er.mae(), te_er.rmse(), te_er.mape(), te_er.smape()))
        
        if min(history.values()) >= tr_er.mape():
            optim_tr_er = tr_er
            optim_te_er = te_er
        
        history[epoch] = tr_er.mape()

        if epoch == args.epochs:
            print('-------------------------------------------------------------------------------------------------------------')
            print('-------------------------------------------------------------------------------------------------------------')
            print('trainMAE: {:.2f} || trainRMSE: {:.2f} || trainMAPE: {:.2f} || trainSMAPE: {:.2f}'.format(optim_tr_er.mae(), optim_tr_er.rmse(), optim_tr_er.mape(), optim_tr_er.smape()))
            print('trainMRAE: {:.2f} || trainMASE: {:.2f} || trainMBRAE: {:.2f} || trainUMBRAE: {:.2f}'.format(optim_tr_er.mrae(), optim_tr_er.mase(), optim_tr_er.mbrae(), optim_tr_er.umbrae()))
            print('-------------------------------------------------------------------------------------------------------------') 
            print("testMAE: {:.2f} || testRMSE: {:.2f} || testMAPE: {:.2f} || testSMAPE: {:.2f}".format(optim_te_er.mae(), optim_te_er.rmse(), optim_te_er.mape(), optim_te_er.smape())) 
            print("testMRAE: {:.2f} || testMASE: {:.2f} || testMBRAE: {:.2f} || testUMBRAE: {:.2f}".format(optim_te_er.mrae(), optim_te_er.mase(), optim_te_er.mbrae(), optim_te_er.umbrae())) 
            print('-------------------------------------------------------------------------------------------------------------')
            print('-------------------------------------------------------------------------------------------------------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cargo',default='LBC')
    parser.add_argument('--target', default='crude') # target category

    parser.add_argument('--input_dim',type=int, default=5)
    parser.add_argument('--gcn_dim',type=int, default=8)
    parser.add_argument('--lstm_dim',type=int,default=12)
    parser.add_argument('--output_dim',default=1, type=int)
    parser.add_argument('--num_layer',default=1, type=int) # LSTM layer
    
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--learning_rate', default=0.001)

    parser.add_argument('--cut_off',default=0.9)
    parser.add_argument('--seq_length',type=int, default=12)    
    
    parser.add_argument('--window', default=10)
    parser.add_argument('--size', default=100)

    parser.add_argument('--num_purpose', default = 2)
    parser.add_argument('--num_level', default = 3)
    
    parser.add_argument('--bl_dir', default = '/data/bl_data/')
    parser.add_argument('--bl_file', default = 'bl_data.csv')
    parser.add_argument('--wv_dir', default = '/data/wv_dir/')
    
    parser.add_argument('--split_ratio', default = 0.8)

    #     parser.add_argument('--dill_dir', default='/data/nono_sohn/UPA_2nd_edition/')
#     parser.add_argument('--p_doc_dir') # import_document, export_document.csv가 있다고 가정
    


    args = parser.parse_args()
    main(args)
