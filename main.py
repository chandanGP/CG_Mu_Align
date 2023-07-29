from torch_geometric.data import HeteroData
import torch
import pandas as pd
import copy
import fasttext
import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data , HeteroData
from torch.nn.parallel import DataParallel
from matplotlib import pyplot as plt
import time
from Models import *


if __name__=="__main__":
    node_types=["song","artist","album"]
    edge_types=[("song","by_artist","artist"),
                ("artist","sung","song"),
                ("song","in","album"),
                ("album","have","song"),
                ("artist","artist","album"),
                ("album","by_artist","artist")]
    
    grapha_path="itunes_amazon/tableA.csv"
    graphb_path="itunes_amazon/tableB.csv"
    pair_train="itunes_amazon/train.csv"
    pair_test="itunes_amazon/test.csv"
    pair_valid="itunes_amazon/valid.csv"

    graph_a_data=pd.read_csv(grapha_path,header=0)
    graph_a_data=graph_a_data.to_dict(orient='records')

    graph_b_data=pd.read_csv(graphb_path,header=0)
    graph_b_data=graph_b_data.to_dict(orient='records')

    pair_data_dict={}

    pair_train_data=pd.read_csv(pair_train,header=0)
    pair_train_data=pair_train_data.to_dict(orient='records')
    pair_data_dict['train']=pair_train_data

    pair_test_data=pd.read_csv(pair_test,header=0)
    pair_test_data=pair_test_data.to_dict(orient='records')
    pair_data_dict['test']=pair_test_data

    pair_valid_data=pd.read_csv(pair_valid,header=0)
    pair_valid_data=pair_valid_data.to_dict(orient='records')
    pair_data_dict['valid']=pair_valid_data

    # Load the pre-trained FastText model
    model = fasttext.load_model('cc.en.300.bin')


    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    graph_a , graph_b , pos_pair_dict , neg_pair_dict = gen_graphs(graph_a_data,graph_b_data,node_types,edge_types,model,pair_data_dict,device)

    del model

    model_layers=[(300,150),(300,150)]

    temp_model=CGNN_net(model_layers,graph_a.metadata()).to(device)

    optzr = torch.optim.Adam(temp_model.parameters(), lr=0.003, weight_decay=5e-4)

    save_path=None

    train_model(temp_model,optzr,graph_a,graph_b,pos_pair_dict['train'],neg_pair_dict['train'],10,save_path)

    acc_test , loss_test , recall_curv_test , precision_curv_test , FPR_curv_test = test_model(temp_model,graph_a,graph_b,pos_pair_dict['test'],neg_pair_dict['test'],0.1)

    acc_valid , loss_valid , recall_curv_valid , precision_curv_valid , FPR_curv_valid = test_model(temp_model,graph_a,graph_b,pos_pair_dict['valid'],neg_pair_dict['valid'],0.1)

    print("______test data______")
    print("accuracy:",acc_test)
    print("loss:",loss_test)
    print("\n\n")
    print("______validation data______")
    print("accuracy:",acc_valid)
    print("loss:",loss_valid)
    print("\n\n")

    plt.plot(FPR_curv_test,recall_curv_test,label='test data',color='green')
    plt.plot(FPR_curv_valid,recall_curv_valid,label='valid data',color='blue')
    plt.xlabel("FPR ->")
    plt.ylabel("TPR ->")
    plt.title("ROC curve")
    plt.legend()
    plt.show()


    


    
