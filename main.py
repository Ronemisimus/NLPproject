from utils.training import TorchTrainer
from utils.models import YourCodeNet
import torch
from sklearn.model_selection import train_test_split
from utils.train_results import FitResult,EpochResult
from utils.plot import plot_fit
import os
from LoadData import get_embeddings, emb, enteilmentDataSet, EMBEDDINGS_SIZE, collect_batch, FROM_FILE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau

DEV_FILE = 'snli_1.0_dev.jsonl'
TRAIN_FILE = 'snli_1.0_train.jsonl'
TEST_FILE = 'snli_1.0_test.jsonl'

NUM_OF_THREADS = 16
HIDDEN_SIZE = 200
CLASSES = 2
NUM_LAYERES = 1
lr = 5e-4
reg = 1e-3
batch_size = 128
EPOCHS = 41
EARLY_STOP = 4
DEPTH = 4
ATTN_HEADS = 4
NAME = 'model_3_classes_attn_learned'


def main():
    os.environ["CUDA_LAUNCH_BLOCKING"]="1"

    # set embedding
    get_embeddings(FROM_FILE, NUM_OF_THREADS)

    # get all data
    dev_lines = open(DEV_FILE).readlines()
    train_lines = open(TRAIN_FILE).readlines()

    train_lines.extend(dev_lines)
    del dev_lines

    # split data
    lines_train, lines_val = train_test_split(train_lines,test_size=0.1)
    lines_test = open(TEST_FILE).readlines()
    del train_lines

    ds_train = enteilmentDataSet(lines_train,CLASSES)
    ds_val = enteilmentDataSet(lines_val, CLASSES)
    ds_test = enteilmentDataSet(lines_test,CLASSES)

    del lines_test, lines_val, lines_train
    
    print(len(emb.wordToNum))
    
    # build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YourCodeNet(HIDDEN_SIZE,
        CLASSES,
        embeddings=emb.indexToVec,
        embedding_size=EMBEDDINGS_SIZE,
        num_layers=NUM_LAYERES,
        head_num=ATTN_HEADS,
        depth=DEPTH)
    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=reg)
    sched = ReduceLROnPlateau(optim)
    trainer = TorchTrainer(model, loss, optim, sched, device)
    
    # create data loaders
    dl_train = torch.utils.data.DataLoader(ds_train,batch_size,shuffle=True,num_workers=NUM_OF_THREADS,collate_fn=collect_batch)
    dl_val = torch.utils.data.DataLoader(ds_val,batch_size,shuffle=False,num_workers=NUM_OF_THREADS,collate_fn=collect_batch)
    dl_test = torch.utils.data.DataLoader(ds_test,batch_size,shuffle=False,num_workers=NUM_OF_THREADS,collate_fn=collect_batch)

    # train
    res:FitResult = trainer.fit(dl_train,dl_val,num_epochs=EPOCHS,checkpoints=NAME,early_stopping=EARLY_STOP,tol=0.001)
    fig = plot_fit(res)
    
    plt.show(block=True)

    print(f'*** Loading checkpoint file {NAME}.pt')
    saved_state = torch.load(NAME+".pt",
                                map_location=device)
    model.load_state_dict(saved_state['model_state'])

    # test
    res:EpochResult = trainer.test_epoch(dl_test,)
    print("test accuracy:",res.accuracy)
    print("test loss", sum(res.losses)/len(res.losses))

    # claculate test prediction
    y_pred = []
    y_true = []
    for idx, batch in enumerate(dl_test):
        X,y = batch
        X,y = X.cuda(), y.cuda()
        batch_pred:torch.Tensor = model(X)
        if len(batch_pred.shape)<2:
            batch_pred = batch_pred[None]
        y_pred.extend(batch_pred.argmax(dim=1).reshape((X.shape[0],)).to('cpu').tolist())
        y_true.extend(y.reshape((X.shape[0],)).to('cpu').tolist())
    
    matrix = confusion_matrix(y_pred,y_true)

    print(matrix)

if __name__ == '__main__':
    main()