# for common
import os
import time
import random
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import visdom

#for dataset
import json
from torch.utils.data import Dataset, DataLoader, ConcatDataset

#for model
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW, get_linear_schedule_with_warmup

#for ddp
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=360)
    parser.add_argument('--max_norm', type=float, default=3.0)
    parser.add_argument('--global_rank', type=int, default=0)
    parser.add_argument('--vis_step', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument("--local_rank", type=int,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument('--world_size', type=int, default=0)
    #parser.add_argument('--port', type=int, default=2022)
    #parser.add_argument('--root', type=str, default='data')
    #parser.add_argument('--start_epoch', type=int, default=0)
    #parser.add_argument('--save_path', type=str, default='./save')
    #parser.add_argument('--save_file_name', type=str, default='vgg_cifar')
    return parser

class PostDataset(Dataset):
    def __init__(self, data_path, train_ratio=0.8):
        model_checkpoint = 'michiyasunaga/BioLinkBERT-base'
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        print(data_path, "Loading..............")
        with open(data_path, "r") as f:
            self.data = json.load(f)

        self.all_sent_bag = [sent[0] for para in self.data for sent in para]
        self.sentences_a = []
        self.sentences_b_next = []
        self.sentences_b_linked = []

        print(data_path, " processing..............")
        for para in tqdm(self.data):
            for sent_idx in range(len(para)):
                self.sentences_a.append(para[sent_idx][0])
                if sent_idx == len(para) - 1:
                    self.sentences_b_next.append(None)
                else:
                    self.sentences_b_next.append(para[sent_idx + 1][0])
                self.sentences_b_linked.append(para[sent_idx][1])
        self.data = ''
        assert len(self.sentences_a) == len(self.sentences_b_next) == len(self.sentences_b_linked)

        self.bag_size = len(self.all_sent_bag)
        self.vocab_ids = list(self.tokenizer.vocab.values())
        print("#n_sentence in", data_path.split('/')[1].split('.')[0],":", self.bag_size)
        print("#n_vocab in", data_path.split('/')[1].split('.')[0],":", len(self.vocab_ids))

        # Split the dataset into training and testing
        train_size = int(len(self.sentences_a) * train_ratio)
        test_size = len(self.sentences_a) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            self, [train_size, test_size]
        )
        print("#n_sentence of train in ", data_path.split('/')[1].split('.')[0],":", train_size)
        print("#n_sentence of test in ", data_path.split('/')[1].split('.')[0],":", test_size)


 
    def __len__(self):
        return len(self.sentences_a)

    def __getitem__(self, idx):
        sentence_a = self.sentences_a[idx]
        sent_b_next = self.sentences_b_next[idx]
        sent_b_linked = self.sentences_b_linked[idx]

        if sent_b_next is None and sent_b_linked is None:
            sentence_b = random.choice(self.all_sent_bag)
            label = 1
        elif sent_b_next is None:
            prob = random.random()
            if prob <= 0.5:
                sentence_b = random.choice(self.all_sent_bag)
                label = 1
            else:
                sentence_b = random.choice(sent_b_linked)
                label = 2
        elif sent_b_linked is None:
            prob = random.random()
            if prob <= 0.5:
                sentence_b = sent_b_next
                label = 0
            else:
                sentence_b = random.choice(self.all_sent_bag)
                label = 1
        else:
            prob = random.random()
            if prob <= 0.33:
                sentence_b = sent_b_next
                label = 0
            elif prob <= 0.66:
                sentence_b = random.choice(self.all_sent_bag)
                label = 1
            else:
                sentence_b = random.choice(sent_b_linked)
                label = 2

        inputs = self.tokenizer(
            sentence_a,
            sentence_b,
            max_length=512,
            truncation=True,
            padding='max_length'
        )
        inputs['nsp_label'] = torch.LongTensor([label])
        inputs['mlm_label'] = inputs.input_ids.copy()

        for i, input_id in enumerate(inputs.input_ids):
            if (
                input_id != self.tokenizer.cls_token_id and
                input_id != self.tokenizer.sep_token_id and
                input_id != self.tokenizer.pad_token_id
            ):
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15

                    if prob < 0.8:
                        inputs.input_ids[i] = self.tokenizer.mask_token_id
                    elif prob < 0.9:
                        inputs.input_ids[i] = random.choice(self.vocab_ids)
                    else:
                        inputs.input_ids[i] = input_id
                else:
                    inputs.input_ids[i] = input_id

        return {key: torch.tensor(val) for key, val in inputs.items()}

class PostModel(nn.Module):
    def __init__(self):
        super(PostModel, self).__init__()
        model_checkpoint = "michiyasunaga/BioLinkBERT-base"
        # model_checkpoint = "roberta-base"
        # model_checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

        self.model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
        self.hiddenDim = self.model.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        """ score matrix """
        self.W = nn.Linear(self.hiddenDim, 3)

    def forward(self, input_ids, token_type_ids, attention_mask, mlm_label):
        output = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=mlm_label, output_hidden_states=True)
        mlm_loss = output['loss']
        cls_token_state = output['hidden_states'][-1][:, 0, :]
        cls_output = self.W(cls_token_state)
        return mlm_loss, cls_output


#def train(model, train_dataloader, test_dataloader, num_epochs, batch_size, lr=1e-3, save_every=1000):



def main(opts):
	# 1. set random seeds
    set_random_seeds(random_seed=0)
	
    # 2. initialization
    init_for_distributed(opts)
    
    # 3. visdom
    vis = None
    #vis = visdom.Visdom(port=opts.port)

    # 4. data set
    train_dataset_list = []
    test_dataset_list = []
    
    for data_n in range(0, 15):
        data_path = './final_train_' + str(data_n) + '.json'
        dataset = PostDataset(data_path)
        train_dataset_list.append(dataset.train_dataset)
        test_dataset_list.append(dataset.test_dataset)
    final_train_dataset = ConcatDataset(train_dataset_list)
    final_test_dataset = ConcatDataset(test_dataset_list)
    print("#n_total_of_sentence for training: ", len(final_train_dataset))
    print("#n_total_of_sentence for testing: ", len(final_test_dataset))
    
    train_sampler = DistributedSampler(dataset=final_train_dataset, shuffle=True)
    test_sampler = DistributedSampler(dataset=final_test_dataset, shuffle=False)
    
    train_loader = DataLoader(dataset=final_train_dataset,
                            batch_size=int(opts.batch_size / opts.world_size),
                            shuffle=False,
                            num_workers=int(opts.num_workers / opts.world_size),
                            sampler=train_sampler,
                            pin_memory=True)
    
    test_loader = DataLoader(dataset=final_test_dataset,
                            batch_size=int(opts.batch_size / opts.world_size),
                            shuffle=False,
                            num_workers=int(opts.num_workers / opts.world_size),
                            sampler=test_sampler,
                            pin_memory=True)
                            
    # 5. model                                
    model = PostModel()
    model = model.cuda(opts.local_rank)
    model = DDP(module=model,
                device_ids=[opts.local_rank])    
                
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr)
    #total_steps = (len(train_loader)//opts.batch_size) * opts.epoch
    total_steps = (len(train_loader)) * opts.epoch
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)    
    scaler = torch.cuda.amp.GradScaler()  # Gradient scaler for mixed-precision training

# 6. traing
    train_loss_history = []
    test_loss_history = []
    learning_rate_history = []
    curr_steps = 0
    
    save_every=5000
    test_every=1000
    vis_every=100
    
    for epoch in range(opts.epoch):
        #if opts.global_rank == 0:
        #    print(f"Epoch {epoch + 1}/{opts.epoch}")
        tic = time.time()
        model.train()
        
        train_losses = []

        #loop = tqdm(train_loader, leave=True)
        #for step, batch in enumerate(loop):
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(opts.local_rank)
            token_type_ids = batch['token_type_ids'].to(opts.local_rank)
            attention_mask = batch['attention_mask'].to(opts.local_rank)
            mlm_label = batch['mlm_label'].to(opts.local_rank)
            nsp_label = batch['nsp_label'].to(opts.local_rank)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                mlm_loss, cls_output = model(input_ids, token_type_ids, attention_mask, mlm_label)
                cls_loss = nn.CrossEntropyLoss()(cls_output, nsp_label.reshape(-1))
                total_loss = mlm_loss.mean() + cls_loss.mean()  # Compute the mean loss
            
            # ----------- update -----------
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opts.max_norm)  # Gradient clipping            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # Scheduler step
            train_losses.append(total_loss.item())
            curr_steps += 1

            # get lr
            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            # time
            toc = time.time()

            # visualization
            if opts.global_rank != 0:
                print('Performance is showing in master pc')            
            
            if (curr_steps % vis_every == 0 or step == len(train_loader) - 1) and opts.global_rank ==0:
                print('Epoch [{0}/{1}] - Iter [{2}/{3}] - Step [{4}/{5}] - MLM_Loss: {6:.4f}, CLS_Loss: {7:.4f}, Total_Loss: {8:.4f}, LR: {9:.5f}, Time: {10:.2f}'.format(epoch,
                                                                                                                                                        opts.epoch,
                                                                                                                                                        step,
                                                                                                                                                        len(train_loader),
                                                                                                                                                        curr_steps,
                                                                                                                                                        total_steps,
                                                                                                                                                        mlm_loss.item(),
                                                                                                                                                        cls_loss.item(),
                                                                                                                                                        total_loss.item(),
                                                                                                                                                        lr,
                                                                                                                                                        toc - tic))
                #if vis is not None and opts.local_rank == 0:
                #    vis.line(X=torch.ones((1, 1)) * step + epoch * len(train_loader),
                #             Y=torch.Tensor([loss]).unsqueeze(0),
                #             update='append',
                #             win='loss',
                #             opts=dict(x_label='step',
                #                       y_label='loss',
                #                       title='loss',
                #                       legend=['total_loss']))
            
            #loop.set_postfix(mlm_loss=mlm_loss.mean().item(), cls_loss=cls_loss.mean().item(), total_loss=total_loss.item())

            if curr_steps % save_every == 0 and opts.global_rank ==0:
                model_path = f"model_step_{curr_steps}.pt"
                torch.save(model.state_dict(), model_path)
                print(f"model_step_{curr_steps} is saved....................................")
                
            if (curr_steps % test_every == 0 or step == len(train_loader) - 1) and opts.global_rank ==0:
                model.eval()
                test_total_losses = []
                test_mlm_losses = []
                test_cls_losses = []
                with torch.no_grad():
                    for test_batch in tqdm(test_loader):
                        test_input_ids = test_batch['input_ids'].to(opts.local_rank)
                        test_token_type_ids = test_batch['token_type_ids'].to(opts.local_rank)
                        test_attention_mask = test_batch['attention_mask'].to(opts.local_rank)
                        test_mlm_label = test_batch['mlm_label'].to(opts.local_rank)
                        test_nsp_label = test_batch['nsp_label'].to(opts.local_rank)
        
                        test_mlm_loss, test_cls_output = model(test_input_ids, test_token_type_ids, test_attention_mask, test_mlm_label)
                        test_cls_loss = nn.CrossEntropyLoss()(test_cls_output, test_nsp_label.reshape(-1))
                        test_total_loss = test_mlm_loss.mean() + test_cls_loss.mean()
        
                        test_mlm_losses.append(test_mlm_loss.item())
                        test_cls_losses.append(test_cls_loss.item())
                        test_total_losses.append(test_total_loss.item())
                        
                avg_train_loss = np.mean(train_losses)
                avg_test_mlm_loss = np.mean(test_mlm_losses)
                avg_test_cls_loss = np.mean(test_cls_losses)
                avg_test_total_loss = np.mean(test_total_losses)
                
                train_loss_history.append(avg_train_loss)
                test_loss_history.append(avg_test_total_loss)
                learning_rate_history.append(optimizer.param_groups[0]['lr'])
        
                print('Epoch [{0}/{1}] - Iter [{2}/{3}] - Step [{4}/{5}] - test_MLM_Loss: {6:.4f}, test_CLS_Loss: {7:.4f}, test_Total_Loss: {8:.4f}'.format(epoch,
                                                                                                                                                        opts.epoch,
                                                                                                                                                        step,
                                                                                                                                                        len(train_loader),
                                                                                                                                                        curr_steps,
                                                                                                                                                        total_steps,
                                                                                                                                                        avg_test_mlm_loss,
                                                                                                                                                        avg_test_cls_loss,
                                                                                                                                                        avg_test_total_loss))
            
    #if opts.global_rank ==0:
    #    plt.plot(train_loss_history, label="Train Loss")
    #    plt.plot(test_loss_history, label="Test Loss")
    #    plt.xlabel("Epoch")
    #    plt.ylabel("Loss")
    #    plt.legend()
    #    plt.show()
    #
    #    plt.plot(learning_rate_history)
    #    plt.xlabel("Steps")
    #    plt.ylabel("Learning Rate")
    #    plt.show()

def init_for_distributed(opts):

    # 1. setting for distributed training
    opts.global_rank = int(os.environ['RANK'])
    opts.local_rank = int(os.environ['LOCAL_RANK'])
    opts.world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(opts.local_rank)
    if opts.global_rank is not None and opts.local_rank is not None:
        print("Use GPU: [{}/{}] for training".format(opts.global_rank, opts.local_rank))

    # 2. init_process_group
    dist.init_process_group(backend="nccl")
    # if put this function, the all processes block at all.
    torch.distributed.barrier()
    setup_for_distributed(opts.global_rank == 0) 	
    return

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print               
                
                
if __name__ == '__main__':

    parser = argparse.ArgumentParser('ebert pubmed fulltext training', parents=[get_args_parser()])
    opts = parser.parse_args()
    main(opts)
    
    