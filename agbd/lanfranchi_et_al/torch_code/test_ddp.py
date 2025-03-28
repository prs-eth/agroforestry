import os
from datetime import datetime
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
import glob
from torch.utils.tensorboard import SummaryWriter

from utils.parser import setup_parser

"""
try:
    import wandb
    wandb.init(sync_tensorboard=True, project="GEDI", entity="clanfranchi")
except ImportError:
    print("Install wandb to log to Weights & Biases")
"""
def main():
    parser = setup_parser()
    args, _ = parser.parse_known_args()
    
    results_file = 'results/results.test_ddp.txt'
    # Remove previous results
    for f in glob.glob(results_file):
        os.remove(f)
    
    args.results_file = results_file
    
    if not torch.cuda.is_available():
        print("CPU")
    
    else:
        args.world_size = args.gpus * args.nodes
        os.environ['MASTER_ADDR'] = args.ip
        os.environ['MASTER_PORT'] = args.port
        with open(results_file, 'a') as f:
                f.write(f'MASTER_ADDR = {args.ip} \n')
                f.write(f'MASTER_PORT = {args.port} \n')
                f.write(f'n_gpus = {args.gpus} \n')
            
    mp.spawn(train, nprocs=args.gpus, args=(args,))


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(gpu, args):
    results_file = args.results_file
    
    rank = args.nr * args.gpus + gpu
    with open(results_file, 'a') as f:
        f.write(f'rank = {rank} \n')  
        
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(args.random_seed)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    
    if rank==0:
        try:
            import wandb
            wandb.init(sync_tensorboard=True, project="GEDI", entity="clanfranchi")
        except ImportError:
            print("Install wandb to log to Weights & Biases")
            
        wandb.config = {
            "learning_rate": 1e-4,
            "epochs": args.n_epochs,
            "batch_size": batch_size
        }
        writer = SummaryWriter("first_try_00")
      
    with open(results_file, 'a') as f:
        f.write('Preparing Dataset' + '\n') 
    
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)
    running_loss = 0
    for epoch in range(args.n_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.n_epochs, i + 1, total_step,
                                                                         loss.item()))
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
    if rank == 0:
        writer.add_scalar("train loss", running_loss / len(train_loader), global_step=epoch)
        


if __name__ == '__main__':
    main()
