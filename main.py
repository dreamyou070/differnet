import config as c
import argparse
from tqdm import tqdm
import torch
from utils import load_datasets, make_dataloaders
from model import DifferNet, save_model, save_weights


class Score_Observer:
    '''Keeps an eye on the current and highest score so far'''

    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = None
        self.last = None

    def update(self, score, epoch, print_score=False):
        self.last = score
        if epoch == 0 or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
        if print_score:
            self.print_score()

    def print_score(self):
        print('{:s}: \t last: {:.4f} \t max: {:.4f} \t epoch_max: {:d}'.format(self.name, self.last, self.max_score,
                                                                               self.max_epoch))

def main(args):

    print(f' step 1. loading dataset')
    dataset_path = args.dataset_path
    class_name = args.class_name
    args.img_dims = [3] + list(args.img_size)
    train_set, test_set = load_datasets(dataset_path,
                                        class_name,
                                        args)
    args.batch_size_test = args.batch_size * args.n_transforms // args.n_transforms_test
    train_loader, test_loader = make_dataloaders(train_set, test_set, args.batch_size, args.batch_size_test)

    print(f' step 2. make model')
    args.n_feat = 256 * args.n_scales
    model = DifferNet(n_scales=args.n_scales,                    # 3
                      n_feat=args.n_feat,                        # 256 * 3
                      n_coupling_blocks=args.n_coupling_blocks,  # 8
                      clamp_alpha=args.clamp_alpha,              # 3
                      fc_internal=args.fc_internal,              # 2048
                      dropout=args.dropout)                      # 0.0
    model.to(args.device)

    print(f' step 3. optimizer')
    """ train only nf """
    optimizer = torch.optim.Adam(model.nf.parameters(),
                                 lr=c.lr_init,
                                 betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)

    print(f' step 4. scoring object')
    score_obs = Score_Observer('AUROC')

    print(f' step 5. Train!')
    for epoch in range(args.meta_epochs):
        # train some epochs
        model.train()
        print(F'\nTrain epoch {epoch}')
        for sub_epoch in range(args.sub_epochs):
            train_loss = list()
            for i, data in enumerate(tqdm(train_loader, disable=args.hide_tqdm_bar)):
                optimizer.zero_grad()
                # 1) raw data
                inputs, labels = data                                           # normal label = 0
                inputs, labels = inputs.to(args.device), labels.to(args.device) # input = batch, transform, 3, 448, 448
                inputs = inputs.view(-1, *inputs.shape[-3:])                    # input = total_num,        3, 448, 448

                # 2) make anomal data
                inputs += torch.randn(*inputs.shape).cuda()
                z = model(inputs)

                # 3) get loss
                #loss = get_loss(z, model.nf.jacobian(run_forward=False))
                #train_loss.append(t2np(loss))
                #loss.backward()
                #optimizer.step()
            #mean_train_loss = np.mean(train_loss)
            #if c.verbose:
            #    print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))
        # evaluate
        model.eval()
        print('\nCompute loss and scores on test set:')
        test_loss = list()
        test_z = list()
        test_labels = list()
        """
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
                inputs, labels = preprocess_batch(data)
                z = model(inputs)
                loss = get_loss(z, model.nf.jacobian(run_forward=False))
                test_z.append(z)
                test_loss.append(t2np(loss))
                test_labels.append(t2np(labels))

        test_loss = np.mean(np.array(test_loss))
        if c.verbose:
            print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, test_loss))

        test_labels = np.concatenate(test_labels)
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
        anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))
        score_obs.update(roc_auc_score(is_anomaly, anomaly_score), epoch,
                         print_score=c.verbose or epoch == c.meta_epochs - 1)
        """
    #if c.grad_map_viz:
    #    export_gradient_maps(model, test_loader, optimizer, -1)

    #if c.save_model:
    #    model.to('cpu')
    #    save_model(model, c.modelname)
    #    save_weights(model, c.modelname)
    #return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ------------------------------------------------------------------------------------------------------------------
    # [1] data & transformation settings
    parser.add_argument('--dataset_path', type=str, default='dummy_dataset')
    parser.add_argument('--class_name', type=str, default='dummy_class')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--n_transforms', type=int, default=4,
                        help='per sample, number of transformations')
    parser.add_argument('--n_transforms_test', type=int, default=64,
                        help='per sample, number of transformations in test set')
    parser.add_argument('--img_size', type=tuple, default=(448, 448))
    parser.add_argument('--transf_rotations', action = 'store_true')
    parser.add_argument('--degrees', type=int, default = 180)
    parser.add_argument('--transf_brightness', type=float, default = 0.0)
    parser.add_argument('--transf_contrast', type=float, default = 0.0)
    parser.add_argument('--transf_saturation', type=float, default = 0.0)
    parser.add_argument('--transf_hue', type=float, default = 0.0)
    parser.add_argument('--norm_mean', type=float, default = [0.485, 0.456, 0.406])
    parser.add_argument('--norm_std', type=float, default = [0.229, 0.224, 0.225])

    # ------------------------------------------------------------------------------------------------------------------
    # [2] model
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_scales', type=int, default=3,
            help='number of scales at which features are extracted, img_size is the highest - others are //2, //4,...')
    parser.add_argument('--n_coupling_blocks', type=int, default=8)
    parser.add_argument('--clamp_alpha', type=int, default=3)
    parser.add_argument('--fc_internal', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.0)

    # ------------------------------------------------------------------------------------------------------------------
    # [3] model
    parser.add_argument('--meta_epochs', type=int, default=24)
    parser.add_argument('--sub_epochs', type=int, default=8)
    parser.add_argument('--hide_tqdm_bar', action='store_true')
    args = parser.parse_args()
    main(args)