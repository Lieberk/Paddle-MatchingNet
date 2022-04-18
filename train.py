import numpy as np
import os
import paddle
import paddle.optimizer as optim
from data.datamgr import SetDataManager
from methods.matchingnet import MatchingNet
from configs import get_resume_file, model_dict, parse_args, get_logger


def train(base_loader, val_loader, logger, model, optimization, start_epoch, stop_epoch, params):
    if optimization == 'Adam':
        optimizer = optim.Adam(parameters=model.parameters())
    else:
        raise ValueError('Unknown optimization, please define by yourself')
    max_acc = 0
    for epoch in range(start_epoch, stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader, logger, optimizer)  # model are called by reference, no need to return
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        acc = model.test_loop(val_loader, logger)
        if acc > max_acc:
            # with DB index
            logger.info("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            paddle.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            paddle.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

    return model


if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args('train')

    base_file = params.data_dir + 'base.json'
    val_file = params.data_dir + 'val.json'
    image_size = 84
    optimization = 'Adam'

    if params.stop_epoch == -1:
        if params.n_shot == 1:
            params.stop_epoch = 300
        elif params.n_shot == 5:
            params.stop_epoch = 400
        else:
            params.stop_epoch = 600  # default

    if params.method in ['matchingnet']:
        n_query = max(1, int(
            16 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
        base_datamgr = SetDataManager(image_size, n_query=n_query, **train_few_shot_params)
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

        test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(image_size, n_query=n_query, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

        model = MatchingNet(model_dict[params.model], **train_few_shot_params)
    else:
        raise ValueError('Unknown method')

    params.checkpoint_dir = '%s/checkpoint/%s/%s_%s' % (params.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = paddle.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_dict(tmp['state'])

    logger = get_logger(
        filename=os.path.join(params.log_path, 'log.txt'),
        logger_name='master_logger')
    model = train(base_loader, val_loader, logger, model, optimization, start_epoch, stop_epoch, params)
