
train_cfg = dict(
    warmup = 5,
    lr = [0.004, 0.002, 0.0004, 0.00004, 0.000004],
    gamma = 0.1,
    end_lr = 1e-6,
    step_lr = dict(
        COCO = [90, 110, 130, 150, 160],
        ),
    )
 
 
def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size, cfg):
    global lr
 
    #在warmup 训练期间，学习率先行增大到初始学习率lr[0] = 0.04
    if epoch <= cfg.train_cfg.warmup:
        lr = cfg.train_cfg.end_lr + (cfg.train_cfg.lr[0]-cfg.train_cfg.end_lr)\
         * iteration / (epoch_size * cfg.train_cfg.warmup)
    
    #在warmup 之后，学习率按照设置的lr进行衰减，也可以自行设置指数衰减的形式
    else:
        for i in range(len(cfg.train_cfg.step_lr.COCO)):
            if cfg.train_cfg.step_lr.COCO[i]>=epoch:
                lr = cfg.train_cfg.lr[i]
                break
        # lr = cfg.train_cfg.init_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
