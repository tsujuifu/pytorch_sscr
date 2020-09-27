# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Training Loop script"""
import os
import glob

import torch
from torch.utils.data import DataLoader

from geneva.data.datasets import DATASETS
from geneva.evaluation.evaluate import Evaluator
from geneva.utils.config import keys, parse_config
from geneva.utils.visualize import VisdomPlotter
from geneva.models.models import MODELS
from geneva.data import codraw_dataset
from geneva.data import clevr_dataset

from tqdm import tqdm
from geneva.utils.logger import Logger

import numpy as np
import nltk

import torchvision as TV

class Trainer():
    def __init__(self, cfg):
        img_path = os.path.join(cfg.log_path,
                                cfg.exp_name,
                                'train_images_*')
        if glob.glob(img_path):
            raise Exception('all directories with name train_images_* under '
                            'the experiment directory need to be removed')
        path = os.path.join(cfg.log_path, cfg.exp_name)

        self.model = MODELS[cfg.gan_type](cfg)
        
        if cfg.load_snapshot is not None:
            self.model.load_model(cfg.load_snapshot)
            print('Load model:', cfg.load_snapshot)
        self.model.save_model(path, 0, 0)
            
        shuffle = cfg.gan_type != 'recurrent_gan'

        self.dataset = DATASETS[cfg.dataset](path=keys[cfg.dataset], cfg=cfg, img_size=cfg.img_size)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=cfg.batch_size,
                                     shuffle=shuffle,
                                     num_workers=cfg.num_workers,
                                     pin_memory=True,
                                     drop_last=True)

        if cfg.dataset == 'codraw':
            self.dataloader.collate_fn = codraw_dataset.collate_data
        elif cfg.dataset == 'iclevr':
            self.dataloader.collate_fn = clevr_dataset.collate_data

        self.visualizer = VisdomPlotter(env_name=cfg.exp_name, server=cfg.vis_server)
        self.logger = Logger(cfg.log_path, cfg.exp_name)
        self.cfg = cfg

    def train(self):
        iteration_counter = 0
        for epoch in tqdm(range(self.cfg.epochs), ascii=True):
            if cfg.dataset == 'codraw':
                self.dataset.shuffle()

            for batch in self.dataloader:
                if iteration_counter >= 0 and iteration_counter % self.cfg.save_rate == 0:
                    
                    torch.cuda.empty_cache()
                    evaluator = Evaluator.factory(self.cfg, self.visualizer,
                                                  self.logger)
                    res = evaluator.evaluate(iteration_counter)
                    
                    print('\nIter %d:' % (iteration_counter))
                    print(res)
                    self.logger.write_res(iteration_counter, res)
                    
                    del evaluator

                iteration_counter += 1
                self.model.train_batch(batch,
                                       epoch,
                                       iteration_counter,
                                       self.visualizer,
                                       self.logger)
    
    def train_with_ctr(self):
        
        cfg = self.cfg
        
        if cfg.dataset == 'codraw':
            self.model.ctr.E.load_state_dict(torch.load('models/codraw_1.0_e.pt'))
        elif cfg.dataset == 'iclevr':
            self.model.ctr.E.load_state_dict(torch.load('models/iclevr_1.0_e.pt'))
        
        iteration_counter = 0
        for epoch in tqdm(range(self.cfg.epochs), ascii=True):
            if cfg.dataset == 'codraw':
                self.dataset.shuffle()

            for batch in self.dataloader:
                if iteration_counter >= 0 and iteration_counter % self.cfg.save_rate == 0:
                    
                    torch.cuda.empty_cache()
                    evaluator = Evaluator.factory(self.cfg, self.visualizer,
                                                  self.logger)
                    res = evaluator.evaluate(iteration_counter)
                    
                    print('\nIter %d:' % (iteration_counter))
                    print(res)
                    self.logger.write_res(iteration_counter, res)
                    
                    del evaluator

                iteration_counter += 1
                self.model.train_batch_with_ctr(batch,
                                                epoch,
                                                iteration_counter,
                                                self.visualizer,
                                                self.logger)
    
    
    def train_ctr(self):
        iteration_counter = 0
        
        with tqdm(range(self.cfg.epochs), ascii=True) as TQ:
            for epoch in TQ:
                if cfg.dataset == 'codraw':
                    self.dataset.shuffle()

                for batch in self.dataloader:
                    
                    loss = self.model.train_ctr(batch, epoch, iteration_counter, self.visualizer, self.logger)
                    TQ.set_postfix(ls_bh=loss)
                
                    if iteration_counter>0 and (iteration_counter%self.cfg.save_rate)==0:
                        torch.cuda.empty_cache()
                        
                        print('Iter %d: %f' % (iteration_counter, loss))
                        loss = self.eval_ctr(epoch, iteration_counter)
                        print('Eval: %f' % (loss))
                        print('')
                        self.logger.write_res(iteration_counter, loss)
                    
                    iteration_counter += 1
    
    def eval_ctr(self, epoch, iteration_counter):
        cfg = self.cfg
        
        dataset = DATASETS[cfg.dataset](path=keys[cfg.val_dataset], cfg=cfg, img_size=cfg.img_size)
        dataloader = DataLoader(dataset, 
                                batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        
        if cfg.dataset == 'codraw':
            dataloader.collate_fn = codraw_dataset.collate_data
        elif cfg.dataset == 'iclevr':
            dataloader.collate_fn = clevr_dataset.collate_data
        
        if cfg.dataset == 'codraw':
            dataset.shuffle()
        rec_loss = []
        for batch in dataloader:
            loss = self.model.train_ctr(batch, epoch, iteration_counter, self.visualizer, self.logger, is_eval=True)
            rec_loss.append(loss)
        
        loss = np.average(rec_loss)
        
        return loss
    
    def infer_ctr(self):
        
        cfg = self.cfg
        
        if cfg.dataset == 'codraw':
            self.model.ctr.E.load_state_dict(torch.load('models/codraw_1.0_e.pt'))
        elif cfg.dataset == 'iclevr':
            self.model.ctr.E.load_state_dict(torch.load('models/iclevr_1.0_e.pt'))
        
        dataset = DATASETS[cfg.dataset](path=keys[cfg.val_dataset], cfg=cfg, img_size=cfg.img_size)
        dataloader = DataLoader(dataset, 
                                batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        
        if cfg.dataset == 'codraw':
            dataloader.collate_fn = codraw_dataset.collate_data
        elif cfg.dataset == 'iclevr':
            dataloader.collate_fn = clevr_dataset.collate_data
        
        glove_key = list(dataset.glove_key.keys())
        
        for batch in dataloader:
            rec_out, loss = self.model.train_ctr(batch, -1, -1, self.visualizer, self.logger, 
                                                 is_eval=True, is_infer=True)
            
            rec_out = np.argmax(rec_out, axis=3)
            
            os.system('mkdir ins_result')
            
            for i in range(30):
                
                os.system('mkdir ins_result/%d' % (i))
                
                F = open('ins_result/%d/ins.txt' % (i), 'w')
                for j in range(rec_out.shape[1]):
                    print([glove_key[rec_out[i, j, k]] for k in range(rec_out.shape[2])])
                    print([glove_key[int(batch['turn_word'][i, j, k].detach().cpu().numpy())] for k in range(rec_out.shape[2])])
                    print()
                    
                    F.write(' '.join([glove_key[rec_out[i, j, k]] for k in range(rec_out.shape[2])]))
                    F.write('\n')
                    F.write(' '.join([glove_key[int(batch['turn_word'][i, j, k].detach().cpu().numpy())] for k in range(rec_out.shape[2])]))
                    F.write('\n')
                    F.write('\n')
                    
                    TV.utils.save_image(batch['image'][i, j].data, 'ins_result/%d/%d.png' % (i, j), normalize=True, range=(-1, 1))
                
                print('\n----------------\n')
                F.close()
            
            break
        
        os.system('tar zcvf ins_result.tar.gz ins_result')
    
    def infer_gen(self):
        
        cfg = self.cfg
        
        if cfg.dataset == 'codraw':
            self.model.ctr.E.load_state_dict(torch.load('models/codraw_1.0.pt'))
        elif cfg.dataset == 'iclevr':
            self.model.ctr.E.load_state_dict(torch.load('models/iclevr_1.0.pt'))
        
        dataset = DATASETS[cfg.dataset](path=keys[cfg.val_dataset], cfg=cfg, img_size=cfg.img_size)
        dataloader = DataLoader(dataset, 
                                batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        
        if cfg.dataset == 'codraw':
            dataloader.collate_fn = codraw_dataset.collate_data
        elif cfg.dataset == 'iclevr':
            dataloader.collate_fn = clevr_dataset.collate_data
        
        glove_key = list(dataset.glove_key.keys())
        
        for batch in dataloader:
            rec_out = self.model.infer_gen(batch)
            
            os.system('mkdir gen_result')
            
            for i in range(30):
                
                os.system('mkdir gen_result/%d' % (i))
                
                F = open('gen_result/%d/ins.txt' % (i), 'w')
                for j in range(rec_out.shape[1]):
                    F.write(' '.join([glove_key[int(batch['turn_word'][i, j, k].detach().cpu().numpy())] for k in range(batch['turn_word'].shape[2])]))
                    F.write('\n')
                    
                    TV.utils.save_image(batch['image'][i, j].data, 'gen_result/%d/_%d.png' % (i, j), normalize=True, range=(-1, 1))
                    TV.utils.save_image(torch.from_numpy(rec_out[i, j]).data, 'gen_result/%d/%d.png' % (i, j), normalize=True, range=(-1, 1))
                    
                F.close()
            
            break
        
        os.system('tar zcvf gen_result.tar.gz gen_result')

if __name__ == '__main__':
    cfg = parse_config()
    
    trainer = Trainer(cfg)
    
    # TRAIN
    trainer.train() # GeNeVA only
    # trainer.train_ctr() # train 
    # trainer.train_with_ctr() # train w/ CTC
    
    # INFERENCE
    # trainer.infer_ctr() # inference E
    # trainer.infer_gen() # inference G
