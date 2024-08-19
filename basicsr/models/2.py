
    def test(self):
        if self.ema_decay > 0:
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output, self.outmask = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output, self.outmask = self.net_g(self.lq)
                _,_,H,W = self.mask.shape
                self.output = self.output[:,:,:H,:W]
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            # sr_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                # gt_img = tensor2img([visuals['gt']])
                del self.gt
            if 'outmask' in visuals:
                # outmask_img = tensor2img([visuals['outmask']])
                del self.outmask

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
