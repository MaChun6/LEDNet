
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

            if save_img and idx%100==0:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')

                imwrite(tensor2img_fast(visuals['result'], min_max=(-1, 1)), save_img_path)
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}_mask.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}_mask.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}_mask.png')

                imwrite(tensor2img_fast(visuals['outmask']), save_img_path)

            if with_metrics:
                # calculate metrics
                if self.opt['val']['use_image']:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        metric_data = dict(img1=sr_img, img2=gt_img)
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
                else:
                    # img1= visuals['result']
                    # img2= visuals['gt']
                    # weight=self.mask
                    # print(f'{img1.shape}, {img2.shape}, {weight.shape}, {img1.max()}, {img2.max()}, {weight.max()}')
                    for name, opt_ in self.opt['val']['metrics'].items():
                        metric_data = dict(img1=visuals['result'], img2= visuals['gt'], weight=self.mask)
                        _metric = calculate_metric(metric_data, opt_)
                        if _metric == float('inf'):
                            print(f'Warning: {self.gt_path} and {self.lq_path} is inf or nan.')
                        self.metric_results[name] += _metric
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
            # if idx>10:
            #     break
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        if hasattr(self, 'outmask'):
            out_dict['outmask'] = self.outmask.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
