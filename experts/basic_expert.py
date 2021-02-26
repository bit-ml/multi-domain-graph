class BasicExpert():
    TASK_CLASSIFICATION = 0
    TASK_REGRESSION = 1

    def get_task_type(self):
        return BasicExpert.TASK_REGRESSION

    def no_maps_as_nn_input(self):
        return self.n_maps

    def no_maps_as_nn_output(self):
        return self.n_maps

    def no_maps_as_ens_input(self):
        return self.n_maps

    def postprocess_eval(self, nn_outp):
        '''
        POST PROCESSING eval - posprocess operations for evaluation (e.g. scale/normalize)
        '''
        return nn_outp

    def gt_train_transform(self, x):
        '''
        GT train - added for normals expert only
        '''
        return x

    def gt_eval_transform(self, x):
        '''
        GT eval - added for sem segm expert only
        '''
        return x

    def gt_to_inp_transform(self, x, n_classes):
        '''
        GT ensemble eval - added for sem segm expert only
        '''
        return x
