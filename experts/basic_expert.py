class BasicExpert():
    TASK_CLASSIFICATION = 0
    TASK_REGRESSION = 1

    gt_train_transform = (lambda x: x)

    def get_task_type(self):
        return BasicExpert.TASK_REGRESSION

    def no_maps_as_nn_input(self):
        return self.n_maps

    def no_maps_as_nn_output(self):
        return self.n_maps

    def no_maps_as_ens_input(self):
        return self.n_maps
