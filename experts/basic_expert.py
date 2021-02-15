class BasicExpert():
    TASK_CLASSIFICATION = 0
    TASK_REGRESSION = 1

    def post_process_ops(self, logits, specific_fcn):
        return specific_fcn(logits)

    def edge_specific_train(self, inp):
        return inp

    def edge_specific_eval(self, inp):
        return inp

    def get_task_type(self):
        return BasicExpert.TASK_REGRESSION

    def no_maps_as_input(self):
        return self.n_maps

    def no_maps_as_output(self):
        return self.n_maps
