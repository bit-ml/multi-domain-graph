class BasicExpert():
    TASK_CLASSIFICATION = 0
    TASK_REGRESSION = 1

    def post_process_ops(self, logits, specific_fcn):
        return specific_fcn(logits)

    def edge_specific(self, inp):
        return inp

    def normalize_output_fcn(self, outp):
        return outp.clamp(min=0, max=1)

    def get_task_type(self):
        return BasicExpert.TASK_REGRESSION

    def no_maps_as_nn_input(self):
        return self.n_maps

    def no_maps_as_nn_output(self):
        return self.n_maps

    def no_maps_as_ens_input(self):
        return self.n_maps
