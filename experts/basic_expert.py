class BasicExpert():
    def post_process_ops(self, logits, specific_fcn):
        return logits

    def edge_specific(self, inp):
        return inp

    def get_n_final_maps(self):
        return self.n_maps