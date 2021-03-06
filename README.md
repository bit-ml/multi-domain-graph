# multi-domain-graph
Temporal Multi-Domain Graph

other materials/decisions:
https://drive.google.com/drive/folders/1urSH618xfQpYFCgU406uVMjcMj1pOTQ8?usp=sharing

### clone repo
git clone ....
git submodule update --init --recursive

### download models
see experts/models/models_path.txt

### setup tracking
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
set settings.network_path = '.........multi-domain-graph/experts/models/'  # Where tracking networks are stored.

python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"


### Implementation Steps
- [ ] Choose/Integrate 3 tasks + experts on them (OF-RAFT eccv2020, depth-SGDepth eccv2020, edges)
- [ ] Choose/Integrate training and testing datasets
- [ ] Train 1hop edges
- [ ] Remove edges for ill-posed transformations
- [ ] Train 2hop edges
- [ ] Cycle over multi-iters for edge removal
- [ ] Add temporal component  
[ ] v1. temporal component per node volume/lstm  
[ ] v2. (maybe) each edge net at frame t receives as input also the global graph state at frame t-1  
- [ ] Add more tasks
- [ ] Ablation experiments  
[ ] spatial vs temporal v1-vn   
[ ] number of tasks
- [ ] Label propagation expriments  
[ ] on segmentation  
