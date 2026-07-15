import numpy as np
from helpers.eta_learnability import deterministic_split,downsample_sum,shot_features,derived_targets

def test_split_is_disjoint_and_complete():
    parts=deterministic_split(101,7); joined=np.concatenate(parts)
    assert len(np.unique(joined))==101 and set(joined)==set(range(101))

def test_downsample_preserves_counts():
    x=np.arange(64).reshape(8,8)
    assert downsample_sum(x,4).sum()==x.sum()

def test_uniform_state_has_zero_shape_contrast():
    g=np.ones((8,8),int)*3; e=np.ones((8,8),int)*5
    summary,profile,weight=shot_features(g,e,1.,4)
    assert np.allclose(profile[weight>0],profile[weight>0][0]) and summary[5]==(g+e).sum() and weight.sum()==summary[5]

def test_final_mean_and_hidden_coordinates_are_independent_linear_combinations():
    eta=np.zeros((2,8)); eta[:,0]=[1,2]; eta[:,2]=[3,4]
    d=derived_targets(eta)
    assert np.allclose(d['mu_xf'],eta[:,0]+3.8*eta[:,2])
