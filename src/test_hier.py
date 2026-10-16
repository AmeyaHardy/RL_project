import json, sys, time, numpy as np
from stages import make_env_config, split_seeds
from evaluate import evaluate_policy, paired_compare, GreedyAdapter
from hier import ScriptedHierPolicy
st = int(sys.argv[1]); n = int(sys.argv[2])
cfg = make_env_config(st, 'shared')
t0 = time.time()
res = evaluate_policy(cfg, lambda e: ScriptedHierPolicy(e), split_seeds('val', n), 'scripted-hier', 'val', verbose=False)
print('time %.1fs (%.0f steps/s)' % (time.time()-t0, n*cfg.max_steps/(time.time()-t0)))
ref = json.load(open('greedy_s%d.json' % st))
sub = {k: v[:n] for k, v in ref['metrics'].items()}
ref = dict(ref, seeds=ref['seeds'][:n], metrics=sub)
for k in ['return','value','deliveries','persist_delivered','temp_delivered','temp_capture','collisions']:
    print('%-18s hier %7.2f | greedy %7.2f' % (k, np.nanmean(res['metrics'][k]), np.nanmean(ref['metrics'][k])))
print(paired_compare(res, ref))
print('per-seed hier  ', [round(x,1) for x in res['metrics']['return']])
print('per-seed greedy', [round(x,1) for x in ref['metrics']['return']])
