python zero_shot/dual_gate.py --ids-json results/exp2/ids_k20.json --modes per_domain --pairs 0.6112,0.0667 --tag exp2_k20_full_independent
python zero_shot/dual_gate.py --ids-json results/exp2/ids_k20.json --modes per_domain --pairs 0.6839,0.1394 --tag exp2_k20_full_global
python zero_shot/dual_gate.py --ids-json results/exp2/ids_k20.json --modes union --pairs 0.6310,0.6310 --tag exp2_k20_union
python zero_shot/dual_gate.py --ids-json results/exp2/ids_k50.json --modes per_domain --pairs 0.7206,0.1103 --tag exp2_k50_full_independent
python zero_shot/dual_gate.py --ids-json results/exp2/ids_k50.json --modes per_domain --pairs 0.7857,0.1754 --tag exp2_k50_full_global
python zero_shot/dual_gate.py --ids-json results/exp2/ids_k50.json --exclusive --modes per_domain --pairs 0.7206,0.1093 --tag exp2_k50_exclusive_independent
python zero_shot/dual_gate.py --ids-json results/exp2/ids_k50.json --exclusive --modes per_domain --pairs 0.7860,0.1747 --tag exp2_k50_exclusive_global
python zero_shot/dual_gate.py --ids-json results/exp2/ids_k50.json --modes union --pairs 0.7386,0.7386 --tag exp2_k50_union
python zero_shot/dual_gate.py --ids-json results/exp2/ids_k100.json --modes per_domain --pairs 0.7455,0.2041 --tag exp2_k100_full_independent
python zero_shot/dual_gate.py --ids-json results/exp2/ids_k100.json --modes per_domain --pairs 0.8146,0.2732 --tag exp2_k100_full_global
python zero_shot/dual_gate.py --ids-json results/exp2/ids_k100.json --exclusive --modes per_domain --pairs 0.7455,0.2034 --tag exp2_k100_exclusive_independent
python zero_shot/dual_gate.py --ids-json results/exp2/ids_k100.json --exclusive --modes per_domain --pairs 0.8146,0.2726 --tag exp2_k100_exclusive_global
python zero_shot/dual_gate.py --ids-json results/exp2/ids_k100.json --modes union --pairs 0.7739,0.7739 --tag exp2_k100_union
python zero_shot/dual_gate.py --ids-json results/exp2/ids_k200.json --modes per_domain --pairs 0.7661,0.2792 --tag exp2_k200_full_independent
python zero_shot/dual_gate.py --ids-json results/exp2/ids_k200.json --modes per_domain --pairs 0.8338,0.3469 --tag exp2_k200_full_global
python zero_shot/dual_gate.py --ids-json results/exp2/ids_k200.json --exclusive --modes per_domain --pairs 0.7647,0.2747 --tag exp2_k200_exclusive_independent
python zero_shot/dual_gate.py --ids-json results/exp2/ids_k200.json --exclusive --modes per_domain --pairs 0.8314,0.3414 --tag exp2_k200_exclusive_global
python zero_shot/dual_gate.py --ids-json results/exp2/ids_k200.json --modes union --pairs 0.8191,0.8191 --tag exp2_k200_union
