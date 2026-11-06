# Next milestone: DANN/GRL after GraphSAGE

The graph reference and edge-free control are implemented. See docs/GRAPHSAGE.md and docs/VALIDATION_M2.md. Their initial results are development measurements; domain adaptation is a hypothesis to test rather than a guaranteed improvement.

1. Reuse the GraphSAGE encoder and task head. Source task labels remain restricted to steps 1-16.
2. Add a domain classifier with a gradient reversal layer. Use steps 17-24 as a historical unlabelled adaptation prefix, keeping its task labels hidden.
3. Minimize source classification loss and domain-head classification loss. Reverse the domain gradient only at the encoder interface.
4. Balance domain batches, sweep a small validation-selected alignment strength, and compare with lambda = 0 under matched task-label budgets.
5. Use validation 25-34 for model selection, and leave final test 35-49 untouched until the protocol is frozen.
6. Report task/domain loss, illicit AP and recall, memory, and update cost. Low domain accuracy alone does not prove useful invariance.

A source-only generalization arm uses a different information budget and must be compared separately.

After offline experiments are reproducible, add PySpark replay of completed snapshots and a committed graph-batch handoff. Record throughput, handoff/inference time, lag, and memory separately. Begin with monitoring and periodic updates under a declared label delay; drift-triggered updates and source replay are controlled extensions. Spark preprocessing does not automatically distribute GNN training.
