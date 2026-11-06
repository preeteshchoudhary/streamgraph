"""Graph semantics and temporal-integrity tests, optional in baseline-only CI."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

HAS_GRAPH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torch_geometric") is not None
if HAS_GRAPH:
    import torch
    from streamgraph.data import Dataset
    from streamgraph.graph import (GraphSAGEClassifier, graph_tensors, masked_task_loss,
                                   matching_baselines, snapshot_layout, train_graphsage)
    from streamgraph.experiments import fit_source_scaler


def toy_dataset():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(8,165)).astype(np.float32)
    # Accessing the final-test feature rows would break training, making this
    # a meaningful check that final-test graphs are not transformed or scored.
    x[6:] = np.nan
    return Dataset(np.array([101,102,103,104,201,202,301,302]),
                   np.array([1,1,1,1,25,25,35,35]), x,
                   np.array([1,0,-1,0,1,0,1,0], dtype=np.int8),
                   np.array([[0,2,4,6],[1,3,5,7]]), {"fixture":"test-only"})


PROTOCOL = {"schema_version":1,"seed":42,"feature_view":"all","splits":{
    "source":[1,16],"adaptation":[17,24],"validation":[25,34],"test":[35,49]}}
SETTINGS = {"hidden_channels":8,"dropout":0.0,"learning_rate":0.001,"weight_decay":0.0,
            "epochs":2,"patience":1,"cpu_threads":1,"gradient_clip_norm":5.0,"edge_mode":"original"}


@unittest.skipUnless(HAS_GRAPH,"Install the optional torch and torch-geometric dependencies")
class GraphSemantics(unittest.TestCase):
    def test_edges_change_predictions_with_identical_node_features(self):
        torch.manual_seed(7)
        model=GraphSAGEClassifier(3,8,0.0).eval()
        x=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        edge=torch.tensor([[0,1],[1,2]])
        with torch.inference_mode():
            linked=model(x,edge)
            unlinked=model(x,torch.empty((2,0),dtype=torch.long))
        self.assertFalse(torch.allclose(linked[1:],unlinked[1:]))

    def test_node_relabelling_preserves_graph_predictions(self):
        torch.manual_seed(9)
        model=GraphSAGEClassifier(3,8,0.0).eval()
        x=torch.randn(4,3)
        edge=torch.tensor([[0,1,2],[1,2,3]])
        order=torch.tensor([2,0,3,1])
        inverse=torch.argsort(order)
        with torch.inference_mode():
            expected=model(x,edge)[order]
            actual=model(x[order],inverse[edge])
        self.assertTrue(torch.allclose(expected,actual,atol=1e-6))

    def test_unknown_label_logits_receive_no_supervised_gradient(self):
        logits=torch.tensor([[1.,0.],[0.,2.],[9.,-9.]],requires_grad=True)
        loss=masked_task_loss(logits,torch.tensor([0,1,-1]),torch.ones(2))
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[2],torch.zeros(2)))
        self.assertGreater(float(logits.grad[:2].abs().sum()),0)

    def test_snapshot_keeps_unknown_context_and_reindexes_directed_edges(self):
        data=toy_dataset()
        layout=snapshot_layout(data,1)
        self.assertEqual(layout.nodes.tolist(),[0,1,2,3])
        self.assertEqual(layout.edges.tolist(),[[0,2],[1,3]])
        scaler=fit_source_scaler(data.features,np.array([True,True,False,True,False,False,False,False]))
        x,edge,y=graph_tensors(data,layout,scaler,"all")
        self.assertEqual(x.shape,(4,165))
        self.assertEqual(y.tolist(),[1,0,-1,0])

    def test_cross_period_edges_are_rejected(self):
        data=toy_dataset()
        data.edge_index=np.array([[0,6],[1,0]])
        with self.assertRaisesRegex(ValueError,"Cross-snapshot"):
            snapshot_layout(data,1)

    def test_edge_free_ablation_has_no_fabricated_links(self):
        self.assertEqual(snapshot_layout(toy_dataset(),1,"none").edges.shape,(2,0))

    def test_end_to_end_run_never_accesses_final_test_graphs(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            run=train_graphsage(toy_dataset(),PROTOCOL,SETTINGS,Path(directory))
            result=json.loads((run/'results.json').read_text())
            self.assertFalse(result['final_test_evaluated'])
            self.assertEqual(result['fit_audit']['source_rows'],3)
            self.assertEqual(result['fit_audit']['source_snapshot_steps'],[1])
            self.assertEqual(result['fit_audit']['validation_snapshot_steps'],[25])
            self.assertEqual(result['fit_audit']['final_test_rows_used_in_fit_or_prediction'],0)
            self.assertEqual(result['validation']['known_nodes'],2)
            saved=torch.load(run/'models/graphsage.pt',weights_only=True)
            self.assertIn('state_dict',saved)
            self.assertEqual(saved['scaler_mean'].shape,(165,))

    def test_mismatched_tabular_results_are_not_treated_as_matched_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'baseline_wrong';folder.mkdir()
            (folder/'results.json').write_text(json.dumps({'dataset_fingerprint':'other','seed':42}))
            self.assertEqual(matching_baselines(Path(directory),PROTOCOL,'this-data'),[])


if __name__ == '__main__':
    unittest.main()
