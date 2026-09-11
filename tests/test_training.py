"""Training pipeline correctness, leakage guards and deployment compatibility."""
import argparse
import contextlib
import csv
import io
import json
import random
from unittest import mock
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from training.data import prepare, shards, read_metadata, resolve_columns
from training.smoke import make_fixture
from training.runner import train, evaluate, metrics, choose_threshold, TrainableTGN, predict, batches, validate_train_args
from model_runtime import ModelRuntime, FEATURE_NAMES
from engine.streaming_runtime import StreamingModelRuntime
from engine.stateful_tgn import StatefulCyberTGN
from features.flow_aggregator import FlowRecord
from graph.graph_formatter import FormattedGraphBatch


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = make_fixture(self.root,rows=12)

    def test_missing_features_and_ips_rejected(self):
        headers = ['src_ip','dst_ip','timestamp','label',*FEATURE_NAMES]
        for missing in ('src_ip',FEATURE_NAMES[3]):
            with self.assertRaises(ValueError):
                resolve_columns([h for h in headers if h!=missing],{})

    def test_capture_overlap_rejected(self):
        config = json.loads(self.manifest.read_text())
        config['sources'][1]['group'] = config['sources'][0]['group']
        self.manifest.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError,'capture group'):
            prepare(self.manifest,self.root/'prepared')

    def test_unknown_labels_rejected(self):
        config = json.loads(self.manifest.read_text())
        del config['labels']['attack']
        self.manifest.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError,'Unknown label'):
            prepare(self.manifest,self.root/'prepared')

    def test_cross_split_row_duplicate_rejected(self):
        with (self.root/'train.csv').open() as f:
            copied = list(csv.DictReader(f))[0]
        target = self.root/'validation.csv'
        with target.open() as f:
            rows = list(csv.DictReader(f))
        rows[0] = copied
        with target.open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(copied))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError,'crosses splits'):
            prepare(self.manifest,self.root/'prepared')

    def test_sorting_train_normalization_and_shard_integrity(self):
        path = self.root/'train.csv'
        with path.open() as f:
            rows = list(csv.DictReader(f))
        expected = np.log1p([[float(r[k]) for k in FEATURE_NAMES] for r in rows]).mean(axis=0)
        with path.open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(reversed(rows))
        metadata = prepare(self.manifest,self.root/'prepared',shard_size=5)
        np.testing.assert_allclose(metadata['train_mean'],expected)
        times = np.concatenate([s['timestamp'] for _,s in shards(self.root/'prepared','train')])
        self.assertTrue(np.all(np.diff(times)>=0))
        shard = self.root/'prepared'/metadata['shards'][0]['path']
        shard.write_bytes(shard.read_bytes()+b'corruption')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            list(shards(self.root/'prepared','train'))

    def test_metrics_handle_ties_and_infeasible_threshold(self):
        y,p = np.array([0,1]),np.array([.5,.5])
        result = metrics(y,p)
        self.assertEqual(result['roc_auc'],.5)
        self.assertEqual(result['average_precision'],.5)
        self.assertEqual(choose_threshold(y,p,.9,.001),(1.,False))

    def test_storage_shards_do_not_change_batches_or_predictions(self):
        for size in (5, 12):
            prepare(self.manifest,self.root/f'prepared-{size}',shard_size=size)
        for sizes in ([8], [1,8]):
            streams = [list(batches(self.root/f'prepared-{s}','train',sizes,random.Random(12))) for s in (5,12)]
            self.assertEqual([len(b['label']) for _,b in streams[0]], [len(b['label']) for _,b in streams[1]])
            for (_,a),(_,b) in zip(*streams):
                for key in a:
                    np.testing.assert_array_equal(a[key],b[key])
        model = TrainableTGN(16,16,16,8,16,32)
        predictions = [predict(model,self.root/f'prepared-{s}','train',8) for s in (5,12)]
        np.testing.assert_array_equal(predictions[0]['probability'],predictions[1]['probability'])

    def test_malformed_csv_rejected(self):
        path = self.root/'train.csv'
        lines = path.read_text().splitlines()
        lines[1] += ',unexpected'
        path.write_text('\n'.join(lines)+'\n')
        with self.assertRaisesRegex(ValueError,'Malformed CSV'):
            prepare(self.manifest,self.root/'prepared')

    def test_invalid_metric_inputs_rejected(self):
        for y,p in [([0,2],[.1,.2]),([0,1],[.5]),([0,1],[.1,float('nan')]),([0,1],[.1,1.1]),([],[])]:
            with self.assertRaises(ValueError):
                metrics(y,p)
            with self.assertRaises(ValueError):
                choose_threshold(y,p,.9,.1)
        with self.assertRaises(ValueError):
            metrics([0,1],[.1,.9],float('nan'))

    def test_rank_metrics_and_threshold_against_independent_oracles(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            y = np.r_[0,1,rng.integers(0,2,18)]
            p = rng.integers(0,6,len(y))/5
            positives,negatives = p[y==1],p[y==0]
            auc = np.mean([(a>b)+.5*(a==b) for a in positives for b in negatives])
            ap = np.mean([np.mean(y[p>=a]) for a in positives])
            actual = metrics(y,p)
            self.assertAlmostEqual(actual['roc_auc'],auc)
            self.assertAlmostEqual(actual['average_precision'],ap)
            eligible = []
            for threshold in np.unique(p):
                selected = p>=threshold
                tp,fp = sum(y[selected]==1),sum(y[selected]==0)
                if tp>0 and tp/(tp+fp)>=.7 and fp/sum(y==0)<=.2:
                    eligible.append(tp)
            threshold,feasible = choose_threshold(y,p,.7,.2)
            self.assertEqual(feasible,bool(eligible))
            if feasible:
                self.assertEqual(sum(y[p>=threshold]==1),max(eligible))


class TrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        manifest = make_fixture(cls.root,rows=32)
        prepare(manifest,cls.root/'prepared',shard_size=16)
        args = argparse.Namespace(data=str(cls.root/'prepared'),output=str(cls.root/'run'),
            seed=42,threads=2,device='cpu',batch_sizes='1,8',history_capacity=32,epochs=3,patience=3,
            learning_rate=.003,node_capacity=16,memory_dim=16,embedding_dim=16,time_dim=8,direct_dim=16,
            weight_decay=.0001,stage_weight=.2,grad_clip=1.,eval_batch_size=8,min_precision=.8,max_fpr=.2,alpha=.1)
        cls.args = args
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
        cls.checkpoint = cls.root/'run/best.pt'

    def test_training_artifacts_and_holdout_evaluation(self):
        history = json.loads((self.root/'run/training-history.json').read_text())
        self.assertLess(history[-1]['train_loss'],history[0]['train_loss'])
        model,ckpt = TrainableTGN.restore(self.checkpoint)
        self.assertEqual(ckpt['architecture_version'],2)
        self.assertEqual(int(model.memory.memory.memory.count_nonzero()),0)
        args = argparse.Namespace(data=str(self.root/'prepared'),checkpoint=str(self.checkpoint),
              operating_point=str(self.root/'run/operating-point.json'),output=str(self.root/'test-report.json'),
              batch_sizes='1,8',short_window=16,device='cpu',threads=2)
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate(args)
        result = json.loads(Path(args.output).read_text())
        self.assertEqual(len(result['reports']),4)
        self.assertTrue(all(r['events']==32 for r in result['reports'].values()))

    def test_v2_first_flow_uses_current_features(self):
        runtime = ModelRuntime(self.checkpoint)
        common = dict(src_ip='192.0.2.1',dst_ip='192.0.2.2',timestamp=1700000001)
        low = runtime.predict_sequence([dict(common,features=[1.]*16)])[0]
        high = runtime.predict_sequence([dict(common,features=[300.]*16)])[0]
        self.assertNotAlmostEqual(low['attack_probability'],high['attack_probability'],places=4)
        self.assertEqual(runtime.timestamp_encoding,'int64_seconds')

    def test_temporal_and_current_flow_parameters_are_learned(self):
        torch.manual_seed(self.args.seed)
        initial = TrainableTGN(16,16,16,8,16,32)
        trained,_ = TrainableTGN.restore(self.checkpoint)
        for name in ('memory.memory.gru.weight_ih','gnn.embedding.conv.lin_key.weight',
                     'prob_head.mlp.0.weight','current_flow_head.net.0.weight'):
            self.assertFalse(torch.equal(initial.state_dict()[name],trained.state_dict()[name]),name)

    def test_invalid_training_arguments_rejected_before_writing(self):
        for key,value in [('grad_clip',-1.),('stage_weight',-1.),('learning_rate',float('nan')),
                          ('weight_decay',float('inf')),('alpha',float('nan')),('threads',0),('seed',-1)]:
            args = argparse.Namespace(**vars(self.args))
            setattr(args,key,value)
            with self.assertRaises(ValueError):
                validate_train_args(args)

    def test_evaluation_rejects_changed_dataset(self):
        args = argparse.Namespace(data=str(self.root/'prepared'),checkpoint=str(self.checkpoint),
              operating_point=str(self.root/'run/operating-point.json'),output=str(self.root/'rejected-report.json'),
              batch_sizes='8',short_window=16,device='cpu',threads=2)
        metadata_path = self.root/'prepared/dataset.json'
        original = metadata_path.read_text()
        try:
            metadata_path.write_text(original+'\n')
            with self.assertRaisesRegex(ValueError,'differs from the checkpoint'):
                evaluate(args)
            self.assertFalse(Path(args.output).exists())
        finally:
            metadata_path.write_text(original)

    def test_failed_validation_cannot_pass_evaluation_gate(self):
        operating = json.loads((self.root/'run/operating-point.json').read_text())
        operating['validation_target_met'] = False
        operating['min_recall'] = .95
        path = self.root/'failed-operating-point.json'
        path.write_text(json.dumps(operating))
        args = argparse.Namespace(data=str(self.root/'prepared'),checkpoint=str(self.checkpoint),
            operating_point=str(path),output=str(self.root/'gated-report.json'),batch_sizes='8',
            short_window=16,device='cpu',threads=2,require_targets=True)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
            evaluate(args)
        self.assertEqual(error.exception.code,2)
        report = json.loads(Path(args.output).read_text())
        self.assertEqual(report['binary_evidence_status'],'insufficient_or_failed')
        self.assertFalse(report['deployment_approved'])

    def test_training_does_not_read_test_shards(self):
        import training.runner as runner
        original = runner.shards
        def guarded(directory, split, **kwargs):
            self.assertNotEqual(split,'test')
            yield from original(directory,split,**kwargs)
        args = argparse.Namespace(**vars(self.args))
        args.output = str(self.root/'no-test-access-run')
        args.epochs = 1
        with mock.patch.object(runner,'shards',guarded), contextlib.redirect_stdout(io.StringIO()):
            train(args)

    def test_temporal_predictions_and_history_rollover_match_runtimes(self):
        with (self.root/'test.csv').open() as f:
            rows = list(csv.DictReader(f))
        flows = [dict(src_ip=r['src_ip'],dst_ip=r['dst_ip'],timestamp=float(r['timestamp']),
                      features=[float(r[k]) for k in FEATURE_NAMES]) for r in rows]
        data = {k:np.concatenate([s[k] for _,s in shards(self.root/'prepared','test')])
                for k in ('src','dst','msg')}
        for capacity in (16,32):
            model,_ = TrainableTGN.restore(self.checkpoint)
            model.history_t = torch.empty(capacity,dtype=torch.long)
            model.history_msg = torch.empty((capacity,16))
            expected = predict(model,self.root/'prepared','test',8)['probability']
            stream = StreamingModelRuntime(self.checkpoint,max_history=capacity).score_flows(flows,batch_size=8)
            np.testing.assert_allclose(expected,[r['attack_probability'] for r in stream],atol=1e-6)
            stateful = StatefulCyberTGN(self.checkpoint,history_capacity=capacity)
            actual = []
            for start in range(0,len(flows),8):
                chunk = flows[start:start+8]
                records = [FlowRecord(str(start+i),f['src_ip'],f['dst_ip'],1234,80,'tcp',f['timestamp'],f['timestamp']) for i,f in enumerate(chunk)]
                batch = FormattedGraphBatch(torch.tensor(data['src'][start:start+8]),torch.tensor(data['dst'][start:start+8]),
                    torch.zeros(8,dtype=torch.long),torch.tensor(data['msg'][start:start+8]),records,np.array([f['features'] for f in chunk]))
                result,_ = stateful.process_graph_batch(batch)
                actual.extend(r['attack_probability'] for r in result)
            np.testing.assert_allclose(expected,actual,atol=1e-6)
            if capacity == 32:
                result = ModelRuntime(self.checkpoint).predict_sequence(flows,batch_size=8)
                np.testing.assert_allclose(expected,[r['attack_probability'] for r in result],atol=1e-6)

    def test_training_and_all_three_runtimes_agree(self):
        with (self.root/'test.csv').open() as f:
            rows = list(csv.DictReader(f))[:8]
        flows = [dict(src_ip=r['src_ip'],dst_ip=r['dst_ip'],timestamp=float(r['timestamp']),
                      features=[float(r[k]) for k in FEATURE_NAMES]) for r in rows]
        baseline = ModelRuntime(self.checkpoint).predict_sequence(flows,batch_size=8)
        stream = StreamingModelRuntime(self.checkpoint,max_history=32).score_flows(flows,batch_size=8)
        _, data = next(shards(self.root/'prepared','test'))
        model,_ = TrainableTGN.restore(self.checkpoint)
        src,dst = torch.tensor(data['src'][:8]),torch.tensor(data['dst'][:8])
        msg = torch.tensor(data['msg'][:8])
        with torch.no_grad():
            probabilities = model.score(src,dst,msg)[0].sigmoid().numpy()
        expected = [p['attack_probability'] for p in baseline]
        np.testing.assert_allclose(expected,probabilities,atol=1e-6)
        np.testing.assert_allclose(expected,[p['attack_probability'] for p in stream],atol=1e-6)
        records = [FlowRecord(str(i),f['src_ip'],f['dst_ip'],1234,80,'tcp',f['timestamp'],f['timestamp']) for i,f in enumerate(flows)]
        batch = FormattedGraphBatch(src,dst,torch.tensor([0]*8),msg,records,np.array([f['features'] for f in flows]))
        stateful = StatefulCyberTGN(self.checkpoint,history_capacity=32)
        result,_ = stateful.process_graph_batch(batch)
        np.testing.assert_allclose(expected,[p['attack_probability'] for p in result],atol=1e-6)
