import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
import torch

from calibration.artifact import save_calibration, load_calibration
from explainability.model_explainer import permutation_shap
from forecasting.window_forecaster import WindowForecaster, STATE_SCHEMA
from training.forecast import fit, load_sources
from training.export_forecast import export
from training.data import sha256
from training.runner import TrainableTGN
from engine.stateful_tgn import StatefulCyberTGN
from features.flow_aggregator import FlowRecord
from graph.graph_formatter import GraphFormatter
from capture.sniffer import RawPacket
from pipeline import CyberDefensePipeline
from soar.policy import SOARPolicyEngine


class ModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_shapley_matches_additive_model_and_null_feature(self):
        result = permutation_shap(lambda x: x @ np.array([2., -3., 0.])+7,
                                  [4., 5., 9.], [1., 2., 0.], permutations=8)
        np.testing.assert_allclose(result['values'], [6., -9., 0.])
        self.assertAlmostEqual(result['base_value']+sum(result['values']), result['prediction'])
        self.assertAlmostEqual(result['additivity_residual'], 0.)

    def test_explanations_match_prediction_and_cannot_change_serving_state(self):
        torch.manual_seed(12)
        candidate = TrainableTGN(16,16,16,8,16,32,time_transform='signed_log1p')
        file = self.root/'model.pt';torch.save(candidate.checkpoint({}),file)
        engine = StatefulCyberTGN(file,history_capacity=32,capture_explanation_context=True)
        formatter = GraphFormatter(16)
        def batch(t):
            flow = FlowRecord(str(t),'192.0.2.1','192.0.2.2',1000,80,'tcp',t,t)
            flow.fwd_packet_lengths=[100,200];flow.fwd_timestamps=[t-1,t];flow.all_timestamps=[t-1,t]
            return formatter.format_flows([flow])
        engine.process_graph_batch(batch(100))
        records,_ = engine.process_graph_batch(batch(101))
        snapshot = engine.last_explanation_context
        with torch.no_grad():expected=float(snapshot(snapshot.x,snapshot.edges).sigmoid()[0])
        self.assertAlmostEqual(expected, records[0]['model_attack_probability'], places=6)
        memory_before = {k:v.clone() for k,v in engine.memory.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        feature = snapshot.explain_features(0, permutations=8)
        graph = snapshot.explain_edges(0,epochs=5)
        self.assertEqual(graph['status'],'computed')
        self.assertGreater(len(graph['edge_mask']),0)
        self.assertAlmostEqual(feature['prediction'],expected,places=6)
        self.assertLess(abs(feature['additivity_residual']),1e-6)
        torch.testing.assert_close(rng,torch.random.get_rng_state())
        for key,value in memory_before.items():torch.testing.assert_close(value,engine.memory.state_dict()[key])
        engine.process_graph_batch(batch(102))
        with torch.no_grad():self.assertAlmostEqual(float(snapshot(snapshot.x,snapshot.edges).sigmoid()[0]),expected,places=6)

    def test_forecasts_correct_only_matching_closed_window_and_reset_gaps(self):
        class StepPredictor(torch.nn.Module):
            def forward(self,x):return {f'forecast_{step*15}s':x+step for step in (1,2,3)}
        service = WindowForecaster(2,'a'*64)
        empty=service.observe_window(0,np.zeros(2),15)
        self.assertEqual(empty['status'],'untrained');self.assertFalse(empty['forecasts'])
        service = WindowForecaster(2,'a'*64);service.model=StepPredictor()
        with self.assertRaises(ValueError):service.observe_window(0,np.zeros(2),14)
        first=service.observe_window(0,np.zeros(2),15)
        self.assertFalse(first['reanchored'])
        second=service.observe_window(1,np.full(2,2.),30)
        self.assertTrue(second['reanchored']);self.assertEqual(second['prior_mse'],1.)
        np.testing.assert_allclose(second['forecasts'][0]['latent_state'], np.full(2,2+100.05/102.05),rtol=1e-6)
        gap=service.observe_window(3,np.full(2,10.),60)
        self.assertFalse(gap['reanchored']);self.assertTrue(gap['gap_reset'])
        np.testing.assert_allclose(gap['forecasts'][0]['latent_state'],[11.,11.])
        with self.assertRaises(ValueError):service.observe_window(3,np.zeros(2),60)

    def test_calibration_is_bound_to_checkpoint_protocol_and_class_count(self):
        path=self.root/'calibration.npz';protocol={'name':'test'}
        metadata=save_calibration(path,'a'*64,protocol,np.linspace(0,.3,100),
                                 np.array([0,1,2]),np.eye(3),provenance={'capture_groups':['cal'],'source_sha256':'b'*64})
        binary,stage,_=load_calibration(path,'a'*64,protocol,3,.10)
        self.assertTrue(binary.is_calibrated);self.assertTrue(stage.is_calibrated)
        self.assertEqual(metadata['diagnostics']['stage_support'],[1,1,1])
        for digest,proto,classes,alpha in [('b'*64,protocol,3,.1),('a'*64,{'name':'different'},3,.1),('a'*64,protocol,8,.1),('a'*64,protocol,3,.05)]:
            with self.assertRaises(ValueError):load_calibration(path,digest,proto,classes,alpha)
        with self.assertRaises(FileExistsError):save_calibration(path,'a'*64,protocol,[.1],provenance={'capture_groups':['cal'],'source_sha256':'b'*64})

    def test_soar_requires_model_and_stage_evidence_and_rejects_ambiguous_sets(self):
        mitigator=Mock();policy=SOARPolicyEngine(mitigator)
        alert=SimpleNamespace(conformal_confirmed=True,attack_probability=.99,stage_name='lateral_movement',
                              alert_id='test',src_ip='192.0.2.1',dst_ip='192.0.2.2',dst_port=445,
                              extra={'model_attack_probability':.99,'stage_validation_passed':True,
                                     'stage_conformal':{'is_calibrated':True,'conformal_set':['lateral_movement']}})
        actions=policy.handle_alert(alert);self.assertEqual(len(actions),2)
        for bad in [[],['benign','lateral_movement'],['stage_9'],['impact']]:
            alert.extra['stage_conformal']['conformal_set']=bad
            self.assertEqual(policy.handle_alert(alert),[])
        alert.extra['stage_conformal']['conformal_set']=['lateral_movement']
        alert.extra['stage_validation_passed']=False
        self.assertEqual(policy.handle_alert(alert),[])
        alert.extra['stage_validation_passed']=True;alert.extra['model_attack_probability']=.2
        self.assertEqual(policy.handle_alert(alert),[])

    def test_pipeline_retains_dns_and_closes_windows_without_random_forecasts(self):
        observed=[]
        pipeline=CyberDefensePipeline(forensics_dir=self.root/'pcap',ledger_file=self.root/'ledger',
                                      explanations_per_batch=0,window_observer=lambda *args:observed.append(args))
        packet=RawPacket(100.,'192.0.2.1','192.0.2.53',40000,53,'udp',80)
        self.assertFalse(pipeline.noise_filter.is_noise(packet))
        self.assertFalse(pipeline.graph_pruner.should_prune(packet))
        pipeline.ingest_packet(packet)
        self.assertFalse(observed)
        result=pipeline.tick(110.)
        self.assertEqual(result.flows_processed,1)
        closed=result.forecast_results
        self.assertEqual(len(observed),1);self.assertEqual(observed[0][0],6)
        self.assertEqual(closed['status'],'untrained');self.assertFalse(closed['forecasts'])
        pipeline.tick(120.);self.assertEqual(len(observed),1)

    def test_forecast_training_keeps_capture_boundaries_and_rejects_changed_sources(self):
        sources=[]
        for i,split in enumerate(['train','validation']):
            file=self.root/(split+'.npz')
            states=np.arange(24,dtype=np.float32).reshape(12,2)/10+i
            windows=np.r_[np.arange(6),np.arange(10,16)]
            np.savez(file,states=states,windows=windows)
            sources.append(dict(path=file.name,sha256=sha256(file),group=split,split=split))
        manifest=self.root/'manifest.json'
        manifest.write_text(json.dumps(dict(encoder_sha256='a'*64,state_schema=STATE_SCHEMA,window_seconds=15,flow_batch_size=1,sources=sources)))
        _,data=load_sources(manifest)
        self.assertEqual(len(data['train'][0][1]),6) # No examples bridge the missing windows.
        with contextlib.redirect_stdout(io.StringIO()):fit(manifest,self.root/'trained',epochs=2,patience=1,hidden_dim=4)
        checkpoint=torch.load(self.root/'trained/best.pt',weights_only=True)
        self.assertEqual(checkpoint['encoder_sha256'],'a'*64)
        self.assertIn('persistence_normalized_mse',checkpoint['validation']['per_capture']['validation'])
        self.assertFalse(checkpoint['deployment_approved'])
        with (self.root/'train.npz').open('ab') as file:file.write(b'changed')
        with self.assertRaises(ValueError):load_sources(manifest)

    def test_idle_flow_reordering_does_not_leak_newer_windows(self):
        observed=[]
        pipeline=CyberDefensePipeline(forensics_dir=self.root/'pcap',ledger_file=self.root/'ledger',
                                      enable_incidents=False,explanations_per_batch=0,
                                      window_observer=lambda *args:observed.append(args))
        # The UDP flow last seen at t=44 expires after the TCP RST at t=46.
        # It must nevertheless update memory before the newer completed flow.
        pipeline.ingest_packet(RawPacket(44.,'192.0.2.1','192.0.2.53',40000,53,'udp',80))
        pipeline.ingest_packet(RawPacket(46.,'192.0.2.2','192.0.2.3',40001,80,'tcp',80,tcp_flags={'RST':True}))
        pipeline.ingest_packet(RawPacket(52.,'192.0.2.4','192.0.2.5',40002,80,'tcp',80,tcp_flags={'RST':True}))
        pipeline.flush_and_process()
        self.assertEqual([row[0] for row in observed],[2])
        self.assertEqual(pipeline.tgn_engine.total_events_seen,3)

    def test_pcap_export_produces_compatible_windows_without_incidents(self):
        from scapy.all import Ether, IP, UDP, Raw, wrpcap
        torch.manual_seed(4)
        candidate=TrainableTGN(16,16,16,8,16,32,time_transform='signed_log1p')
        checkpoint=self.root/'graph.pt';torch.save(candidate.checkpoint({}),checkpoint)
        sources=[]
        for offset,split in enumerate(['train','validation']):
            packets=[]
            for i in range(9):
                packet=Ether(src='02:00:00:00:00:01',dst='02:00:00:00:00:02')/IP(src='192.0.2.1',dst='192.0.2.53')/UDP(sport=40000+i,dport=53)/Raw(bytes([offset+1])*20)
                packet.time=offset*1000+i*15+1;packets.append(packet)
            path=self.root/(split+'.pcap');wrpcap(str(path),packets)
            sources.append(dict(path=path.name,sha256=sha256(path),split=split,group=split))
        manifest=self.root/'pcaps.json';manifest.write_text(json.dumps(dict(sources=sources)))
        with contextlib.redirect_stdout(io.StringIO()):result=export(manifest,checkpoint,self.root/'windows')
        self.assertEqual(result['encoder_sha256'],sha256(checkpoint))
        _,data=load_sources(self.root/'windows/manifest.json')
        self.assertTrue(data['train']);self.assertTrue(data['validation'])


if __name__ == '__main__':unittest.main()
