import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
from training.smoke import make_fixture
from training.data import prepare as prepare_graph
from training.tabular import prepare,train,FeatureModel,iter_data,feature_columns
from model_runtime import FEATURE_NAMES


class TabularBaselineTests(unittest.TestCase):
    def test_headers_require_real_features_but_not_graph_identifiers(self):
        columns,label=feature_columns([*FEATURE_NAMES,'Label'])
        self.assertEqual(columns,FEATURE_NAMES)
        with self.assertRaises(ValueError):feature_columns([*FEATURE_NAMES[:-1],'Label'])

    def test_end_to_end_baseline_preserves_holdout_and_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=make_fixture(root/'graph',rows=32)
            prepare_graph(manifest,root/'graph-prepared',shard_size=16)
            sources=[]
            for split in ('train','validation'):
                raw=(root/'graph'/f'{split}.csv').read_text().replace(',attack,',',Bot,').replace(',benign,',',Benign,')
                path=root/(split+'.csv');path.write_text(raw)
                sources.append(dict(group='tab-'+split,split=split,path=path.name))
            path=root/'manifest.json';path.write_text(json.dumps(dict(sources=sources)))
            with contextlib.redirect_stdout(io.StringIO()):prepare(path,root/'prepared')
            expected=np.concatenate([x for _,x,_ in iter_data(root/'prepared',root/'graph-prepared','train')]).mean(0)
            args=argparse.Namespace(data=str(root/'prepared'),graph_data=str(root/'graph-prepared'),output=str(root/'run'),epochs=2,patience=2,hidden=8,threads=2,seed=42)
            import training.tabular as module
            original=module.iter_data
            splits=[]
            def tracked(*a,**kw):
                splits.append(a[2]);yield from original(*a,**kw)
            with mock.patch.object(module,'iter_data',tracked),contextlib.redirect_stdout(io.StringIO()):train(args)
            self.assertEqual(splits[-1],'test')
            self.assertEqual(splits.count('test'),1)
            checkpoint=torch.load(root/'run/best.pt',weights_only=True)
            np.testing.assert_allclose(checkpoint['state']['mean'].numpy(),expected,atol=1e-6)
            result=json.loads((root/'run/test-report.json').read_text())
            self.assertEqual(result['events'],32)
            self.assertFalse(result['deployment_approved'])
            self.assertEqual(checkpoint['model_type'],'feature_only_mlp')
            # A controlled comparison must neither read the holdout nor include
            # IDS2018 training rows when the CIC2017-only arm is selected.
            only_graph=list(iter_data(root/'prepared',root/'graph-prepared','train',training_source='cic2017'))
            self.assertTrue(only_graph)
            self.assertTrue(all(not group.startswith('tab-') for group,_,_ in only_graph))
            validation_groups={group for group,_,_ in iter_data(root/'prepared',root/'graph-prepared','validation',training_source='cic2017')}
            self.assertIn('tab-validation',validation_groups)
            args.output=str(root/'validation-only')
            args.validation_only=True
            args.training_source='cic2017'
            args.family_weighting=True
            splits.clear()
            with mock.patch.object(module,'iter_data',tracked),contextlib.redirect_stdout(io.StringIO()):train(args)
            self.assertNotIn('test',splits)
            self.assertFalse((root/'validation-only/test-report.json').exists())
            checkpoint=torch.load(root/'validation-only/best.pt',weights_only=True)
            np.testing.assert_allclose(checkpoint['state']['mean'].numpy(),np.concatenate([x for _,x,_ in only_graph]).mean(0),atol=1e-6)
            self.assertEqual(sum(checkpoint['training_counts']),sum(len(y) for _,_,y in only_graph))

    def test_linear_baseline_has_single_logit_per_flow(self):
        model=FeatureModel(0)
        self.assertEqual(tuple(model(torch.zeros(5,16)).shape),(5,))
