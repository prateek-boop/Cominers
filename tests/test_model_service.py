"""Real checkpoint and in-process HTTP contract checks, without mock predictions."""
import asyncio
import json
import unittest
import numpy as np
from api import create_app, SequenceRequest
from model_runtime import ModelRuntime


def flow(t=1700000000, features=None):
    return dict(src_ip='192.0.2.1', dst_ip='192.0.2.2', timestamp=t,
                features=features if features is not None else [10.0]*16)


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = ModelRuntime()

    def test_checkpoint_loaded_and_repeatable(self):
        self.assertEqual(self.runtime.metadata['epoch'], 5)
        self.assertEqual(self.runtime.num_classes, 14)
        flows = [flow(),flow(1700000200),flow(1700000400)]
        first = self.runtime.predict_sequence(flows)
        self.runtime.predict_sequence([flow(features=[999.]*16)]*3)
        self.assertEqual(first,self.runtime.predict_sequence(flows))
        for prediction in first:
            self.assertTrue(0 <= prediction['attack_probability'] <= 1)
            self.assertAlmostEqual(sum(prediction['stage_probabilities']), 1, places=5)

    def test_features_affect_future_not_first_event(self):
        a = self.runtime.predict_sequence([flow(features=[0.]*16)]*3, batch_size=1)
        b = self.runtime.predict_sequence([flow(features=[1000.]*16)]*3, batch_size=1)
        self.assertEqual(a[0],b[0])
        self.assertNotEqual(a[1]['attack_probability'],b[1]['attack_probability'])

    def test_transform_matches_processed_features(self):
        values = list(range(-8,8))
        raw = [flow(features=values)]*3
        transformed = [flow(features=np.sign(values)*np.log1p(np.abs(values)))]*3
        self.assertEqual(self.runtime.predict_sequence(raw, batch_size=1),
                         self.runtime.predict_sequence(transformed, 'signed_log1p', batch_size=1))

    def test_default_batches_use_history_after_200_flows(self):
        result = self.runtime.predict_sequence([flow()]*201)
        self.assertTrue(all(p['cold_start'] for p in result[:200]))
        self.assertFalse(result[200]['cold_start'])
        self.assertNotEqual(result[0]['attack_probability'],result[200]['attack_probability'])

    def test_invalid_inputs(self):
        for flows in ([], [flow(2),flow(1)], [flow(features=[0.]*15)],
                      [flow(features=[float('nan')]*16)]):
            with self.assertRaises(ValueError):
                self.runtime.predict_sequence(flows)
        for body in ({'flows': [flow()|{'src_ip':'bad'}]}, {'flows': []},
                     {'flows':[flow()], 'threshold':2},
                     {'flows':[flow(features=[float('inf')]*16)]}):
            with self.assertRaises(ValueError):
                SequenceRequest.model_validate(body)

    def test_real_http_contract(self):
        async def exercise():
            app = create_app()
            async with app.router.lifespan_context(app):
                async def request(method,path,body=None):
                    data = json.dumps(body).encode() if body is not None else b''
                    sent = []
                    async def receive():
                        return {'type':'http.request','body':data,'more_body':False}
                    async def send(message):
                        sent.append(message)
                    scope = dict(type='http',asgi={'version':'3.0'}, http_version='1.1',
                        method=method,scheme='http',path=path,raw_path=path.encode(),
                        query_string=b'',root_path='',headers=[(b'content-type',b'application/json')],
                        client=('127.0.0.1',123),server=('test',80))
                    await app(scope,receive,send)
                    status = next(m['status'] for m in sent if m['type']=='http.response.start')
                    content = b''.join(m.get('body',b'') for m in sent if m['type']=='http.response.body')
                    return status,json.loads(content)
                self.assertEqual((await request('GET','/health'))[0],200)
                status,info = await request('GET','/model')
                self.assertEqual(len(info['feature_order']),16)
                status,result = await request('POST','/predict/sequence',{'flows':[flow()]*3})
                self.assertEqual(status,200)
                self.assertEqual(result,self.runtime.predict_sequence([flow()]*3))
                self.assertEqual((await request('POST','/predict',flow()))[0],200)
                self.assertEqual((await request('POST','/predict',flow(features=[0.])))[0],422)
                self.assertEqual((await request('POST','/predict/sequence',{'flows':[flow(2),flow(1)]}))[0],422)
                status,schema = await request('GET','/openapi.json')
                self.assertEqual(status,200)
                self.assertIn('/predict/sequence',schema['paths'])
        asyncio.run(exercise())


if __name__ == '__main__':
    unittest.main()
