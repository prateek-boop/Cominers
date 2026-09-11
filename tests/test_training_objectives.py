"""Independent numerical checks for losses, classification metrics and targets."""
import unittest
import numpy as np
import torch
from sklearn import metrics as reference
from training.runner import metrics, choose_threshold, report_predictions
from training.assessment import wilson_interval, assess_protocol
from training.losses import supervised_loss


class ObjectiveTests(unittest.TestCase):
    def test_weighted_binary_loss_and_gradient_against_formula(self):
        x = torch.tensor([-1000.,-2.,0.,3.,1000.],dtype=torch.float64,requires_grad=True)
        y = torch.tensor([1.,0.,1.,0.,1.],dtype=torch.float64)
        stage = torch.randn(5,8,dtype=torch.float64,requires_grad=True)
        loss,binary,aux,denominator = supervised_loss(x,stage,y,torch.full((5,),-1),
            torch.tensor(3.),torch.ones(8,dtype=torch.float64),.2)
        expected = np.mean((1-y.numpy())*np.logaddexp(0,x.detach().numpy()) + 3*y.numpy()*np.logaddexp(0,-x.detach().numpy()))
        self.assertAlmostEqual(float(binary.detach()),expected)
        loss.backward()
        expected_grad = ((1-y)*x.detach().sigmoid() - 3*y*(1-x.detach().sigmoid()))/len(x)
        torch.testing.assert_close(x.grad,expected_grad)
        self.assertIsNone(stage.grad)
        self.assertEqual(float(aux),0.)
        self.assertEqual(float(denominator),0.)

    def test_unknown_stages_cannot_change_loss_or_gradients(self):
        x = torch.tensor([.1,.2,.3],requires_grad=True)
        logits = torch.randn(3,8,requires_grad=True)
        labels = torch.tensor([0.,1.,1.])
        stages = torch.tensor([0,-1,3])
        weights = torch.arange(1,9,dtype=torch.float32)
        loss,_,stage,denom = supervised_loss(x,logits,labels,stages,torch.tensor(1.),weights,.2)
        expected = -(logits.log_softmax(-1)[0,0]+4*logits.log_softmax(-1)[2,3])/5
        torch.testing.assert_close(stage,expected)
        self.assertEqual(float(denom),5.)
        loss.backward()
        self.assertEqual(int(logits.grad[1].count_nonzero()),0)
        self.assertGreater(int(logits.grad[0].count_nonzero()),0)

    def test_metrics_match_sklearn_with_imbalance_ties_and_extremes(self):
        rng = np.random.default_rng(81)
        for prevalence in (.01,.1,.5,.99):
            y = np.r_[0,1,(rng.random(298)<prevalence).astype(int)]
            p = rng.integers(0,11,len(y))/10
            r = metrics(y,p,.6)
            predicted = p>=.6
            expected = dict(precision=reference.precision_score(y,predicted,zero_division=0),
                recall=reference.recall_score(y,predicted,zero_division=0),f1=reference.f1_score(y,predicted),
                f2=reference.fbeta_score(y,predicted,beta=2),mcc=reference.matthews_corrcoef(y,predicted),
                accuracy=reference.accuracy_score(y,predicted),balanced_accuracy=reference.balanced_accuracy_score(y,predicted),
                average_precision=reference.average_precision_score(y,p),roc_auc=reference.roc_auc_score(y,p),
                brier=reference.brier_score_loss(y,p),log_loss=reference.log_loss(y,np.clip(p,1e-15,1-1e-15)))
            for key,value in expected.items():
                self.assertAlmostEqual(r[key],value,places=10,msg=key)
            self.assertEqual(r['confusion_matrix'],reference.confusion_matrix(y,predicted).tolist())

    def test_zero_false_alarms_does_not_mean_zero_uncertainty(self):
        low,high = wilson_interval(0,100)
        self.assertAlmostEqual(low,0.)
        self.assertGreater(high,.03)
        self.assertLess(high,.04)
        self.assertIsNone(wilson_interval(0,0))

    def test_minimum_recall_rejects_high_precision_low_coverage(self):
        y,p = np.array([1,0,1,1,1]),np.array([.9,.8,.7,.6,.5])
        self.assertTrue(choose_threshold(y,p,1.,0.)[1])
        self.assertFalse(choose_threshold(y,p,1.,0.,.95)[1])

    def test_bad_capture_cannot_hide_in_aggregate(self):
        good = metrics(np.array([0,1]*100),np.array([.1,.9]*100))
        bad = metrics(np.array([0,1]),np.array([.1,.1]))
        good['per_capture'] = {'good':dict(good),'missed_attack':bad}
        result = assess_protocol(good,.95,.95,.001)
        self.assertTrue(result['aggregate']['empirical_targets_met'])
        self.assertFalse(result['empirical_targets_met'])
        self.assertIn('recall',result['per_capture']['missed_attack']['failed_metrics'])

    def test_stage_confusion_ignores_unlabeled_rows(self):
        data = dict(label=np.array([0,1,1]),probability=np.array([.1,.9,.9]),stage=np.array([0,-1,3]),
                    stage_pred=np.array([0,7,2]),cold=np.array([True,False,False]),group=np.array(['g']*3),elapsed_seconds=0.)
        report = report_predictions(data,.5)
        self.assertEqual(sum(report['stage_support']),2)
        self.assertEqual(report['stage_confusion_matrix'][0][0],1)
        self.assertEqual(report['stage_confusion_matrix'][3][2],1)
        self.assertEqual(report['stage_per_class_f1']['credential_access'],0.)
        self.assertEqual(report['stage_per_class']['credential_access']['recall'],0.)
        self.assertIsNone(report['stage_per_class']['credential_access']['precision'])
        self.assertEqual(report['stage_macro_f1_present_classes'],reference.f1_score([0,3],[0,2],average='macro'))
