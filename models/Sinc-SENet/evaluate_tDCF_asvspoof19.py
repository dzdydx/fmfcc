import sys
import numpy as np
import eval_metrics as em
import matplotlib.pyplot as plt
import os

# Replace CM scores with your own scores or provide score file as the first argument.
cm_score_file =  'scores/output.txt' #'scores/cm_dev.txt'
# Replace ASV scores with organizers' scores or provide score file as the second argument.
asv_score_file = 'scores/asv_dev.txt'

det_file = '{}_det.txt'
sys_id = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09',
          'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19']

vc_id = ['A05', 'A06', 'A13', 'A14', 'A15', 'A17', 'A18', 'A19']
tts_id = ['A01', 'A02', 'A03', 'A04', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A16']

args = sys.argv
if len(args) > 1:
    if len(args) != 3:
        print('USAGE: python evaluate_tDCF_asvspoof19.py <CM_SCOREFILE> <ASV_SCOREFILE>')
        exit()
    else:
        cm_score_file = args[1]
        asv_score_file = args[2]
        det_file = det_file.format(cm_score_file.split('/')[-1])
        fig_file = cm_score_file.split('/')[-1].split('_')
        fig_file.pop()
        dev_or_eval = fig_file.pop()
        fig_file = '_'.join(fig_file)

# Fix tandem detection cost function (t-DCF) parameters
Pspoof = 0.05
cost_model = {
    'Pspoof': Pspoof,  # Prior probability of a spoofing attack
    'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
    'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
    'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
    'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
    'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
    'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
}

# Load organizers' ASV scores
asv_data = np.genfromtxt(asv_score_file, dtype=str)
asv_sources = asv_data[:, 0]
asv_keys = asv_data[:, 1]
asv_scores = asv_data[:, 2].astype(np.float)

# Load CM scores
cm_data = np.genfromtxt(cm_score_file, dtype=str)
cm_utt_id = cm_data[:, 0]
cm_sources = cm_data[:, 1]
cm_keys = cm_data[:, 2]
cm_scores = cm_data[:, 3].astype(np.float)

# Extract target, nontarget, and spoof scores from the ASV scores
tar_asv = asv_scores[asv_keys == 'target']
non_asv = asv_scores[asv_keys == 'nontarget']
spoof_asv = asv_scores[asv_keys == 'spoof']

# Extract bona fide (real human) and spoof scores from the CM scores
bona_cm = cm_scores[cm_keys == 'bonafide']
spoof_cm = cm_scores[cm_keys == 'spoof']

spoof_cm_tts = np.concatenate((cm_scores[cm_sources == tts_id[0]],
                               cm_scores[cm_sources == tts_id[1]],
                               cm_scores[cm_sources == tts_id[2]],
                               cm_scores[cm_sources == tts_id[3]],
                               cm_scores[cm_sources == tts_id[4]],
                               cm_scores[cm_sources == tts_id[5]],
                               cm_scores[cm_sources == tts_id[6]],
                               cm_scores[cm_sources == tts_id[7]],
                               cm_scores[cm_sources == tts_id[8]],
                               cm_scores[cm_sources == tts_id[9]],
                               cm_scores[cm_sources == tts_id[10]]
                               ))
# print(spoof_cm_tts)
spoof_cm_vc = np.concatenate((cm_scores[cm_sources == vc_id[0]],
                               cm_scores[cm_sources == vc_id[1]],
                               cm_scores[cm_sources == vc_id[2]],
                               cm_scores[cm_sources == vc_id[3]],
                               cm_scores[cm_sources == vc_id[4]],
                               cm_scores[cm_sources == vc_id[5]],
                               cm_scores[cm_sources == vc_id[6]],
                               cm_scores[cm_sources == vc_id[7]]
                              ))
# print(spoof_cm_vc)

# compute accuracy
n_tn = len(bona_cm[np.where(bona_cm<0.)])#tn
n_tp = len(spoof_cm[np.where(spoof_cm>0.)])#tp
n_fp = len(bona_cm[np.where(bona_cm>=0.)])#fp
n_fn = len(spoof_cm[np.where(spoof_cm<=0.)])#fn
n_tp1 = len(spoof_cm_vc[np.where(spoof_cm_vc>0.)])
n_tp2 = len(spoof_cm_tts[np.where(spoof_cm_tts>0.)])

# EERs of the standalone systems and fix ASV operating point to EER threshold
eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
eer_cm = em.compute_eer(spoof_cm, bona_cm, det_file)[0]
eer_cm_1 = em.compute_eer(spoof_cm_vc, bona_cm)[0]
eer_cm_2 = em.compute_eer(spoof_cm_tts, bona_cm)[0]


cm_eer_list = []
if dev_or_eval == 'dev':
    for id in range(6):
        tmp = []
        tmp.append(sys_id[id])
        tmp_eer = em.compute_eer(cm_scores[cm_sources == sys_id[id]], bona_cm)[0]
        tmp.append(tmp_eer)
        cm_eer_list.append(tmp)
else:
    for id in range(6, 19):
        tmp = []
        tmp.append(sys_id[id])
        tmp_eer = em.compute_eer(cm_scores[cm_sources == sys_id[id]], bona_cm)[0]
        tmp.append(tmp_eer)
        cm_eer_list.append(tmp)


[Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)


# Compute t-DCF
tDCF_curve, CM_thresholds = em.compute_tDCF(spoof_cm, bona_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)

# Minimum t-DCF
min_tDCF_index = np.argmin(tDCF_curve)
min_tDCF = tDCF_curve[min_tDCF_index]


print('ASV SYSTEM')
print('   EER            = {:8.5f} % (Equal error rate (target vs. nontarget discrimination)'.format(eer_asv * 100))
print('   Pfa            = {:8.5f} % (False acceptance rate of nontargets)'.format(Pfa_asv * 100))
print('   Pmiss          = {:8.5f} % (False rejection rate of targets)'.format(Pmiss_asv * 100))
print('   1-Pmiss,spoof  = {:8.5f} % (Spoof false acceptance rate)'.format((1 - Pmiss_spoof_asv) * 100))

print('\nCM SYSTEM')
print('   EER            = {:8.9f} % (Equal error rate for countermeasure)'.format(eer_cm * 100))
print('   ACC            = {:8.9f} % (Detection accuracy for countermeasure)'.format(((float)(n_tp+n_tn)/len(cm_scores)) * 100))

print('\nCM towards VC SYSTEM')
print('   EER            = {:8.9f} % (Equal error rate for countermeasure)'.format(eer_cm_1 * 100))
print('   ACC            = {:8.9f} % (Detection accuracy for countermeasure)'.format(((float)(n_tp1+n_tn)/(len(bona_cm)+len(spoof_cm_vc))) * 100))

print('\nCM towards TTS SYSTEM')
print('   EER            = {:8.9f} % (Equal error rate for countermeasure)'.format(eer_cm_2 * 100))
print('   ACC            = {:8.9f} % (Detection accuracy for countermeasure)'.format(((float)(n_tp2+n_tn)/(len(bona_cm)+len(spoof_cm_tts))) * 100))


print('\nCM SYSTEM DETAILS')
for e in cm_eer_list:
    print('   {}  {:8.9f}%'.format(e[0], e[1]*100))


print('\nTANDEM')
print('   min-tDCF       = {:8.9f}'.format(min_tDCF))


# Visualize ASV scores and CM scores
# ax = plt.subplot(121)
plt.figure()
plt.hist(tar_asv, histtype='step', density=True, bins=50, label='Target')
plt.hist(non_asv, histtype='step', density=True, bins=50, label='Nontarget')
plt.hist(spoof_asv, histtype='step', density=True, bins=50, label='Spoof')
plt.plot(asv_threshold, 0, 'o', markersize=10, mfc='none', mew=2, clip_on=False, label='EER threshold')
plt.legend()
plt.xlabel('ASV score')
plt.ylabel('Density')
plt.title('ASV score histogram')
plt.savefig(os.path.join('{}_{}'.format('temp_info', fig_file), '{}_{}_{}'.format(fig_file, dev_or_eval, 'asv.png')))

# ax = plt.subplot(122)
plt.figure()
plt.hist(bona_cm, histtype='step', density=True, bins=50, label='Bona fide')
plt.hist(spoof_cm, histtype='step', density=True, bins=50, label='Spoof')
plt.legend()
plt.xlabel('CM score')
#plt.ylabel('Density')
plt.title('CM score histogram')
plt.savefig(os.path.join('{}_{}'.format('temp_info', fig_file), '{}_{}_{}'.format(fig_file, dev_or_eval, 'cm.png')))

# Plot t-DCF as function of the CM threshold.
plt.figure()
plt.plot(CM_thresholds, tDCF_curve)
plt.plot(CM_thresholds[min_tDCF_index], min_tDCF, 'o', markersize=10, mfc='none', mew=2)
plt.xlabel('CM threshold index (operating point)')
plt.ylabel('Norm t-DCF');
plt.title('Normalized tandem t-DCF')
plt.plot([np.min(CM_thresholds), np.max(CM_thresholds)], [1, 1], '--', color='black')
plt.legend(('t-DCF', 'min t-DCF ({:.9f})'.format(min_tDCF), 'Arbitrarily bad CM (Norm t-DCF=1)'))
plt.xlim([np.min(CM_thresholds), np.max(CM_thresholds)])
plt.ylim([0, 1.5])

# plt.show()
plt.savefig(os.path.join('{}_{}'.format('temp_info', fig_file), '{}_{}_{}'.format(fig_file, dev_or_eval, 't-DCF.png')))