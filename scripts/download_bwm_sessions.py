"""Script for downloading list of sessions used for brainwide map."""

from one.api import ONE

import numpy as np


def main():

    one = ONE().setup()

    str_query = (  
       'session__project__name__icontains,ibl_neuropixel_brainwide_01,'
       'session__json__IS_MOCK,False,session__qc__lt'
       ',50,~json__qc,CRITICAL,'
       'session__extended_qc__behavior,1,'
       'json__extended_qc__tracing_exists,True,'
       '~session__extended_qc___task_stimOn_goCue_delays__lt,0.9,'
       '~session__extended_qc___task_response_feedback_delays__lt,0.9,'
       '~session__extended_qc___task_wheel_move_before_feedback__lt,0.9,'
       '~session__extended_qc___task_wheel_freeze_during_quiescence__lt,0.9,'
       '~session__extended_qc___task_error_trial_event_sequence__lt,0.9,'
       '~session__extended_qc___task_correct_trial_event_sequence__lt,0.9,'
       '~session__extended_qc___task_reward_volumes__lt,0.9,'
       '~session__extended_qc___task_reward_volume_set__lt,0.9,'
       '~session__extended_qc___task_stimulus_move_before_goCue__lt,0.9,'
       '~session__extended_qc___task_audio_pre_trial__lt,0.9')                  
        
    str_query2 = (
       'session__project__name__icontains,ibl_neuropixel_brainwide_01,'
       'session__json__IS_MOCK,False,session__qc__lt,50,'
       '~json__qc,CRITICAL,session__extended_qc__behavior,1,'       
       'json__extended_qc__tracing_exists,True,'
       'session__extended_qc___experimenter_task,PASS')        
            
    ins = np.concatenate([one.alyx.rest('insertions', 'list', django = x)
                          for x in [str_query, str_query2]])

    #eid_probe = set([x['session']+'_'+x['name'] for x in ins]) # pid via x['id']
    #ins = [x.split('_') for x in eid_probe] 
 

    # Print session EID values
    print(ins[0])



if __name__ == '__main__':
    main()

