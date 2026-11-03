"""Choose a saved same-cache output using only request-local training state."""


def select_answer_arm(mode, verified_practice_questions, accepted_steps, teacher_main_signal_used):
    if (type(verified_practice_questions) is not int or verified_practice_questions<0 or
        type(accepted_steps) is not int or accepted_steps<0 or
        type(teacher_main_signal_used) is not bool):
        raise ValueError('Valid practice and optimizer state is required')
    if mode=='original':return 'trained'
    if mode=='teacher_format':return 'trained_format'
    if mode=='practice_gated_format':
        # The main path precedes practice when present. Only select the trained
        # output when at least one independently verified practice path was
        # actually reached by the joint optimizer.
        minimum_steps=2 if teacher_main_signal_used else 1
        use_update=verified_practice_questions>0 and accepted_steps>=minimum_steps
        return 'trained_format' if use_update else 'zero_format'
    raise ValueError('Unknown selected-cache output policy')
