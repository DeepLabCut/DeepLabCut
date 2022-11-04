import torch

def test_train_valid_call():

    tmp_model = torch.nn.Linear(3, 10)
    to_train_mode = getattr(tmp_model, 'train')
    to_train_mode()
    assert tmp_model.training == True
    to_valid_mode = getattr(tmp_model, 'eval')
    to_valid_mode()
    assert tmp_model.training == False