import torch

if torch.cuda.is_available():
    _td = torch.device(torch.cuda.current_device())
    test_eq(default_device(-1), _td)
    test_eq(default_device(True), _td)
else:
    test_eq(default_device(False), torch.device('cpu'))
default_device(-1);