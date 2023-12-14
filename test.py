import torch
import ultralytics
import paddle


print(f'Cuda torch available - {torch.cuda.is_available()}')
print(f'Torch cuda version {torch.version.cuda}')
print('Ultralytics')
ultralytics.checks()
print('Paddle')
paddle.utils.run_check()
print(f'Paddle cuda version {paddle.version.cuda()}')
