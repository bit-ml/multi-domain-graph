import lpips
import pylab
import torch

spatial = True
loss_fn = lpips.LPIPS(net='alex', spatial=spatial)

ex_ref = lpips.im2tensor(lpips.load_image('./sim_imgs/ex_ref.png'))
ex_p0 = lpips.im2tensor(lpips.load_image('./sim_imgs/ex_p0.png'))
ex_p1 = lpips.im2tensor(lpips.load_image('./sim_imgs/ex_p1.png'))

ex_d0 = loss_fn.forward(ex_ref, ex_p0)
ex_d1 = loss_fn.forward(ex_ref, ex_p1)

if not spatial:
    print('Distances: (%.3f, %.3f)' % (ex_d0, ex_d1))
else:
    print('Distances: (%.3f, %.3f)' % (ex_d0.mean(), ex_d1.mean()))
    # The mean distance is approximately the same as the non-spatial distance

    # Visualize a spatially-varying distance map between ex_p0 and ex_ref
    pylab.imshow(ex_d0[0, 0, ...].data.cpu().numpy())
    pylab.show()
