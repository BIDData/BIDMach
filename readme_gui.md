```
git clone https://github.com/BIDData/BIDMach.git
cd BIDMach
git checkout gui
```

First download the models.tar.gz and data.tar.gz from the google drive.

Extract the models.tar.gz at the BIDMach folder.
Extract the data.tar.gz to somewhere you like.


Before running these scripts, config the `traindir` variable as the right data location. Change `pretrain_model_dir`  and `pretrain_discriminator_dir` if you extract models into different location.

DEMO for mnist:
```
./bidmach scripts/viz/mnist.ssc
```

DEMO for CIFAR:
```
./bidmach scripts/viz/testcifar_norm.ssc
```

After loading the scripts, type `s.launch` to start the MCMC process. Use `s.stop` to stop.

If you want to use discriminator (Require pixel value in [0,256)）, run command `o.clipping = true` in the shell. And set `base` as 0 in the UI.

By default, mnist.ssc set `clipping` as false, testcifar_norm.ssc set `clipping` as true.

Change L2 weight and discriminator weight to see the effect.

