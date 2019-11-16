// Sum before Layernorm

import BIDMach.networks.TransformerLT
import BIDMach.networks.layers._

val ddir = "/code/BIDMach/data/wikitext/"
val fname = ddir + "train/part%04d.imat.lz4"

val dict = loadCSMat(ddir + "wikitext_spm_vocab.txt")(?,0) on "막"

val (nn, opts) = TransformerLT.learner(fname);

opts.lrate = 5e-5f
opts.seqlength = 2048
opts.batchSize = 2048
opts.npasses = 40
opts.degree = 128
opts.decay = 0.999f
opts.depth = 16
opts.nheads = 8
opts.dim = 2048
opts.indim = opts.dim
opts.outdim = opts.dim
opts.dropout= 0.8f;
opts.normInit = 2f
opts.decay = 0.999f
opts.texp = 0f
opts.vel_decay = 0.8f;
opts.lrate = opts.lrate*(1-opts.vel_decay)
opts.gsq_decay = 0.999f;
opts.clip_grad_norm = 10f
opts.scoreType = SoftmaxOutputLayer.CrossEntropyScore
opts.pstep = 0.01f
opts.useCache = false
opts.useGPUcache = true
//opts.resScale = 0.9f
//opts.resLinks = 2 \ 4 on 5 \ 7 on 9 \ 11 on 12 \ 14
//opts.resLinks = 4 \ 8

val lrfinal = opts.lrate.v
val lrinit = lrfinal / 2

def lr_update(ipass:Float, istep:Float, frac:Float):Float = {
  val lr = if (ipass < 1) { 
    lrinit + frac * (lrfinal - lrinit)
  } else { 
    lrfinal
  }
  opts.lrate = lr;
  lr
}

opts.lr_policy = lr_update _;

opts.logfile = "logTrans_d%d_n%d_m%d_lr%7.6f.txt" format (opts.degree, opts.depth, opts.dim, opts.lrate.v)

val tt = nn.model.asInstanceOf[TransformerLT]

//nn.train
nn.launchTrain
Thread.sleep(6000)


val net = tt.txNets(0)
val fe = tt.frontEnd
val be = tt.backEnd

