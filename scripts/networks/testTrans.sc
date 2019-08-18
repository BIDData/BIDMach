// Sum before Layernorm

import BIDMach.networks.TransformerLT
import BIDMach.networks.layers._

val ddir = "/code/BIDMach/data/wikitext/"
val fname = ddir + "train/part%04d.imat.lz4"

val dict = loadCSMat(ddir + "wikitext_spm_vocab.txt")(?,0) on "막"

val (nn, opts) = TransformerLT.learner(fname);

opts.lrate = 2e-5f
opts.seqlength = 1024
opts.batchSize = 2048
opts.npasses = 40
opts.degree = 256
opts.depth = 12
opts.nheads = 8
opts.dim = 1024*2
opts.indim = opts.dim
opts.outdim = opts.dim
opts.dropout=0.9f;
opts.texp = 0f
opts.vel_decay = 0.8f;
opts.lrate = opts.lrate*(1-opts.vel_decay)
opts.gsq_decay = 0.999f;
opts.clip_grad_norm = 10f
opts.scoreType = SoftmaxOutputLayer.CrossEntropyScore
opts.pstep = 0.01f
opts.useCache = false
opts.useGPUcache = true

opts.logfile = "logTrans_d%d_n%d_m%d_lr%7.6f.txt" format (opts.degree, opts.depth, opts.dim, opts.lrate.v)

val tt = nn.model.asInstanceOf[TransformerLT]

//nn.train
nn.launchTrain
Thread.sleep(4000)


val net = tt.txNets(0)
val fe = tt.frontEnd
val be = tt.backEnd

