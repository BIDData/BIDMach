package BIDMach
import BIDMat.{Mat,SBMat,CMat,CSMat,Dict,DMat,FMat,IDict,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import java.io._
import BIDMach.datasources._
import BIDMach.models._
import BIDMach.updaters._

object Experiments {
  
  def clearbit(a:IMat) {
    var i = 0 
    while (i < a.length) {
      a.data(i) = a.data(i) & 0x7fffffff
      i += 1
    }
  }
}

object NYTIMES {
  def preprocess(dict:String, fname:String) {
    val cols = loadIMat(dict+"docword."+fname+".txt")
    val nrows = maxi(cols(?,1)).v
    val ncols = maxi(cols(?,0)).v
    val m = sparse(cols(?,1) - 1, cols(?,0) - 1, FMat(cols(?,2)), nrows, ncols)
    saveSMat(dict+fname+".smat.lz4", m)
  }
}

object DIGITS {
  def preprocess(dict:String, fname:String) {
    val mat = loadFMat(dict+fname+".txt")
    val srow = sum(abs(mat),2)
    val inds = IMat((cumsum(srow==0)-1)/660)
    val ii = find(srow > 0)
    val mm = mat(ii,?)
    val inn = inds(ii,?)
    saveFMat(dict+fname+".fmat.lz4", mm.t)
    val cats = zeros(mm.nrows, maxi(inn).v + 1)
    cats(icol(0->(inn.nrows)) + inn*mm.nrows) = 1f
    saveFMat(dict+fname+"_cats.fmat.lz4", cats.t)
  }
}

object RCV1 {
  
  def prepare(dict:String) {
    println("Preprocessing"); preprocess(dict)
    println("Making Sparse Data Matrix"); mksparse(dict)
    println("Making Category Matrix"); mkcats(dict)
  }
  
  def preprocess(dict:String) {
    val dictm = CSMat(loadSBMat(dict+"dict.gz"))
    val wc = loadIMat(dict+"wcount.gz")
    val a0 = loadIMat(dict+"lyrl2004_tokens_test_pt0.dat.gz")
    val a1 = loadIMat(dict+"lyrl2004_tokens_test_pt1.dat.gz")
    val a2 = loadIMat(dict+"lyrl2004_tokens_test_pt2.dat.gz")
    val a3 = loadIMat(dict+"lyrl2004_tokens_test_pt3.dat.gz")
    val a = (a0 on a1) on (a2 on a3)
    val (swc, ii) = sortdown2(wc)
    val sdict = dictm(ii)
    val bdict = SBMat(sdict)
    val n = ii.length
    val iinv = izeros(n, 1)
    iinv(ii) = icol(0->n)
    val jj = find(a > 0)
    a(jj,0) = iinv(a(jj,0)-1)
    saveIMat(dict+"tokens.imat.lz4", a)
    saveSBMat(dict+"../sdict.sbmat.lz4", bdict)
    saveIMat(dict+"../swcount.imat.lz4", swc)
  }
  
  def mksparse(dict:String) {
    val a = loadIMat(dict+"tokens.imat.lz4")
    val dictm = Dict(loadSBMat(dict+"../sdict.sbmat.lz4"))
    val swc = loadIMat(dict+"../swcount.imat.lz4")
    val tab = izeros(a.nrows,2)
    tab(?,1) = a
    val ii = find(a == dictm(".i"))
    val wi = find(a == dictm(".w"))
    tab(ii,1) = -1
    tab(wi,1) = -1
    val lkup = a(ii+1)
    Experiments.clearbit(lkup)
    tab(ii,0) = 1
    tab(0,0) = 0
    tab(?,0) = cumsum(tab(?,0))
    val iikeep = find(tab(?,1) >= 0)
    val ntab = tab(iikeep,?)
    val sm = sparse(ntab(?,1), ntab(?,0), ones(ntab.nrows,1), swc.length, ii.length)
    saveSMat(dict+"../docs.smat.lz4", sm)
    saveIMat(dict+"../lkup.imat.lz4", lkup)    
  }
  
  def mkcats(dict:String) {
    val lkup = loadIMat(dict+"../lkup.imat.lz4")
    val catids=loadIMat(dict+"../catname.imat")
    val docids=loadIMat(dict+"../docid.imat")
    val nd = math.max(maxi(lkup).v,maxi(docids).v)+1
    val nc = maxi(catids).v
    val cmat = izeros(nc,nd)
    val indx = catids - 1 + nc*docids
    cmat(indx) = 1
    val cm = FMat(cmat(?,lkup))
    saveFMat(dict+"../cats.fmat.lz4", cm) 
  }
}

object Twitter { 
  
   def dodicts(threshold:Int=10, rebuild:Boolean=false):Unit = {
  		 val stokdir = "/twitter/smiley/tokenized/"
    	 val tokdir = "/twitter/tokenized/"
  		 val dy1 = mergedicts(2011, 2013, "/disk%02d" + stokdir, "/big" + stokdir, threshold, rebuild)
  		 val dy2 = mergedicts(2011, 2013, "/disk%02d" + tokdir, "/big" + tokdir, threshold, rebuild)
  		 val dy = Dict.union(dy1, dy2)
  		 val (sv, iv) = sortdown2(dy.counts)
  		 HMat.saveSBMat("/big"+tokdir+"alldict.gz", SBMat(dy.cstr(iv)))
  		 HMat.saveDMat("/big"+tokdir+"allwcount.gz", sv)
	}
  
	def mergedicts(year1:Int, year2:Int, infname:String, outfname:String, threshold:Int=10, rebuild:Boolean=false):Dict = {
  	val dd = new Array[Dict](6)
  	val md = new Array[Dict](6)
  	val yd = new Array[Dict](5)
  	var dy:Dict = null
  	var nmerged = 0
	  for (yy <- year1 to year2) {
	  	for (mm <- 1 to 12) {
	  		print("\n%d/%02d" format (yy, mm))
	  		val ff = new File(outfname + "%04d/%02d/wcount.gz" format (yy, mm))
	  		if (rebuild || ! ff.exists) {
	  			var ndone = 0
	  			for (id <- 1 to 31) {
	  				var ielem = 372*yy + 31*mm + id
	  				var idisk = ielem % 16
	  				val fname = (infname + "%04d/%02d/%02d/" format (idisk, yy, mm, id))
	  				val ff = new File(fname + "wcount.gz")
	  				if (ff.exists) {
	  					val bb = HMat.loadSBMat(fname + "dict.gz")
	  					val cc = HMat.loadIMat(fname + "wcount.gz")
	  					dd(ndone % 6) = Dict(bb, cc, threshold)
	  					ndone = ndone + 1
	  					print("-")
	  					if (ndone % 6 == 0) {
	  						md(ndone / 6 - 1) = Dict.union(dd:_*)
	  						print("+")
	  					}
	  				}
	  			}
	  			if (ndone % 6 != 0) {
	  				md(ndone / 6) = Dict.union(dd.slice(0, ndone % 6):_*)
	  				print("+")
	  			}
	  			if (ndone > 0) {
	  				val dx = Dict.union(md.slice(0, (ndone-1)/6+1):_*)
	  				val (sv, iv) = sortdown2(dx.counts)
	  				val dxx = Dict(dx.cstr(iv), sv)
	  				HMat.saveSBMat(outfname + "%04d/%02d/dict.gz" format (yy, mm), SBMat(dxx.cstr))
	  				HMat.saveDMat(outfname + "%04d/%02d/wcount.gz" format (yy, mm), dxx.counts)
	  			}
//	  			println("")
	  		}
	  		val f2 = new File(outfname + "%04d/%02d/wcount.gz" format (yy, mm))
	  		if (f2.exists) {
	  			val bb = HMat.loadSBMat(outfname + "%04d/%02d/dict.gz" format (yy, mm))
	  			val cc = HMat.loadDMat(outfname + "%04d/%02d/wcount.gz" format (yy, mm))
	  			yd(nmerged % 5) = Dict(bb, cc, 4*threshold)
	  			nmerged += 1
	  			print("*")
	  			if (nmerged % 5 == 0) {
	  			  val dm = Dict.union(yd:_*)
	  			  if (nmerged == 5) {
	  			    dy = dm
	  			  } else {
	  			  	dy = Dict.union(dy, dm)
	  			  }
	  			}
	  		}
	  	}
	  }
  	if (nmerged % 5 != 0) {
  		val dm = Dict.union(yd.slice(0, nmerged % 5):_*)
  		dy = Dict.union(dy, dm)
  	}
  	println
  	val (sv, iv) = sortdown2(dy.counts)
  	val dyy = Dict(dy.cstr(iv), sv)
  	HMat.saveSBMat(outfname + "dict.gz", SBMat(dyy.cstr))
  	HMat.saveDMat(outfname + "wcount.gz", dyy.counts)
  	dyy
	}
	
	def getDict = {
  	val bd = loadSBMat("/big/twitter/tokenized/alldict.gz")
    val bc = loadDMat("/big/twitter/tokenized/allwcount.gz")
    Dict(bd, bc)
	}
	
  def getBiDict = {
  	val bd = loadIMat("/big/twitter/tokenized/allbdict.lz4")
    val bc = loadDMat("/big/twitter/tokenized/allbcnts.lz4")
    IDict(bd, bc)
	}
  
  def getTriDict = {
  	val bd = loadIMat("/big/twitter/tokenized/alltdict.lz4")
    val bc = loadDMat("/big/twitter/tokenized/alltcnts.lz4")
    IDict(bd, bc)
	}
  
  def junk:CSMat = {
    csrow("<id>", "</id>", "<status>", "</status>", "<user>", "</user>", "<created_at>",
        "</created_at>", "<screen_name>", "</screen_name>", "<lang>", "</lang>", "<statuses_count>", "</statuses_count>",
        "<followers_count>", "</followers_count>", "<friends_count>", "</friends_count>", "<favorites_count>", "</favorites_count>" +
        "<listed_count>", "</listed_count>", "<location>", "</location>", "<text>", "</text>", "<url>", "</url>", 
        "<in_reply_to_user_id>", "</in_reply_to_user_id>", "<in_reply_to_screen_name>", "</in_reply_to_screen_name>",
        "<in_reply_to_status_id>", "</in_reply_to_status_id>", "<retweet>", "</retweet>", "<retweet_count>", "</retweet_count>",
        "<type>", "</type>", "<name>", "</name>", "<full_name>", "</full_name>", "<country>", "</country>", "<place>", "</place>",
        "<country_code>", "</country_code>", "<bounding_box>", "</bounding_box>", "<coordinates>", "</coordinates>", 
        "http", "https", "apos", "kml", "amp", "www", "quot", "id", "latitude", "longitude", "latlonbox", "geo", "json")
  }
	
	def findEmoticons(n:Int, dd:Dict) = {
    val smiles = csrow(":-)", ":)", ":o)", ":]", ":3", ":c)", ":>", "=]", "8)", "=)", ":}", ":^)", ":っ)")
    val laughs = csrow(":-d", ":d", "8-d", "8d", "x-d", "xd", "x-x", "=-d", "=d", "=-3", "=3", "b^d")
    val frowns = csrow(">:[", ":-(", ":(", "", ":-c", ":c", ":-<", "", ":っc", ":<", ":-[", ":[", ":{")
    val angry = csrow(":-||", ":@", ">:(")
    val crying = csrow(":'-(", ":'(", "qq")
    val horror = csrow("d:<", "d:", "d8", "d;", "d=", "dx", "v.v", "d-':")
    val surprise = csrow(">:o", ":-o", ":o", "°o°", "°o°", ":o", "o_o", "o_0", "o.o", "8-0")
    val wink = csrow(";-)", ";)", "*-)", "*)", ";-]", ";]", ";d", ";^)", ":-,")
    val all = List(smiles, laughs, frowns, angry, crying, horror, surprise, wink, junk)
    val out = zeros(all.length, n)
    for (i <- 0 until all.length) {
      val mm = all(i)
      var j = 0
      while (j < mm.length) {
        val k = dd(mm(j))
        if (k >= 0 && k < n) out(i, k) = 1
        j += 1
      }      
    }
    out    
	}
	
	def getGramDict(nuni0:Int=50, nbi0:Int=100, ntri0:Int=200, rebuild:Boolean=false):Dict = {
	  val nuni = nuni0 * 1000
	  val nbi = nbi0 * 1000
	  val ntri = ntri0 * 1000
	  val fname = "/big/twitter/tokenized/dict_%d_%d_%d" format (nuni0, nbi0, ntri0)
	  if (!rebuild && (new File(fname + "_SBMat.lz4").exists) && (new File(fname + "_dmat.lz4").exists)) {
	    val bm = loadSBMat(fname + "_SBMat.lz4")
	    val dm = loadDMat(fname + "_dmat.lz4")
	    Dict(bm, dm)
	  } else {
	  	val ud = getDict
	  	val bd = getBiDict
	  	val td = getTriDict
	  	val dd = IDict.gramDict(nuni, nbi, ntri, ud, bd, td)
	  	saveSBMat(fname + "_SBMat.lz4", SBMat(dd.cstr))
	  	saveDMat(fname + "_dmat.lz4", dd.counts)
	  	dd
	  }
	}
	
	def getEmoticonMap(nuni0:Int=50, nbi0:Int=100, ntri0:Int=200, rebuild:Boolean=false):FMat = {
	   val nuni = nuni0 * 1000
	   val nbi = nbi0 * 1000
	   val ntri = ntri0 * 1000
  	 val fname = "/big/twitter/tokenized/dict_%d_%d_%d" format (nuni0, nbi0, ntri0)
  	 if (!rebuild && (new File(fname + "_emos.lz4").exists)) {
  	   loadFMat(fname + "_emos.lz4")
  	 } else {
  		 val ud = getDict
  		 val bdt = getBiDict.grams(0->nbi,?)
  		 val tdt = getTriDict.grams(0->ntri,?)
  		 val em = findEmoticons(1 + maxi(irow(nuni) \ maxi(bdt) \ maxi(tdt)).v, ud)
  		 val bv = zeros(em.nrows, nbi)
  		 val tv = zeros(em.nrows, ntri)
  		 for (i <- 0 until em.nrows) {
  			 bv(i, ?) = max(em(i, bdt(?, 0)), em(i, bdt(?, 1)))
  			 tv(i, ?) = max(em(i, tdt(?, 0)), max(em(i, tdt(?, 1)), em(i, tdt(?, 2))))
  		 }
  		 val emos = em(?, 0->nuni) \ bv(?, 0->nbi) \ tv(?, 0->ntri)
  		 saveFMat(fname + "_emos.lz4", emos)
  		 emos
  	 }
	}
	
	def logisticModelPar(
	    nstart0:Int = FilesDS.encodeDate(2012,3,1,0),
			nend0:Int = FilesDS.encodeDate(2013,7,1,0),
			nuni0:Int = 50,
			nbi0:Int = 100,
			ntri0:Int = 200		
			) = {
	  val ds = SFilesDS.twitterNgramBlend(nstart0, nend0)
//	  val ds = SFilesDataSource.twitterWords(nstart0, nend0)
	  ds.opts.addConstFeat = true
	  ds.opts.featType = 0
	  val gd = getGramDict(nuni0, nbi0, ntri0)
	  val em = getEmoticonMap(nuni0, nbi0, ntri0)
	  val nfeats = gd.length + 1
	  val mask = (sum(em) == 0f) \ 1
//	  val targets = em(0->(em.nrows-1), ?) \ zeros(em.nrows-1,1)
	  val targets = em(0->1, ?) \ 0
	  val ntargets = targets.nrows
	  val exptsv = col(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
	  val exptst = col(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
//	  val expts = col(0.5)
	  val avalues = col(0.1f, 1f, 10f)
	  val expts1 = ones(avalues.length*ntargets, 1) ⊗ exptsv ⊗ ones(exptst.length, 1)
	  val expts2 = ones(avalues.length*exptsv.length*ntargets, 1) ⊗ exptst 
	  val alphas = ones(ntargets, 1) ⊗ avalues ⊗ ones(exptst.length*exptsv.length, 1)
	  val aopts = new ADAGrad.Options
	  aopts.vexp = expts1
	  aopts.texp = expts2
	  aopts.alpha = alphas
	  aopts.mask = mask
	  val gopts = new GLM.Options
	  gopts.links = iones(expts1.length, 1)
	  gopts.rmask = mask
	  gopts.targmap = mkdiag(ones(ntargets, 1)) ⊗ ones(expts1.length/ntargets, 1)
	  gopts.targets = targets
  	new ParLearnerF(ds, gopts, GLM.mkGLMModel _, null, null, aopts, GLM.mkUpdater _)	  
	}
	
	def logisticModel(
	    mat:SMat,
	    ntargs:Int = 1,
	    exptsv:FMat = col(0.4, 0.5, 0.6),
	    exptst:FMat = col(0.4, 0.5, 0.6),
	    avalues:FMat = col(0.1, 0.3, 1),
			nuni0:Int = 50,
			nbi0:Int = 100,
			ntri0:Int = 200		
			) = { 
	  val ds = new MatDS(Array(mat:Mat))
	  val gd = getGramDict(nuni0, nbi0, ntri0)
	  val em = getEmoticonMap(nuni0, nbi0, ntri0)
	  val nfeats = gd.length + 1
	  val mask = (sum(em) == 0f) \ 1
	  val targets0 = em(0->(em.nrows-1), ?) \ zeros(em.nrows-1,1)
	  val targets = targets0(0->ntargs, ?)
	  val ntargets = targets.nrows
	  val expts1 = ones(avalues.length*ntargets, 1) ⊗ exptsv ⊗ ones(exptst.length, 1)
	  val expts2 = ones(avalues.length*exptsv.length*ntargets, 1) ⊗ exptst 
	  val alphas = ones(ntargets, 1) ⊗ avalues ⊗ ones(exptst.length*exptsv.length, 1)
	  val aopts = new ADAGrad.Options
	  aopts.vexp = expts1
	  aopts.texp = expts2
	  aopts.alpha = alphas
	  aopts.mask = mask
	  val gopts = new GLM.Options
	  gopts.links = iones(expts1.length, 1)
	  gopts.rmask = mask
	  gopts.targmap = mkdiag(ones(ntargets, 1)) ⊗ ones(expts1.length/ntargets, 1)
	  gopts.targets = targets
  	Learner(ds, new GLM(gopts), null, new ADAGrad(aopts))	  
	}
	

}