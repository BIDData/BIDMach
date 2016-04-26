package BIDMach.tools

import BIDMat.{Mat,SBMat,CMat,CSMat,Dict,DMat,FMat,IDict,IMat,HMat,GMat,GIMat,GSMat,SMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import java.io.File

/**
  * @author bjaros
  * 
  * Prepare data for consumption by SeqToSeq LSTMs. 
  * 
  * The SeqToSeqData class processes a pair of parsed sentences, each in IMat format, and produces SMat
  * outputs of length-collated sentences for SeqToSeq training. 
  *
  * Parsed sentences are IMat of format (e.g. the result of running nnparse.exe):
  *         p1 s1 w1
  *         p2 s2 w2
  *         p3 s3 w3
  *         p4 s4 w4
  *         p5 s5 w5
  *         p6 s6 w6
  *   e.g.
  *          0  0 96
  *          0  0 17
  *          0  0 23
  *          1  0  7
  *          1  0 31
  *          2  0 86      
  *   would be first sentence with id's "<96> <17> <23>" and second sentence with
  *   id's "7 31" and third sentence with id's "<86>".
  *   
  * (For SeqToSeq we assume each line contains one sentence, so the paragraphid (the first column)
  * denotes the sentence and sentenceid (the second column) is always ignored).
  * 
  * The two parsed sentence IMats are paired line-by-line:  the ith line of the src IMat corresponds
  * to the ith line of the dst IMat.
  *
  * The output is two SMat's of the following form:
  * 
  *          w00  w01  w02  w03  w04  w05  ...
  *          w10  w11  w12  w13  w14  w15P ...
  *          w20  w21  w22  w23P w24  w25P ...
  *          w30  w31P w32                 ...
  *          w40P w32P w33                 ...
  * 
  * where
  *    wij is the dictionary index of the i'th word in the j'th sentence and 
  *    words with a P suffix are padding symbols. 
  *
  * The columns of the two output SMat's are still paired:  column j of the src output SMat and
  * column j of the dst output SMat correspond to line j of the src input and line j of the dst input
  * respectively.
  *
  * Furthermore, the sentences are collated into batches of similar lengths.
  * 
  * The minibatches are randomly permuted after collation to avoid training bias.
  * 
  * Use opts.srcvocabmincount & dstvocabmincount to trim the dictionary to a minimum count (and update
  * the output matrices correspondingly).
  * Use opts.srcvocabmaxsize & dstvocabmaxsize to trim the dictionary to a maximum size (and update
  * the output matricescorrespondingly).
  *
  * The dictionaries corresponding to the output matrices are saved to the outputdir as well.
  *         
  * LIMITATIONS:
  *
  *   Since float values are used to hold word ids, the maximum dictionary size is 16M. Use SDMat if this is a problem. 
 */

object SeqToSeqDict {
  val specialsyms = cscol("<zero>","<pad>","<oov>","<eos>");
  val padsym = 1;                                                // Index of special padding symbol
  val eossym = 2;                                                // Index of special end-of-sentence symbol
  val oovsym = 3;                                                // Index of out-of-vocabulary symbol
  val numspecialsyms = specialsyms.length;
  
  def apply(csmat:CSMat):Dict = {
    val cs0 = if (csmat(padsym)==specialsyms(padsym)) {  // Already SeqToSeqDict?
      csmat;
    } else {                                             // Add specialsyms
      specialsyms on csmat;
    }
    val out = new Dict(cs0);
    return out;
  }

  def apply(sbmatpath:String):Dict = {
    val cs = CSMat(loadSBMat(sbmatpath));
    return SeqToSeqDict(cs);
  }

  def apply(fpaths:(String,String)):Dict = {
    /*
     * Load dict with counts.
     * Input is tuple of path to dict (an sbmat) and path to dict counts (an imat).
     */
    val sbmatfpath = fpaths._1;
    val imatfpath = fpaths._2;    
    val out = SeqToSeqDict(sbmatfpath);
    if (imatfpath != null) {
      val dictcnt = loadIMat(imatfpath);
      out.counts = if (dictcnt(0)==Int.MaxValue) {     // Already SeqToSeqDict?
        DMat(dictcnt);
      } else {
        Int.MaxValue*ones(numspecialsyms,1) on DMat(dictcnt); // TODO correct MaxValue?
      }
    }
    return out;
  }

  def top(dict:Dict, maxsize:Int):Dict = {
    /*
     * Return a dict with the top #maxsize tokens by count
     */
    val (ss, ii) = sortrows(dict.counts(numspecialsyms->dict.length).t);         // Sort the dict counts of non-special symbols
    val ii2 = ii((ii.length-1) to (ii.length-maxsize) by -1);                    // Take the top maxsize counts.  Reverse order for convenience when inspecting.
    ii2 ~ ii2 + numspecialsyms;                                                  // Offset those indices to again count for special symbols.
    val cstr = specialsyms on dict.cstr(ii2.t);                                  // Recreate the cstr
    val cnt = Int.MaxValue*ones(numspecialsyms,1) on dict.counts(ii2.t);      // Recreate the counts
    Dict(cstr, cnt)
  }
}

object printmat {
  /*
   * Utility to print out the words corresponding to given indices in an IMat or SMat.
   * Each column is assumed to be a sentence.
   */
  def apply(mat:IMat, dict:Dict):Unit = {
    /*
     * Translate the indices in IMat into words and print out sentences (each column is a sentence).
     */
    apply(sparse(mat),dict);
  }

  def apply(mat:SMat, dict:Dict, maxcols:Int=100):Unit = {
    /*
     * Translate the indices in integer SMat into words and print out sentences (each column is a sentence).
     */
    for (j <- 0 until mat.ncols) {
      var i = 0;
      var rowdone = false;
      print("[%d] " format j);
      while ((i < mat.nrows) & !rowdone) {
        val indx:Int = mat(i,j).toInt;
        if (indx==0) {
          rowdone = true;
        } else {
          val token:String = dict(indx); 
          print("%s (%d) " format (token,indx));
        }
        i += 1;
      }
      println("")
      if (j==maxcols-1) {
        println("[Truncated printing at %d cols.  Adjust using maxcols argument to printmat]" format maxcols);
        return;
      }
    }
  }
}

class SeqToSeqData(val opts:SeqToSeqData.Opts = new SeqToSeqData.Options) {    
  var dictinitialized:Boolean=false; 
  var srcdict:Dict=null;
  var dstdict:Dict=null;
  var srcindexmapping:IMat=null;
  var dstindexmapping:IMat=null;
  
  def getStartsAndLens(parsedSents:IMat):(IMat,IMat) = {
    /*
     * Given parsedSents IMat of format described above:
     *   e.g.
     *          0  0 96
     *          0  0 17
     *          0  0 23
     *          1  0  7
     *          1  0 31
     *          2  0 86      
     *   
     *   Return starts, a [nsents x 1] matrix:
     *          0
     *          3
     *          5
     *          
     *   and lens, also a [nsents x 1] matrix
     *          3
     *          2
     *          1
     *          
     *   Also filters out sentences shorter than opts.minlen.
     */
    val starts0 = 0 on 1+find(parsedSents(0->(parsedSents.nrows-1),opts.sentcol) !=
                              parsedSents((1->parsedSents.nrows),opts.sentcol)); // Starting indices.  [nbatches x 1]
    val posts0 = starts0 on parsedSents.nrows;
    val lens0 = posts0(1->posts0.nrows) - posts0(0->(posts0.nrows-1));
    // We would be done here, except that there could be zero-length sentences which would have
    // been skipped altogether.  We want to make sure sentence #i is in the ith slot of starts & lens.
    val senti = parsedSents(starts0,opts.sentcol);
    val numsents = 1+maxi(parsedSents(?,opts.sentcol))(0);
    val starts = izeros(numsents,1); 
    starts(senti) = starts0;
    val lens = izeros(1,numsents);
    lens(0,senti) = lens0;
    return (starts,lens)

//    val starts = 0 on 1+find(parsedSents(0->(parsedSents.nrows-1),opts.sentcol) !=
//                             parsedSents((1->parsedSents.nrows),opts.sentcol)); // Starting indices.  [nbatches x 1]
//    val posts = starts on parsedSents.nrows;
//    val lens = posts(1->posts.nrows) - posts(0->(posts.nrows-1));
//    return (starts,lens)
  }

  def getoutputdir(inputdir:String, outputdir0:String=null):String = {
    /*
     * Return the outputdir (or default to inputdir+"/output" if none provided), mkdir'ing
     * if necessary.
     */
    val outputdir = if (outputdir0==null) {
      val suffix = "%s%s%s%s_sl%d-%d_dl%d-%d_b%d%s%s" format (
          if (opts.srcvocabmaxsize > 0) "_s%d" format opts.srcvocabmaxsize else "",
          if (opts.srcvocabmincount > 1) "_smin%d" format opts.srcvocabmincount else "",
          if (opts.dstvocabmaxsize > 0) "_d%d" format opts.dstvocabmaxsize else "",
          if (opts.dstvocabmincount > 1) "_dmin%d" format opts.dstvocabmincount else "",
          opts.srcminlen, opts.srcmaxlen, opts.dstminlen, opts.dstmaxlen, opts.bsize,
          if (opts.revsrc) "_revsrc" else "", if (opts.revdst) "_revdst" else "");
      inputdir+"/../seq2seq%s" format suffix
    } else {
      outputdir0;
    }
    // Make outputdir
    val outputdirFile = new File(outputdir);
    if (!outputdirFile.exists()) {
      val successful = outputdirFile.mkdirs();
      assert(successful, "Failed to make outputdir at %s" format outputdir);
    }
    return outputdir;
  }  


  def loadDict(origdictfpaths:(String,String), outputdir:String=null, vocabmaxsize:Int=0, vocabmincount:Int=0):(Dict,IMat) = {
    /* 
     * Given a tuple of paths to dict sbmat and (optionally) imat of counts,
     * 1) load the dictionary
     * 2) optionally trim the dictionary based on vocabmaxsize
     * 3) if outputdir provided, save the dictionary to outputdir
     * 4) return the dictionary and optionally (based on (2)), the mapping from the original indices
     *    to the trimmed indices  
     */
    if (origdictfpaths._1 == null) {
      assert(vocabmaxsize <= 0, "If vocabmaxsize > 0, you must provide a dict and counts");
      assert(vocabmincount <= 1, "If vocabmincount > 1, you must provide a dict and counts");
      return (null,null);
    } else {
      var origDict = SeqToSeqDict(origdictfpaths);
      var needmap = false;
      var dict:Dict = origDict;
      if (vocabmincount > 1) {
        needmap = true;
        dict = origDict.trim(vocabmincount);
      }
      if ((vocabmaxsize > 0) && (vocabmaxsize < origDict.length)) {
        needmap = true;
        dict = SeqToSeqDict.top(dict, vocabmaxsize);
      }
      val indexmapping = if (needmap) {
        origDict --> dict;
      } else {
        null;
      }
      // Save dict to outputdir (with same filenames)
      if (outputdir!=null) {
        val sbmatfname = (new File(origdictfpaths._1)).getName;
        val imatfname = (new File(origdictfpaths._2)).getName;
        val sbmatfpath = outputdir+"/"+sbmatfname;
        val imatfpath = outputdir+"/"+imatfname;
        println("Saving dicts to %s and %s" format (sbmatfpath,imatfpath));
        saveSBMat(sbmatfpath, SBMat(dict.cstr));
        saveIMat(imatfpath, IMat(dict.counts));
      }
      return (dict,indexmapping)
    }
  }

  def loadData(fpath:String, indexmapping:IMat=null):IMat = {
    /*
     * 1) Load the (imat) data from fpath
     * 2) If indexmapping provided, map the indices and fill in oovsym
     * 3) Offset to make room for special characters
     */
    val parsedSents = loadIMat(fpath);
    parsedSents(?,opts.wordcol) += SeqToSeqDict.numspecialsyms - 1;
    if (indexmapping!=null) {  // trimming
      parsedSents(?,opts.wordcol) = indexmapping(parsedSents(?,opts.wordcol));
      val ii = find(parsedSents(?,opts.wordcol)<0);
      parsedSents(ii,opts.wordcol) = SeqToSeqDict.oovsym;
    }
    return parsedSents;
  }

  
  /*
   * prepSeqToSeqDataWildcard
   * 
   *   fnamepatterns: a tuple of 2 strings (1 for src pattern, 1 for dstpattern) with single asterisk
   *                  denoting wildcard.
   *                
   *   Example:
   *       prepSeqToSeqDataWildcard("/path/to/indir", ("data-*.src","data-*.dst"), "/path/to/indir");
   *
   *           will call prepSeqToSeqData() on "/path/to/indir/data-01.src", "/path/to/indir/data-01.dst"
   *                                           "/path/to/indir/data-02.src", "/path/to/indir/data-02.dst"
   *                                           "/path/to/indir/data-03.src", "/path/to/indir/data-03.dst"
   *                                           ...
   *   
   */  
  def prepSeqToSeqDataWildcard(inputdir:String, fnamepatterns:(String,String)):Unit = {
    prepSeqToSeqDataWildcard(inputdir, fnamepatterns, ((null,null),(null,null)), null);
  }

  def prepSeqToSeqDataWildcard(inputdir:String, fnamepatterns:(String,String),
                               outputdir0:String):Unit = {
    prepSeqToSeqDataWildcard(inputdir, fnamepatterns, ((null,null),(null,null)), outputdir0);
  }

  def prepSeqToSeqDataWildcard(inputdir:String, fnamepatterns:(String,String),
                               origdictfnames:((String,String),(String,String))):Unit = {
    prepSeqToSeqDataWildcard(inputdir, fnamepatterns, origdictfnames, ((null,null),(null,null)), null);
  }  
  
  def prepSeqToSeqDataWildcard(inputdir:String, fnamepatterns:(String,String),
                               origdictfnames:((String,String),(String,String)),
                               outputdir0:String):Unit = {
    prepSeqToSeqDataWildcard(inputdir, fnamepatterns, origdictfnames, ((null,null),(null,null)), outputdir0);
  }
  
  def prepSeqToSeqDataWildcard(inputdir:String, fnamepatterns:(String,String),
                               origdictfnames:((String,String),(String,String))=((null,null),(null,null)),
                               targetdictfnames:((String,String),(String,String))=((null,null),(null,null)),
                               outputdir0:String=null):Unit = {    
    val files = new File(inputdir).listFiles;
    assert(files!=null,"No directory %s" format inputdir)
    val allfnames = files.map(_.getName).sorted;
    
    val srcpattern = fnamepatterns._1;
    val srcparts = srcpattern.split("\\*");
    val srcfiles0 = allfnames.filter(_.startsWith(srcparts(0)));
    val srcfiles = if (srcparts.length==2) srcfiles0.filter(_.endsWith(srcparts(1))) else srcfiles0;

    val dstpattern = fnamepatterns._2;
    val dstparts = dstpattern.split("\\*");
    val dstfiles0 = allfnames.filter(_.startsWith(dstparts(0)));
    val dstfiles = if (dstparts.length==2) dstfiles0.filter(_.endsWith(dstparts(1))) else dstfiles0;

    val outputdir = getoutputdir(inputdir,outputdir0);
    
    assert(srcfiles.length==dstfiles.length);
    for (ifile <- 0 until srcfiles.length) {
      //    try {
        val srcfname = srcfiles(ifile);
        val dstfname = dstfiles(ifile);
        println("Processing %s <--> %s" format (srcfname,dstfname));
        val (srcmat,dstmat) = prepSeqToSeqData(inputdir, (srcfname, dstfname), 
                                               origdictfnames, targetdictfnames, outputdir);
        // Save
        saveSMat(outputdir + "/src%04d.smat.lz4" format ifile, srcmat);
        saveSMat(outputdir + "/dst%04d.smat.lz4" format ifile, dstmat);
      //    }
      //    catch {
      //    case _: Exception => {println("problem with file %d" format ifile)}
      //    case _: Throwable => {println("problem with file %d" format ifile)}
      //    }
    }
  }

  /*
   * prepSeqToSeqData
   * 
   *   origdictfnames:    Filename of the dicts used to create the parsed data
   *                      There are two reasons to provide this:
   *                         1) To prune the vocabulary (opts.srcvocabmaxsize and opts.dstvocabmaxsize)
   *                         2) To map to another dict, targetorigdictfnames
   *   targetdictfnames:  Provide if you want to match the indices of another dictionary
   *   
   */  
  def prepSeqToSeqData(inputdir:String, fnames:(String,String)):(SMat,SMat) = {
    prepSeqToSeqData(inputdir, fnames, ((null,null),(null,null)), null)
  }  
  
  def prepSeqToSeqData(inputdir:String, fnames:(String,String), outputdir0:String):(SMat,SMat) = {
    prepSeqToSeqData(inputdir, fnames, ((null,null),(null,null)), outputdir0)
  }

  def prepSeqToSeqData(inputdir:String, fnames:(String,String), origdictfnames:((String,String),(String,String))):(SMat,SMat) = {
    prepSeqToSeqData(inputdir, fnames, origdictfnames, ((null,null),(null,null)), null)
  }
    
  def prepSeqToSeqData(inputdir:String, fnames:(String,String), origdictfnames:((String,String),(String,String)),
                       outputdir0:String):(SMat,SMat) = {
    prepSeqToSeqData(inputdir, fnames, origdictfnames, ((null,null),(null,null)), outputdir0)
  }
  
  def prepSeqToSeqData(inputdir:String, fnames:(String,String), origdictfnames:((String,String),(String,String)),
                       targetdictfnames:((String,String),(String,String)),
                       outputdir0:String=null):(SMat,SMat) = {    
    val srcpath:String = inputdir+"/"+fnames._1;
    val dstpath:String = inputdir+"/"+fnames._2;
    val origsrcdictfpath:String = if (origdictfnames._1._1==null) null else inputdir+"/"+origdictfnames._1._1;
    val origsrcdictcntfpath:String = if (origdictfnames._1._2==null) null else inputdir+"/"+origdictfnames._1._2;
    val origdstdictfpath:String = if (origdictfnames._2._1==null) null else inputdir+"/"+origdictfnames._2._1;
    val origdstdictcntfpath:String = if (origdictfnames._2._2==null) null else inputdir+"/"+origdictfnames._2._2;
    val targetsrcdictfpath:String = if (targetdictfnames._1._1==null) null else inputdir+"/"+targetdictfnames._1._1;
    val targetsrcdictcntfpath:String = if (targetdictfnames._1._2==null) null else inputdir+"/"+targetdictfnames._1._2;
    val targetdstdictfpath:String = if (targetdictfnames._2._1==null) null else inputdir+"/"+targetdictfnames._2._1;
    val targetdstdictcntfpath:String = if (targetdictfnames._2._2==null) null else inputdir+"/"+targetdictfnames._2._2;
    val outputdir = getoutputdir(inputdir,outputdir0);
    prepSeqToSeqData((srcpath,dstpath),
                     ((origsrcdictfpath,origsrcdictcntfpath),(origdstdictfpath,origdstdictcntfpath)),
                     ((targetsrcdictfpath,targetsrcdictcntfpath),(targetdstdictfpath,targetdstdictcntfpath)),
                     outputdir);
  }
  
  def prepSeqToSeqData(fpaths:(String,String), origdictfpaths:((String,String),(String,String)),
                       outputdir:String):(SMat,SMat) = {   
    prepSeqToSeqData(fpaths, origdictfpaths,((null,null),(null,null)),outputdir);
  }

  def prepSeqToSeqData(fpaths:(String,String), origdictfpaths:((String,String),(String,String)),
                       targetdictfpaths:((String,String),(String,String))):(SMat,SMat) = {    
    prepSeqToSeqData(fpaths, origdictfpaths,targetdictfpaths,null); 
  }
  
  def prepSeqToSeqData(fpaths:(String,String), origdictfpaths:((String,String),(String,String)),
                       targetdictfpaths:((String,String),(String,String)),
                       outputdir:String):(SMat,SMat) = {    
    // Make outputdir if necessary
    val outputdirFile = new File(outputdir);
    if (!outputdirFile.exists()) {
      val successful = outputdirFile.mkdirs();
      assert(successful, "Failed to make outputdir at %s" format outputdir);
    }

    // Prepare dicts and indexmappings
    if (!dictinitialized) {
      if (targetdictfpaths._1._1 != null) {
        if ((opts.srcvocabmaxsize > 0) || (opts.dstvocabmaxsize > 0))
          print(s"Warning: opts.srcvocabmaxsize & opts.dstvocabmaxsize will be ignored (using whatever used in ${targetdictfpaths._1._1} and ${targetdictfpaths._2._1})");
        val srcdict = loadDict(origdictfpaths._1)._1;  // Don't save these
        val dstdict = loadDict(origdictfpaths._2)._1;  // Don't save these
        val targetsrcdict = loadDict(targetdictfpaths._1, outputdir)._1;
        val targetdstdict = loadDict(targetdictfpaths._2, outputdir)._1;
        srcindexmapping = srcdict --> targetsrcdict;
        dstindexmapping = dstdict --> targetdstdict;
        dictinitialized=true;      
      } else {
        val res1 = loadDict(origdictfpaths._1, outputdir, opts.srcvocabmaxsize, opts.srcvocabmincount);
        srcdict = res1._1;  srcindexmapping = res1._2;
        val res2 = loadDict(origdictfpaths._2, outputdir, opts.dstvocabmaxsize, opts.dstvocabmincount);
        dstdict = res2._1;  dstindexmapping = res2._2;
        dictinitialized=true;
      }
    }
    val srcsents:IMat = loadData(fpaths._1,srcindexmapping);
    val dstsents:IMat = loadData(fpaths._2,dstindexmapping);
    
    val (srcstartsAll,srclensAll) = getStartsAndLens(srcsents);
    val (dststartsAll,dstlensAll) = getStartsAndLens(dstsents);

    // Filter sentences where the src or dst isn't long enough
    val numsents = math.min(srclensAll.length, dstlensAll.length); // In case src/dst finished with empty (length-0) sentences
    val iigmonotonic = find((srclensAll >= opts.srcminlen)(0->numsents) *@
                            (dstlensAll >= opts.dstminlen)(0->numsents))   // Check min length threshold.  [nsents x 1]
    val iig = if (opts.maintainordering) {
      iigmonotonic;
    } else {
      iigmonotonic(randperm(iigmonotonic.length).t);         // Randomize.  Otherwise sorting would be monotonic over sentences with ties in lengths.  [nsents x 1]
    }
    
    // Make starts and lens mats
    val srcstarts = srcstartsAll(iig);
    val srclens = srclensAll(iig);
    srclens(find(srclens>opts.srcmaxlen)) = opts.srcmaxlen;
    val dststarts = dststartsAll(iig);
    val dstlens = dstlensAll(iig);
    dstlens(find(dstlens>opts.dstmaxlen)) = opts.dstmaxlen;
    
    // Sort by lengths (unless opts.maintainordering)
    val (ss, ii) = if (opts.maintainordering) {
      (null, icol(0->srclens.length))                        // Original order
    } else {
      val sortbylens = maxi(dstlens) * srclens + dstlens;    // primary sort by srclens; secondary sort by dstlens 
      sortrows(sortbylens);                                  // lex sort the length pairs and get permutation indices in ii.  ii: [sents x 1]
    }
    
    val nsents = srcstarts.size;
    val nbatches = nsents / opts.bsize;
    val nsents2 = opts.bsize * (nsents / opts.bsize);        // Round length to a multiple of batch size
    val ii2 = ii((nsents - nsents2)->nsents);                // Drop the shortest sentences, giving a multiple of batch size.  [1 x nsents2]
    val i2dsorted = ii2.view(opts.bsize, nbatches).t;        // Put inds in a 2d matrix, with columns which are minibatches.  [nbatches x bsize]

    if (opts.maintainordering) {
      assert(nsents2 == nsents, "%d != %d.  If opts.maintainordering, you want the bsize to divide evenly into nsents (%d)" format (nsents,nsents2,nsents))
    }
        
    val i2d = if (opts.maintainordering) {
      i2dsorted      
    } else {
      val ip = randperm(nbatches);
      i2dsorted(ip,?);                                       // Randomly permute the minibatches.  *Sentence indices divided into batches.* [nbatches x bsize]
    }
    
    val srcmat = mkmat(srcsents, srcstarts, srclens, i2d, opts.revsrc, true);
    val dstmat = mkmat(dstsents, dststarts, dstlens, i2d, opts.revdst, false);
    return (srcmat,dstmat);    
  }
  
  def mkmat(parsedSents:IMat, starts:IMat, lens:IMat, i2d:IMat, rev:Boolean=false, rightjustify:Boolean=false):SMat = {    
    /*
     * parsedSents:  in format described above.
     * starts:       the starting index (in parsedSents) of each sentence.  [nsents x 1]
     * lens:         the length of each sentence in parsedSents.  [nsents x 1]
     * i2d:          the sentence index for each location in the batch, [nbatches x bsize]
     * rev:          whether to reverse the order of the word indices in the sentence
     * rightjustify: whether to have all sentences end on the last column of the batch matrix
     *               (default of false, i.e. leftjustify, has all sentences start on the first col of the batch matrix)
     * 
     *    5 6 7 8, with n=6, padsym=1 -->
     *        rev=0 rightjustify=0      5 6 7 8 1 1 
     *        rev=0 rightjustify=1      1 1 5 6 7 8
     *        rev=1 rightjustify=0      8 7 6 5 1 1
     *        rev=1 rightjustify=1      1 1 8 7 6 5
     */
    val starts2d = starts(i2d);                              // Sentence start indices arranged by minibatch.  [nbatches x bsize]
    val lens2d = lens(i2d);                                  // Sentence lengths arranged by minibatch.  [nbatches x bsize]
    val maxlen = maxi(lens2d,2);                             // Max length in each minibatch - others get padded to this.  [nsents2 x 1]
    val nnz = sum(maxlen).v*opts.bsize;                      // Number of non-zeros in matrix.  

    val nsents = starts.size;
    val nbatches = nsents / opts.bsize;
    val nsents2 = opts.bsize * (nsents / opts.bsize);        // Round length to a multiple of batch size    
    
    // Prepare output
    val i = izeros(nnz, 1);                                  // row, col, val matrices for the final SMat
    val j = izeros(nnz, 1);
    val v = zeros(nnz, 1);
    var p = 0;
  
    var ibatch = 0;
    var longest = 0;
    while (ibatch < nbatches) {
      val n = maxlen(ibatch);                               // max length for this minibatch
      if (n>longest) longest=n;
      
      val blk = izeros(n, opts.bsize);
      val thisstarts = starts2d(ibatch,?);                  // Start index for each sentence of batch. [1 x bsize]
      val thislens = lens2d(ibatch,?);                      // Length for each sentence of batch.  [1 x bsize]
      if (rightjustify ^ rev) {                             // rightjustify + rev is the outcome of reversing leftjustify
        val thisends = thisstarts + thislens - 1;
        var posi = 0;
        while (posi < n) {                                  // Step through each position in sentence
          val validi = (thislens > posi);                   // (only for sentences where position < its length). [1 x bsize]
          val ii = (thisends - posi) *@ validi;             // Indices in parsedSents
          val vals = (parsedSents(ii,opts.wordcol).t - SeqToSeqDict.padsym) *@ validi;  // Values from parsed sents (offset by padsym, so we can add back in next step). [1 x bsize]
          blk(n-1-posi,?) = vals + SeqToSeqDict.padsym;
          posi += 1;
        }
      } else {
        var posi = 0;
        while (posi < n) {                                  // Step through each position in sentence
          val validi = (thislens > posi);                   // (only for sentences where position < its length). [1 x bsize]
          val ii = (thisstarts + posi) *@ validi;           // Indices in parsedSents
          val vals = (parsedSents(ii,opts.wordcol).t - SeqToSeqDict.padsym) *@ validi;  // Values from parsed sents (offset by padsym, so we can add back in next step). [1 x bsize]
          blk(posi,?) = vals + SeqToSeqDict.padsym;
          posi += 1;
        }
      }
      if (rev) {                                            // Reverse the sentences
        val revinds = icol((n-1) to 0 by -1);
        blk <-- blk(revinds,?);
      }
      
      val (ii, jj, vv) = find3(blk);                       // back to sparse indices.  [nnz x 1] == [n*bsize x 1]
      val ilen = ii.length;
      i(p->(p+ilen),0) = ii;                               // Add the src data to the global buffers
      j(p->(p+ilen),0) = jj + ibatch*opts.bsize;           // Offset the column indices appropriately
      v(p->(p+ilen),0) = vv;
      p += ilen;

      ibatch += 1;
    }
    val mat = sparse(i, j, v, longest, nsents2);
    return mat;
  }
}

object SeqToSeqData {
  trait Opts {
    var srcvocabmaxsize = -1;                                      // Max vocabulary size.  If <= 0, no maxsize performed.  If >= 1, must provide dict name.
    var srcvocabmincount = 0;                                      // Vocabulary minimum counts.  If <= 1, no trimming performed.  If > 1, must provide dict name.
    var srcmaxlen = 40;                                            // Maximum sentence length, truncate longer sentences
    var srcminlen = 1;                                             // Minimum sentence length, discard shorter sentences
    var dstvocabmaxsize = -1;                                      // Max vocabulary size.  If <= 0, no maxsize performed.  If >= 1, must provide dict name.
    var dstvocabmincount = 0;                                      // Vocabulary minimum counts.  If <= 1, no trimming performed.  If > 1, must provide dict name.
    var dstmaxlen = 40;                                            // Maximum sentence length, truncate longer sentences
    var dstminlen = 1;                                             // Minimum sentence length, discard shorter sentences
    var revsrc = true;                                             // Reverse the src sentences
    var revdst = false;                                            // Reverse the dst sentences
    var maintainordering = false;                                  // Output matrix columns are in same order as input
    var bsize = 128;                                               // Batch size
    var sentcol = 0;                                               // Column of input parsed data containing sentence ids
    var wordcol = 2;                                               // Column of input parsed data containing word ids
  }

  class Options extends Opts {}  
}