package BayesNet

import BIDMat.{CMat,CSMat,DMat,Dict,IDict,FMat,GMat,GIMat,GSMat,HMat,IMat,Mat,SMat,SBMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._

class Graph(val dag: SMat, val n: Int) {
 
 //val sdag: SMat = SMat(dag)
 var mrf: FMat = null
 var colors: IMat = null
 var ncolors = 0
   
 val maxColor = 100
  
 def connectParents(moral: FMat, parents: IMat) = {
   
   val l = parents.length
   
   for( i <- 0 until l)
     for( j <- 0 until l){
       if(parents(i) != parents(j)){
         moral(parents(i), parents(j)) = 1f
         //moral = moral + sparse(parents(i), parents(j), 1)
       }
     }
   moral

 } 
 
 def moralize = {
   
   //var moral = dag.copy
   var moral = full(dag)
 
   for(i <- 0 until n){
     var parents = find(dag(?, i))
     moral = connectParents(moral, parents)
   }
   mrf = ((moral + moral.t) > 0)
 }
 
 def color = {
   
   moralize

   var colorCount = izeros(maxColor, 1)
   colors = -1 * iones(n, 1)
   ncolors = 0
   
   // randomize node
   //val r = rand(n, 1)
   //val (v, seq) = sort2(r)
   
   // sequential node
   val seq = IMat(0 until n)
   
   // sequential coloring
   for(i <- 0 until n){
     var node = seq(i)
     var nbs = find(mrf(?, node))
     
     // mask forbidden colors
     var colorMap = iones(ncolors, 1)
     for(j <-0 until nbs.length){
       if(colors(nbs(j)) > -1){
         colorMap(colors(nbs(j))) = 0
       }
     }
     
     // find the legal color with least count
     var c = -1
     var minc = 999999
     for(k <-0 until ncolors){
       if((colorMap(k) > 0) && (colorCount(k) < minc)){
         c = k
         minc = colorCount(k)
       }
     }
     
     // in case no legal color, increase ncolor 
     if(c == -1){
    	 c = ncolors
    	 ncolors = ncolors + 1
     }
     
     colors(node) = c
     colorCount(c) += 1
     
   }
   
    println("number of colors used: %d" format ncolors)
    for(i <- 0 until ncolors){
       println("color %d: %d" format (i, colorCount(i)))
    }
    println("total: %d" format (sum(colorCount(0 until ncolors)).v))
    
    colors
 }

}