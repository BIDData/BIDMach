package BIDMach.ui

import BIDMat.{Mat,FMat,Image,MatFunctions}
import java.util.concurrent.Future;
import BIDMach.Learner
import javax.swing._;
import com.mxgraph.swing.mxGraphComponent;
import com.mxgraph.view.mxGraph;
import scala.collection.immutable.HashMap
import java.awt.event._


class GraphFrame(val layers:Array[HashMap[String,Any]])extends JFrame("Graph"){
    val graph = new mxGraph();
    val _parent = graph.getDefaultParent();
    graph.getModel().beginUpdate();
    var i = 0
    val nodes = layers.map(d=>{
        graph.insertVertex(_parent, null, d("name"), 20, 20 + 40*i, 200,30)
        i+=1
    })
        //            graph.insertEdge(parent2, null, "Edge", v1, v2);
    graph.getModel().endUpdate();

    val graphComponent = new mxGraphComponent(graph) {
        override def installDoubleClickHandler(){
            getGraphControl()
                .addMouseListener(new MouseAdapter(){
                    override def mouseClicked(e: MouseEvent){
                        if(e.getClickCount()==2){
                            val cell = getCellAt(e.getX(), e.getY());
                            if (cell != null)
                                Plot.plot_layer_code(graph.getLabel(cell))//.split("\"")(1));
                        }
                    }
                })
        }
    }
    getContentPane().add(graphComponent);  
    graphComponent.getGraphControl()
        .addMouseListener(new MouseAdapter()
                          {
                              override def mouseReleased(e:MouseEvent)
                              {
                                  val  cell = graphComponent.getCellAt(e.getX(), e.getY());

                                  if (cell != null)
                                  {
                                      println("cell="+graph.getLabel(cell));
                                  }
                              }                              
                          }); 
}

object Plot{
    var tot = 0
    var interval = 1
    val useWeb = false
        
        
    def getMsg(id:String, ipass:Int, names:List[String], values:Array[Float]) = {
         "{\"id\":\"%s\",\"ipass\":%d,\"names\":[%s],\"values\":[%s]}" format(id,ipass,names.map('"'+_+'"').reduce(_+","+_),values.map('"'+_.toString+'"').reduce(_+","+_))                    
    }
        
    def plot(fn:()=>Mat,names:List[String]=List("data")) {
        /**if (server == null)
            start()
        tot += 1
        val id = "Plot_" + tot
        var ipass = 0
        val runme = new Runnable {
          override def run() = {
            while (true) {
                Thread.sleep((1000*interval).toLong);
                val mat = MatFunctions.cpu(fn());
                ipass += 1
                val message = getMsg(id,ipass,names,FMat(mat).data)
                server.send(message)
//                println(message)
            }
          }
        }
        Image.getService.submit(runme);*/
    }
    
    def plot_web(learner:Learner) = {        
        /**
        if (server == null)
            start()
        server.send(utils.getModelGraph(learner)._2.toString)*/
        null
    }
    
    def plot_jframe(learner:Learner) = {
        val obj = utils.getModelGraph(learner)
        val layers = obj("layers").asInstanceOf[Array[HashMap[String,Any]]]
        val frame = new GraphFrame(layers)
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        frame.setVisible(true);
        frame
    }
    
    def plot(learner:Learner):GraphFrame = {
        if (useWeb) plot_web(learner)
        else plot_jframe(learner)
    }

    def plot_code(filename:String) = {
        val title = filename.split("/").reverse.head
        val frame = new JFrame(title);
        val textArea = new JTextArea(60, 40);
        val scrollPane = new JScrollPane(textArea);
        val s=scala.io.Source.fromURL(filename).getLines.mkString("\n");
        textArea.append(s)        
        frame.getContentPane().add(scrollPane)
        //frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        frame.setVisible(true);
        frame
    }
    
    def plot_layer_code(layerName: String) = {
        val bidmachURL = "https://raw.githubusercontent.com/BIDData/BIDMach/master/src/main/"
        plot_code(bidmachURL + "scala/BIDMach/networks/layers/" + layerName + ".scala")
    }
}

