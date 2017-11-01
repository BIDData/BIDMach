

BIDMach is a very fast machine learning library. Check the latest <b><a href="https://github.com/BIDData/BIDMach/wiki/Benchmarks">benchmarks</a></b>

The github distribution contains source code only. You also need a jdk 8, an installation of NVIDIA CUDA 8.0 (if you want to use a GPU) and CUDNN 5 if you plan to use deep networks. For building you need <a href="https://maven.apache.org/docs/history.html">maven 3.X</a>.

After doing <code>git clone</code>, cd to the BIDMach directory, and build and install the jars with <code>mvn install</code>. You can then run bidmach with `./bidmach`. More details on installing and running are available <b><a href="https://github.com/BIDData/BIDMach/wiki/Installing-and-Running">here</a></b>.

The main project page is <b><a href="http://bid2.berkeley.edu/bid-data-project/">here</a></b>.

Documentation is <b><a href="https://github.com/BIDData/BIDMach/wiki">here in the wiki</a></b>

<b>New</b> BIDMach has a <b><a href="https://groups.google.com/forum/#!forum/bidmach-users-group">discussion group</a></b> on Google Groups.

BIDMach is a sister project of BIDMat, a matrix library, which is 
<b><a href="https://github.com/BIDData/BIDMat">also on github</a></b>

BIDData also has a project for deep reinforcement learning. <b><a href="https://github.com/BIDData/BIDMach_RL">BIDMach_RL</a></b> contains state-of-the-art implementations of several reinforcement learning algorithms.


- - - - -

Note for DSS2 team:

# DSS BIDMach Branch Technical Detail

Hmm... No idea what to put here. 
Blablabla Poopoo....??

## Network Hierachy

Assuming dealing with 2D grid.

                MM
         RM             CM
      A1     A2     A3      A4  

### Workflow for MM

```
Boot up MM

for each online node x
  node x informs MM of its existence;
  
if (# of online nodes >= threshold){
  MM.generateGrid();
}

if (additional node x joins the cluster later){
  x informs MM of its existence.
  MM changes the membership based on the GRID algorithm.
}
```

### Workflow for RM/CM
```
Send AllReduce_Start msg to all its nodes.
Repeatedly doing AllReduce until the terminating condition is met.

//Stop Training
if (RM/CM receives Stop_Training msg from the MM){
  RM/CM informs all the nodes to stop calculating gradient.
  All nodes are only doing scattering and aggregating.
}

//Stop AllReducing
if (RM/CM receives Stop_AllReducing msg from the MM){
  RM/CM stops the allReduce on all its children nodes.
  Kill the nodes gracefully.
  After terminating all the nodes, terminate itself.
}
```

### Workflow for AllReduce on Actor 
Assumption:
We are looking at Actor 2
There are total 10 Actors in the neighborhood.
```
Initialize itself with the same random weight //need to be explained in more detail (TODO)

while (it is not told to stop){
  //Calculate gradient
  Calculate its weight gradient based on its training data and the old weight.

  //Split & Scatter
  Split the gradient into 10 chunks and scatter 9 chunks to its peers.

  //Aggregate & Average
  Gather the 2nd chunks from all its peers 
  Calculate the average of the 2nd chunks gradient.

  //Broadcast
  Broadcast the averaged 2nd chunk gradient to all the peers
  
  //Gather
  Gather the other broadcasted gradient (other chunks) from the peers
}
```

### Gradient Descent Algorithm
Credit to Lincoln, here is a hand-written clarification \
https://github.com/Yao-The-Beast/BIDMach/blob/dss2/remote-actor/Gradient_Descent.png


### Detail on How to Run the Cluster
Credit to Mick (our spiritual leader)

Starting and Logging in
- At local machine
	- Make sure we have environment variables set
	export AWS_ACCESS_KEY_ID="<>"
	export AWS_SECRET_ACCESS_KEY="<>"
	export CLUSTER="bidcluster1"
-	Make sure python environment version is 2.7
- cd ~/BIDMach/scripts, start cluster by executing `./cluster_start.sh`
	- If you have problem with rsa key, modify the script to use the right file name
	- Wait for the script to finish, verify on Amazon console that all nodes (master/workers) are up. 
- to log-in to master node, execute `./cluster_login.sh`. 

Upload code and deploy
- From your local machine, push latest code on a specific branch
- On master once you log-in, go to `/code/BIDMach`, and pull latest code: "git fetch origin; git checkout <branch feature>; git pull"
- Now we want to compile that using mvn at the same directory: "mvn clean install". Note this only executes in master instance
- To compile this latest code on workers. At `/code/BIDMach/scripts` execute `./runall.sh "cd /code/BIDMach; git fetch origin; git checkout <branch feature>; git pull; mvn clean install"

Run code
- Assuming that we have 2 sets of code, one for master and the other for workers
	- On master node, go to `/code/BIDMach`, and execute the compiled bidmach by `./bidmach scripts/<.ssc script>`
	- Now we want to start workers at `/code/BIDMach/script`, execute `./start_workers.sh <.ssc script>`
  - To verify running workers, login to the slaves via ssh, execute `ps aux | grep java` to find the running script
	- To stop BIDMach processes on workers, `./stop_workers.sh`. 

- Now if we want to see output from workers, one work-around is to use `./runall.sh <>` to start the worker scripts. We should figure out a way to easily output of inidivudal worker

Stop Cluster
Locally execute 'cluster_stop.sh'


### TO DO 
So many things to do..........


### Q/A:

1) What is the definition of the Row/Column Master Synchronization? 
Is it block or non-blocking if one doesn't recieve enough packets.
soft threshold? dont need 100% to start allreduce

It is non-blocking, as master also has threshold of getting completion message from worker before moving on to next iteration.
In fact we have 3 thresholds in the system:
	- Worker: Gathering threshold, before reducing - do run-time adding before boardcasting
	- Worker: AllGathering threshold, before sending completed to Row master
	- Row/Col Master: completion message threshold, before moving on to new timestep


2) How multiple buffers on a single node works?

Assuming that the "caller" of AllReduce api is the entity that wants to reduce its data with other entities. In our case, it is the actor that hosts the training model, computing the gradient. This actor will do so by asking another actor - all-reducer actor - by sending a message to him. 

Now in all-reduce actor, we need temporal buffers to store results from other peer workers (all-reducer actors on other machines) who are faster. The number of future buffer we need depends on the maximal lag we allow any worker in the system to be. This lag should be configurable.

When the worker's lag is more than the acceptable level, the all-reduce actor will send back message to the "caller" with any intermediate results it has collected so far, and then it will purge these buffers. 

For example, when it is at time T, but it nows recieves message from master T+N, it will sends N buffers back to clients.

3) What are problem simplification for now?

We can consider working only all-reduce api that gives summuation with data point counts (so that caller can do average themselves).
The random mapping of which worker is responsible for which data chunk is going to be outside this all-reduce layer
