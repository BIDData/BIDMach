

BIDMach is a very fast machine learning library. Check the latest <b><a href="https://github.com/BIDData/BIDMach/wiki/Benchmarks">benchmarks</a></b>

The github distribution contains source code only. To get the libraries for your platform, run `./getdevlibs.sh` from this directory. Then you can run bidmach with `./bidmach`. Also, you can run the iScala notebook using `./bidmach notebook`.

**Update 01/22/2017**: `./getdevlibs.sh` is obsolete, so please see the <a href="https://github.com/BIDData/BIDMach/wiki/Installing-and-Running">installation instructions here using maven</a>.

You can build the Java/Scala main jar with sbt (included). There are build scripts for both Scala 2.10 and 2.11. Copy the appropriate one into `build.sbt`, and then do `./sbt package`. You can then run bidmach with `./bidmach` (you still need to download the libraries some of which are native).

You can also download an executable bundle from <b><a href="http://bid2.berkeley.edu/bid-data-project/download/">here</a></b>. You will need the libs from there in order to build from a git branch. We use a lot of native code which isn't all available from repos, and you will save a lot of time and headaches by grabbing compiled versions.

The main project page is <b><a href="http://bid2.berkeley.edu/bid-data-project/">here</a></b>.

Documentation is <b><a href="https://github.com/BIDData/BIDMach/wiki">here in the wiki</a></b>

<b>New</b> BIDMach has a <b><a href="https://groups.google.com/forum/#!forum/bidmach-users-group">discussion group</a></b> on Google Groups.

BIDMach is a sister project of BIDMat, a matrix library, which is 
<b><a href="https://github.com/BIDData/BIDMat">also on github</a></b>



[![Join the chat at https://gitter.im/BIDData/BIDMach](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BIDData/BIDMach?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)