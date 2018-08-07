Set-StrictMode -Version Latest

$memSize = "-Xmx14G"
$bidmachRoot = $PSScriptRoot
$libDir = "${bidmachRoot}\lib"

$loggingConf = "${bidmachRoot}\conf\logging.conf"
$logOpts = if (Test-Path $loggingConf) { "-Djava.util.logging.config.file=`"$loggingConf`"" } else { "" }
$env:JAVA_OPTS = "$memSize -Xms128M -Dfile.encoding=UTF-8 $logOpts $env:JAVA_OPTS" # Set as much memory as possible

$bidmatVersion = (Get-ChildItem -Filter "BIDMat-*-cpu-*.jar" $libDir | Select-Object -First 1).Name `
                   -replace "BIDMat-(.+)-cpu.*",'$1'

$arch = if ($env:PROCESSOR_ARCHITECTURE -eq "AMD64") {"x86_64"} else {"x86"}
$bidmachJars = Get-ChildItem -Filter "*.jar" $libDir | % { $_.Name }
$bidmachLibs = New-Object System.Collections.ArrayList<String>
$bidmachLibs.Add("${bidmachRoot}\target\BIDMach-${bidmatVersion}.jar") > $null
foreach ($lib in $bidmachJars)
{
    if (-not ($lib -like "IScala*") -and -not ($lib -like "scala*"))
    {
        $bidmachLibs.Add("${libDir}\${lib}") > $null
    }
}
$bidmachLibs = $bidmachLibs.ToArray() -join ";"

$toolLibs = "${bidmachRoot}\conf;${env:JAVA_HOME}\lib\tools.jar;${libDir}\IScala-1.0.0.jar"

$allLibs = "${toolLibs};${bidmachLibs}"

$userArgs = New-Object System.Collections.ArrayList<String>
for ($i = 1; $i -lt $args.Count; $i++)
{
    $userArgs.Add("-Duser.arg$($i - 1)=`"$($args[$i])`"") > $null
}

if ($args.Count -ge 1 -and $args[0] -eq "notebook")
{
    if ($args.Count -gt 1)
    {
        $kernelCmd = "[\`"java\`", \`"-cp\`", r\`"${allLibs}\`", \`"${memSize}\`", \`"-Xms128M\`", \`"-Dfile.encoding=UTF-8\`", \`"org.refptr.iscala.IScala\`", \`"--profile\`", \`"{connection_file}\`", \`"--parent\`", \`"$($args[1..($args.Count - 1)] -join " ")\`"]"
    }
    else
    {
        $kernelCmd = "[\`"java\`", \`"-cp\`", r\`"${allLibs}\`", \`"${memSize}\`", \`"-Xms128M\`", \`"-Dfile.encoding=UTF-8\`", \`"org.refptr.iscala.IScala\`", \`"--profile\`", \`"{connection_file}\`", \`"--parent\`"]"
    }
    ipython notebook --profile=scala --KernelManager.kernel_cmd="${kernelCmd}"
}
else
{
    $firstArg = if ($args.Count -gt 0) { @($args[0]) } else { @() }
    $scalaArgs = "-toolcp","`"${toolLibs}`"","-Dscala.repl.maxprintstring=8000" + $userArgs + "-nobootcp","-cp","`"${allLibs}`"","-Yrepl-sync","-i","`"${libDir}\bidmach_init.scala`"" + $firstArg
    &"${bidmachRoot}\scripts\scala\scala.bat" $scalaArgs
}
