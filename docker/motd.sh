#!/bin/bash
# DLC docker message of the day

check_root() {
  if [[ $(id -un) == "root" ]]; then
    echo !!! Warning: !!!
    echo It seems like you run the container as root, which is not recommended.
    echo If this is not intended, make sure to launch the container with the
    echo '-u $(id -u)' flag set or use the helper scripts in the main
    echo DLC repo, https://github.com/DeepLabCut/DeepLabCut
  fi
}

print_version() {
  DLC_VERSION=$(
	python3 -c "import deeplabcut; print(deeplabcut.__version__)" \
	2>/dev/null \
	|| echo [unknown version] 
  )
  echo Welcome to DeepLabCut ${DLC_VERSION}!
  echo You are running the container as user $(id -un) \($(id -u)\).
}

cat << "EOF"
                    .--,       .--,
                    ( (  \.---./  ) )
                     '.__/o   o\__.'
                       `{=  ^  =}´
                         >  u  <
 ____________________.""`-------`"".______________________  
\   ___                   __         __   _____       __  /
/  / _ \ ___  ___  ___   / /  ___ _ / /  / ___/__ __ / /_ \
\ / // // -_)/ -_)/ _ \ / /__/ _ `// _ \/ /__ / // // __/ /
//____/ \__/ \__// .__//____/\_,_//_.__/\___/ \_,_/ \__/  \
\_________________________________________________________/
                       ___)( )(___ `-.___. 
                      (((__) (__)))      ~`

EOF

print_version
check_root
