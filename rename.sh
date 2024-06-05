#!/bin/bash

if command -v gsed >/dev/null 2>&1; then
  sed_cmd="gsed"
else
  sed_cmd="sed"
fi

read -ep "Package name: " PACKAGE
read -ep "Package description: " DESCRIPTION
read -ep "Package URL: " URL
echo $URL
read -ep "Author name [Rodrigo Martínez (brunneis)]: " AUTHOR
AUTHOR=${AUTHOR:-"Rodrigo Martínez (brunneis)"}

read -ep "Author email [dev@brunneis.com]: " EMAIL
EMAIL=${EMAIL:-"dev@brunneis.com"}

$sed_cmd -i "s~package_name~$PACKAGE~" package_name/__init__.py 
mv package_name/package_name.py package_name/$PACKAGE.py
mv package_name $PACKAGE
$sed_cmd -i "s~package_name~$PACKAGE~g" setup.py
$sed_cmd -i "s~description_value~$DESCRIPTION~g" setup.py
$sed_cmd -i "s~url_value~$URL~g" setup.py
$sed_cmd -i "s~author_value~$AUTHOR~g" setup.py
$sed_cmd -i "s~email_value~$EMAIL~g" setup.py

rm rename.sh
