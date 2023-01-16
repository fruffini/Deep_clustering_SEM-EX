#!/bin/bash
############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Add description of the script functions here."
   echo
   echo "Syntax: scriptTemplate [-g|h|v|V]"
   echo "options:"
   echo "g     Print the GPL license notification."
   echo "h     Print this Help."
   echo "v     Verbose mode."
   echo "V     Print software version and exit."
   echo
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################

# Set variables
# shellcheck disable=SC1068
Hor=""
Elast=""
Aff=""


############################################################
# Process the input options. Add options as needed.        #
############################################################
# Get the options
while getopts ":h:f:a:e:q:r:p:" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      p)
         p1="--horizontal_flip";;
      r)
         p2="--horizontal_flip";;
      q)
         p3="--horizontal_flip";;
      f) # Enter surname
         Hor="--horizontal_flip";;
      a) # Enter Affine
         Aff="--affine";;
      e) # Enter Affine
         Elast="--elastic_deform";;
     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done

echo "Affine $Aff!"
echo "Elastic $Elast!"
echo "Horizontal flip $Hor!"

echo "print my $1"

