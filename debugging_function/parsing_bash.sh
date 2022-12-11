#!/usr/bin/env bash




usage()
{
  echo "Usage: Parser [ -s | --k_in START]
                        [ -f | --k_fin  END]
                        [ -d | --dataset DATASET ]
                        [ -i | --id  experiment_name ]"
  exit 2
}

PARSED_ARGUMENTS=$(getopt -a -n parsing_bash -o s:f:d:i: --long k_in:,k_fin:,dataset:,id: -- "$@")


VALID_ARGUMENTS=$?
if [ "$VALID_ARGUMENTS" != "0" ]; then
  usage
fi


echo "PARSED_ARGUMENTS is $PARSED_ARGUMENTS"
eval set -- "$PARSED_ARGUMENTS"
while :
do
  case "$1" in
    -s | --k_in)   k_in=$1   ; shift 1   ;;
    -f | --k_fin)    k_fin=$2   ; shift 1 ;;
    -d | --dataset) dataset="$3" ; shift 1;;
    -i | --id)   id="$4"  ; shift 1  ;;
    # -- means the end of the arguments; drop this, and break out of the while loop
    --) shift; break ;;
    # If invalid options were passed, then getopt should have reported an error,
    # which we checked as VALID_ARGUMENTS when getopt was called...
    *) echo "Unexpected option: $1 - this should not happen."
       usage ;;
  esac
done


echo "k_initial : $k_in"
echo "k_final : $k_fin"
echo "dataset : $dataset"
echo "ID : $id"