#!/usr/bin/env bash
#SBATCH --partition=xxl
#SBATCH --array=0-999
#SBATCH --job-name='dl0_to_dl1'
#SBATCH --output=/home/cyril.alispach/logs/%x-%A_%a.out
#SBATCH --mem=2G
#SBATCH --time=4-00:00:00

source ~/.bashrc
conda activate lst-dev
VERSION='v2'

# LOG_FILE=$HOME'/logs/'$SLURM_JOB_NAME'-'$SLURM_JOB_ID'_'$SLURM_ARRAY_TASK_ID'_tee.out'

OUTPUT_DIR='/fefs/aswg/workspace/'$USER'/DL1/'$VERSION'/'
mkdir -p $OUTPUT_DIR
CONFIG_FILE=$HOME'/ctasoft/lstchain/lstchain/data/lstchain_lhfit_config.json'

INPUT_PATH='/fefs/aswg/data/mc/DL0/20190415/'
pointing='south_pointing/'
GAMMA_FILES=$INPUT_PATH'gamma/'$pointing'*.simtel.gz'
GAMMA_FILES=($(ls $GAMMA_FILES))
PROTON_FILES=$INPUT_PATH'proton/'$pointing'*simtel.gz'
PROTON_FILES=($(ls $PROTON_FILES))
GAMMA_DIFFUSE_FILES=$INPUT_PATH'gamma-diffuse/'$pointing'*simtel.gz'
GAMMA_DIFFUSE_FILES=($(ls $GAMMA_DIFFUSE_FILES))
FILES=("${GAMMA_FILES[@]}" "${PROTON_FILES[@]}" "${GAMMA_DIFFUSE_FILES[@]}")
# FILES=("${GAMMA_FILES[@]}")
echo ${#PROTON_FILES[@]} "proton files," ${#GAMMA_FILES[@]} "gamma files," ${#GAMMA_DIFFUSE_FILES[@]}  "gamma gamma_diffuse files"
echo "Total" ${#FILES[@]} "files"

STEP=1000
INDEX=$(( $SLURM_ARRAY_TASK_ID + $STEP * $1 ))
FILE=${FILES[$INDEX]}

echo "Processing" $FILE
OUTPUT_FILE=$(basename $FILE)
OUTPUT_FILE=${OUTPUT_FILE:0:-10}'.h5'
OUTPUT_FILE=$OUTPUT_DIR'dl1_'$OUTPUT_FILE

rm $OUTPUT_FILE
echo "Writing to "$OUTPUT_FILE
python -u lstchain/scripts/lstchain_mc_r0_to_dl1.py --input-file=$FILE --output=$OUTPUT_DIR --config=$CONFIG_FILE
echo "DL1 file saved to" $OUTPUT_FILE
exit