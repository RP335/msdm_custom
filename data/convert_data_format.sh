DATA_DIR="$(dirname $0)"
STEMS="speech non_speech"
OUT_DIR="$DATA_DIR/speech_n_speech_data"

for STEM in $STEMS
do
	STEM_DIR="$DATA_DIR/${STEM}_22050_1"
	
	for SPLIT_DIR in $STEM_DIR/*
	do
		SPLIT=$(basename $SPLIT_DIR)
		
		for TRACK_FILE in "$SPLIT_DIR/"*.wav
		do 
			TRACK=$(basename "$TRACK_FILE" .wav)

			SOURCE_FILE="$TRACK_FILE"
			TARGET_FILE="$OUT_DIR/$SPLIT/$TRACK/$STEM.wav"

			echo "hard-linking file \"${SOURCE_FILE}\" to \"${TARGET_FILE}\""
			
			mkdir -p "$OUT_DIR/$SPLIT/$TRACK"
			ln "${SOURCE_FILE}" "${TARGET_FILE}" 
		done
	done
done
