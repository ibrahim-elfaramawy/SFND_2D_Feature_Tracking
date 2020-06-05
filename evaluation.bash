cd build
declare -a detector=("SHITOMASI" "HARRIS" "FAST" "BRISK" "ORB" "AKAZE")
declare -a descriptor=("BRISK" "BRIEF" "ORB" "FREAK" "AKAZE")
for i in "${detector[@]}"
do
   for j in "${descriptor[@]}"
   do
      ./2D_feature_tracking "$i" "$j"
   done
done
